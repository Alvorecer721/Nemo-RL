# Copyright (c) 2026, the Apertus project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the online-DPO driver's pure logic (pair selection + batch build).

``online_dpo.py`` imports the heavy GRPO/torchdata stack at module load, so we exec
its source with those imports stripped (as ``test_mpo_loss.py`` does for the locked
runtime) and inject a fake ``preference_collate_fn`` that captures the constructed
``PreferenceDatumSpec`` list. That isolates the selection/trimming/masking logic —
the real collate is already covered upstream. A torch-only surrogate locks the
reference-update snapshot contract that the worker methods implement.
"""

import ast
from pathlib import Path

import pytest
import torch

ONLINE_DPO_SRC = Path(__file__).resolve().parents[1] / "online_dpo.py"


def _load_online_dpo(fake_collate):
    """Exec online_dpo.py with nemo_rl/torchdata imports stripped; inject a fake collate.

    The strip also removes the (stdlib-only, importable) nemo_rl_apertus.online_judge
    import, so we inject the real last_assistant_index + judge_inputs_from_conversation
    the driver depends on.
    """
    from nemo_rl_apertus.online_judge import (
        judge_inputs_from_conversation,
        last_assistant_index,
    )

    tree = ast.parse(ONLINE_DPO_SRC.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module.startswith("nemo_rl") or node.module.startswith("torchdata"))
        )
    ]
    namespace: dict = {
        "preference_collate_fn": fake_collate,
        "last_assistant_index": last_assistant_index,
        "judge_inputs_from_conversation": judge_inputs_from_conversation,
    }
    exec(compile(ast.Module(body=tree.body, type_ignores=[]), str(ONLINE_DPO_SRC), "exec"), namespace)
    return namespace


def _capture_collate(data_batch, tokenizer, make_sequence_length_divisible_by, add_loss_mask):
    """Stand-in for preference_collate_fn: return the data_batch verbatim for inspection."""
    return {"_data_batch": data_batch, "add_loss_mask": add_loss_mask}


@pytest.fixture(scope="module")
def od():
    return _load_online_dpo(_capture_collate)


def _msg(role, content, token_ids, **extra):
    return {"role": role, "content": content, "token_ids": torch.tensor(token_ids), **extra}


def _rollout_log(prompt_ids, response_ids, response_text="resp"):
    """A post-rollout message log: prompt + assistant(+generation_logprobs) + judge env obs."""
    return [
        _msg("user", "prompt", prompt_ids),
        _msg(
            "assistant",
            response_text,
            response_ids,
            generation_logprobs=torch.zeros(len(response_ids)),
        ),
        _msg("environment", "judge_score=3.0", [9]),
    ]


# ---------------------------------------------------------------------------
# select_pairs_from_scores
# ---------------------------------------------------------------------------
def test_select_pairs_basic(od):
    pairs = od["select_pairs_from_scores"]([5.0, 1.0, 2.0, 4.0], num_generations=2, tie_eps=1e-4)
    assert pairs[0] == (0, 1, False)  # group 0: best=5.0@0, worst=1.0@1
    assert pairs[1] == (3, 2, False)  # group 1 (offset 2): best=4.0@3, worst=2.0@2


def test_select_pairs_degenerate_tie(od):
    pairs = od["select_pairs_from_scores"]([3.0, 3.0], num_generations=2, tie_eps=1e-4)
    assert pairs[0][2] is True  # max - min == 0 <= tie_eps -> degenerate


def test_select_pairs_group_of_four(od):
    pairs = od["select_pairs_from_scores"]([1.0, 9.0, 4.0, 2.0], num_generations=4, tie_eps=1e-4)
    assert pairs == [(1, 0, False)]  # best=9.0@1, worst=1.0@0


def test_select_pairs_tie_eps_inclusive_boundary(od):
    # exactly tie_eps apart -> degenerate (comparison is <=); base 0.0 keeps the diff exact
    assert od["select_pairs_from_scores"]([0.0, 1e-4], num_generations=2, tie_eps=1e-4)[0][2] is True
    # strictly more than tie_eps apart -> kept
    assert od["select_pairs_from_scores"]([0.0, 2e-4], num_generations=2, tie_eps=1e-4)[0][2] is False


# ---------------------------------------------------------------------------
# _mask_truncated_rollouts (truncation -> validity fold; gated by train_on_truncated)
# ---------------------------------------------------------------------------
def test_mask_truncated_default_masks_truncated(od):
    rb = {
        "truncated": torch.tensor([False, True, False]),
        "loss_multiplier": torch.tensor([1.0, 1.0, 1.0]),
    }
    od["_mask_truncated_rollouts"](rb, False)
    assert rb["loss_multiplier"].tolist() == [1.0, 0.0, 1.0]  # truncated rollout masked out


def test_mask_truncated_train_on_truncated_is_noop(od):
    rb = {
        "truncated": torch.tensor([False, True]),
        "loss_multiplier": torch.tensor([1.0, 1.0]),
    }
    od["_mask_truncated_rollouts"](rb, True)
    assert rb["loss_multiplier"].tolist() == [1.0, 1.0]  # truncated stays eligible (-> can be rejected)


def test_mask_truncated_absent_flag_is_noop(od):
    rb = {"loss_multiplier": torch.tensor([1.0, 1.0])}
    od["_mask_truncated_rollouts"](rb, False)  # no "truncated" key -> nothing to fold
    assert rb["loss_multiplier"].tolist() == [1.0, 1.0]


def test_mask_truncated_preserves_prior_masking(od):
    # an already-masked rollout (e.g. over-length prompt) stays masked (multiply, not overwrite)
    rb = {
        "truncated": torch.tensor([False, False]),
        "loss_multiplier": torch.tensor([0.0, 1.0]),
    }
    od["_mask_truncated_rollouts"](rb, False)
    assert rb["loss_multiplier"].tolist() == [0.0, 1.0]


# ---------------------------------------------------------------------------
# _trim_to_last_assistant / _clean_message
# ---------------------------------------------------------------------------
def test_trim_drops_env_obs_and_extra_keys(od):
    trimmed = od["_trim_to_last_assistant"](_rollout_log([1, 2], [3, 4]))
    assert [m["role"] for m in trimmed] == ["user", "assistant"]  # env obs dropped
    # cleaned to exactly role/content/token_ids (generation_logprobs removed)
    assert set(trimmed[-1].keys()) == {"role", "content", "token_ids"}


def test_trim_returns_none_without_assistant(od):
    log = [_msg("user", "p", [1, 2])]
    assert od["_trim_to_last_assistant"](log) is None


def test_trim_multi_turn_keeps_context_drops_trailing_env(od):
    # multi-turn prompt context + a trailing judge env observation
    log = [
        _msg("user", "u1", [1]),
        _msg("assistant", "a1", [2]),
        _msg("user", "u2", [3]),
        _msg("assistant", "a2", [4]),  # last assistant turn (the DPO-trained span)
        _msg("environment", "judge_score=3.0", [9]),
    ]
    trimmed = od["_trim_to_last_assistant"](log)
    # full conversation up to & incl. the last assistant turn is kept (earlier assistant too);
    # the trailing env observation is dropped so only_unmask_final unmasks a2, not the env turn.
    assert [m["role"] for m in trimmed] == ["user", "assistant", "user", "assistant"]
    assert trimmed[-1]["token_ids"].tolist() == [4]


# ---------------------------------------------------------------------------
# build_preference_batch
# ---------------------------------------------------------------------------
def _repeated_batch(message_logs, loss_multiplier):
    # build_preference_batch only indexes these two keys
    return {"message_log": message_logs, "loss_multiplier": torch.tensor(loss_multiplier)}


def test_build_preference_batch_selection_and_masking(od):
    # 2 prompts, R=2. group0: clear winner; group1: tie -> degenerate/masked.
    logs = [
        _rollout_log([1, 2], [3, 4], "good"),  # g0 r0
        _rollout_log([1, 2], [5, 6], "bad"),  # g0 r1
        _rollout_log([7, 8], [10], "a"),  # g1 r0
        _rollout_log([7, 8], [11], "b"),  # g1 r1
    ]
    scores = [5.0, 1.0, 3.0, 3.0]
    batch, metrics = od["build_preference_batch"](
        _repeated_batch(logs, [1.0, 1.0, 1.0, 1.0]),
        scores,
        num_generations=2,
        tie_eps=1e-4,
        tokenizer=None,
        make_sequence_length_divisible_by=1,
    )
    data_batch = batch["_data_batch"]
    assert len(data_batch) == 2

    # group 0: chosen = the high-scoring rollout (token_ids [3,4]), rejected = [5,6]
    g0 = data_batch[0]
    assert g0["loss_multiplier"] == 1.0
    assert g0["message_log_chosen"][-1]["token_ids"].tolist() == [3, 4]
    assert g0["message_log_rejected"][-1]["token_ids"].tolist() == [5, 6]
    # length = prompt(2) + response(2)
    assert g0["length_chosen"] == 4

    # group 1: tie -> masked
    assert data_batch[1]["loss_multiplier"] == 0.0
    assert metrics["num_degenerate_pairs"] == 1.0
    assert metrics["num_pairs"] == 2.0
    assert metrics["chosen_reward_mean"] == pytest.approx(5.0)
    assert metrics["rejected_reward_mean"] == pytest.approx(1.0)


def test_build_preference_batch_masks_missing_assistant_turn(od):
    # A rollout that produced no assistant turn -> _trim returns None -> pair masked,
    # falls back to cleaned raw logs without crashing.
    good = _rollout_log([1, 2], [3, 4])
    no_assistant = [_msg("user", "prompt", [1, 2])]  # generation produced nothing
    batch, metrics = od["build_preference_batch"](
        _repeated_batch([good, no_assistant], [1.0, 1.0]),
        [5.0, 1.0],
        num_generations=2,
        tie_eps=1e-4,
        tokenizer=None,
        make_sequence_length_divisible_by=1,
    )
    assert batch["_data_batch"][0]["loss_multiplier"] == 0.0
    assert metrics["num_degenerate_pairs"] == 1.0
    # the rejected (no-assistant) log fell back to its cleaned raw form (still valid tensors)
    assert batch["_data_batch"][0]["message_log_rejected"] is not None


def test_build_preference_batch_masks_invalid_prompt(od):
    logs = [_rollout_log([1, 2], [3, 4]), _rollout_log([1, 2], [5, 6])]
    # dataset marked these rollouts invalid (loss_multiplier 0) -> pair masked even if scores differ
    batch, metrics = od["build_preference_batch"](
        _repeated_batch(logs, [0.0, 0.0]),
        [5.0, 1.0],
        num_generations=2,
        tie_eps=1e-4,
        tokenizer=None,
        make_sequence_length_divisible_by=1,
    )
    assert batch["_data_batch"][0]["loss_multiplier"] == 0.0
    assert metrics["num_degenerate_pairs"] == 1.0


# ---------------------------------------------------------------------------
# build_rollout_log (per-rollout JSONL dump records)
# ---------------------------------------------------------------------------
def test_build_rollout_log_all_and_selection(od):
    logs = [
        _rollout_log([1, 2], [3, 4], "good"),  # g0 r0 (best -> chosen)
        _rollout_log([1, 2], [5, 6], "bad"),  # g0 r1 (worst -> rejected)
        _rollout_log([7, 8], [10], "a"),  # g1 r0 (tie -> degenerate)
        _rollout_log([7, 8], [11], "b"),  # g1 r1 (tie -> degenerate)
    ]
    scores = [5.0, 1.0, 3.0, 3.0]
    out = od["build_rollout_log"](
        {"message_log": logs}, scores, num_generations=2, tie_eps=1e-4, num_logged_rollouts=-1
    )
    assert len(out["selection"]) == 4  # -1 -> all rollouts
    assert out["selection"][0] == "chosen" and out["selection"][1] == "rejected"
    # exact tie: best==worst==first index -> that index "degenerate", the other "unused"
    assert out["selection"][2] == "degenerate" and out["selection"][3] == "unused"
    assert out["judge_score"] == [5.0, 1.0, 3.0, 3.0]
    assert out["group"] == [0, 0, 1, 1]
    assert out["rollout_in_group"] == [0, 1, 0, 1]
    # prompt/response are the judge-scored spans (env obs dropped, last assistant = response)
    assert out["response"][0] == "good" and out["prompt"][0] == "prompt"


def test_build_rollout_log_uses_clean_judge_prompt(od):
    # extra_env_info[i]["judge_prompt"] (clean prompt) wins over the rollout's rendered turns
    logs = [_rollout_log([1, 2], [3, 4], "good"), _rollout_log([1, 2], [5, 6], "bad")]
    rb = {
        "message_log": logs,
        "extra_env_info": [{"judge_prompt": "clean A"}, {"judge_prompt": "clean B"}],
    }
    out = od["build_rollout_log"](
        rb, [5.0, 1.0], num_generations=2, tie_eps=1e-4, num_logged_rollouts=-1
    )
    assert out["prompt"] == ["clean A", "clean B"]  # clean judge_prompt, not the rendered turn
    assert out["response"] == ["good", "bad"]


def test_build_rollout_log_first_n_and_off(od):
    logs = [_rollout_log([1, 2], [3, 4]), _rollout_log([1, 2], [5, 6])]
    first1 = od["build_rollout_log"](
        {"message_log": logs}, [5.0, 1.0], num_generations=2, tie_eps=1e-4, num_logged_rollouts=1
    )
    assert len(first1["selection"]) == 1  # N -> first N
    none = od["build_rollout_log"](
        {"message_log": logs}, [5.0, 1.0], num_generations=2, tie_eps=1e-4, num_logged_rollouts=0
    )
    assert len(none["selection"]) == 0  # 0 -> none


def test_build_rollout_log_thinking_column(od):
    # the per-prompt thinking flag the processor stashed in extra_env_info is surfaced; absent -> None
    logs = [_rollout_log([1, 2], [3, 4], "good"), _rollout_log([1, 2], [5, 6], "bad")]
    rb = {"message_log": logs, "extra_env_info": [{"enable_thinking": True}, {}]}
    out = od["build_rollout_log"](
        rb, [5.0, 1.0], num_generations=2, tie_eps=1e-4, num_logged_rollouts=-1
    )
    assert out["thinking"] == [True, None]


def test_build_rollout_log_redecodes_with_tokenizer(od):
    # on the reasoning-aspect path a tokenizer is passed so the logged response matches the
    # special-token-preserving completion the judge scored (not the stripped content)
    class _Tok:
        def decode(self, token_ids, skip_special_tokens=False):
            assert skip_special_tokens is False
            return "<|inner_prefix|>r<|inner_suffix|>resp"

    logs = [_rollout_log([1, 2], [3, 4], "resp"), _rollout_log([1, 2], [5, 6], "resp")]
    out = od["build_rollout_log"](
        {"message_log": logs}, [5.0, 1.0], num_generations=2, tie_eps=1e-4,
        num_logged_rollouts=-1, tokenizer=_Tok(),
    )
    assert out["response"][0] == "<|inner_prefix|>r<|inner_suffix|>resp"


# ---------------------------------------------------------------------------
# validate_online_dpo (held-out judge metrics; generation faked)
# ---------------------------------------------------------------------------
class _ScoreCol(list):
    def tolist(self):
        return list(self)


class _ValBatch:
    def repeat_interleave(self, r):
        return None  # the injected fake rollout ignores its input


def _install_fake_rollout(od, per_batch_scores):
    """Inject a fake run_multi_turn_rollout returning canned judge scores per val batch."""
    calls = {"i": 0}

    def fake_rollout(**kwargs):
        scores = per_batch_scores[calls["i"]]
        calls["i"] += 1
        return {"total_reward": _ScoreCol(scores)}, {"mean_gen_tokens_per_sample": 10.0}

    od["run_multi_turn_rollout"] = fake_rollout  # validate_online_dpo resolves it from the namespace


def test_validate_online_dpo_none_dataloader_returns_empty(od):
    assert od["validate_online_dpo"](None, None, None, None, 2, 1e-4, 2048, 1, None) == {}


def test_validate_online_dpo_aggregates(od):
    _install_fake_rollout(od, [[5.0, 1.0], [3.0, 3.0]])  # batch0 clear pair, batch1 tie
    m = od["validate_online_dpo"](
        object(), [_ValBatch(), _ValBatch()], None, None, 2, 1e-4, 2048, 1, None
    )
    assert m["num_samples"] == 4.0 and m["num_pairs"] == 2.0
    assert m["frac_valid_pairs"] == 0.5  # 1 of 2 groups non-degenerate
    assert m["judge_score_mean"] == pytest.approx(3.0)  # mean(5,1,3,3)
    assert m["mean_gen_tokens_per_sample"] == pytest.approx(10.0)


def test_validate_online_dpo_caps_batches(od):
    _install_fake_rollout(od, [[5.0, 1.0], [2.0, 4.0]])
    m = od["validate_online_dpo"](
        object(), [_ValBatch(), _ValBatch()], None, None, 2, 1e-4, 2048, 1, 1
    )
    assert m["num_samples"] == 2.0 and m["num_pairs"] == 1.0  # only the first val batch


def test_validate_online_dpo_all_degenerate_no_div_by_zero(od):
    _install_fake_rollout(od, [[3.0, 3.0]])  # single tied group -> no usable pair
    m = od["validate_online_dpo"](object(), [_ValBatch()], None, None, 2, 1e-4, 2048, 1, None)
    assert m["frac_valid_pairs"] == 0.0  # 0/1, guarded (no ZeroDivisionError)
    assert m["num_pairs"] == 1.0 and m["num_samples"] == 2.0
    assert m["judge_score_mean"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _should_run_validation (activate/deactivate gate; cadence)
# ---------------------------------------------------------------------------
def test_should_run_validation_deactivated(od):
    f = od["_should_run_validation"]
    # no val dataloader -> never fires, regardless of flags (safe no-op)
    assert f(None, 2, 2, True, True) is False
    # val data present but all cadence off -> never fires
    assert f(object(), 0, 5, False, False) is False


def test_should_run_validation_periodic(od):
    f = od["_should_run_validation"]
    dl = object()
    assert f(dl, 2, 2, False, False) is True  # total_steps % val_period == 0
    assert f(dl, 2, 3, False, False) is False  # 3 % 2 != 0
    assert f(dl, 2, 4, False, False) is True


def test_should_run_validation_at_end(od):
    f = od["_should_run_validation"]
    dl = object()
    assert f(dl, 0, 7, True, True) is True  # val_at_end + last step (period off)
    assert f(dl, 0, 7, False, True) is False  # last step but val_at_end disabled


# ---------------------------------------------------------------------------
# real preference_collate_fn contract (the seam the driver consumes; full runtime only)
# ---------------------------------------------------------------------------
def test_real_preference_collate_interleave_and_mask():
    # Runs only where the full runtime is importable (skips on login node: decord absent).
    collate_mod = pytest.importorskip("nemo_rl.data.collate_fn")
    preference_collate_fn = collate_mod.preference_collate_fn

    class _Tok:
        pad_token_id = 0

    def datum(idx, keep):
        return {
            "message_log_chosen": [_msg("user", "p", [1, 2]), _msg("assistant", "c", [3, 4])],
            "message_log_rejected": [_msg("user", "p", [1, 2]), _msg("assistant", "r", [5, 6])],
            "length_chosen": 4,
            "length_rejected": 4,
            "loss_multiplier": 1.0 if keep else 0.0,
            "idx": idx,
        }

    batch = preference_collate_fn(
        [datum(0, True), datum(1, False)],
        tokenizer=_Tok(),
        make_sequence_length_divisible_by=1,
        add_loss_mask=True,
    )
    # 2 pairs -> 4 interleaved rows (chosen, rejected, chosen, rejected)
    assert batch["input_ids"].shape[0] == 4
    # sample_mask is the per-row loss_multiplier, doubled per pair
    assert batch["sample_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# _build_dpo_loss_config (online_dpo block -> DPOLossConfig)
# ---------------------------------------------------------------------------
def test_build_dpo_loss_config_selects_keys(od):
    cfg = {
        "reference_update_freq": -1,  # online-only extra, must be dropped
        "tie_eps": 1e-4,  # online-only extra, must be dropped
        "reference_policy_kl_penalty": 0.25,
        "preference_loss_weight": 1.0,
        "sft_loss_weight": 0.5,
        "preference_average_log_probs": True,
        "sft_average_log_probs": False,
    }
    out = od["_build_dpo_loss_config"](cfg)
    assert out["reference_policy_kl_penalty"] == 0.25  # stock DPOLossConfig key, verbatim
    assert out["preference_loss_weight"] == 1.0
    assert out["sft_loss_weight"] == 0.5
    assert out["preference_average_log_probs"] is True
    assert out["sft_average_log_probs"] is False
    # online-only extras are not forwarded to the loss config
    assert "reference_update_freq" not in out and "tie_eps" not in out


def test_build_dpo_loss_config_sft_average_falls_back(od):
    # sft_average_log_probs absent -> mirrors preference_average_log_probs
    out = od["_build_dpo_loss_config"](
        {
            "reference_policy_kl_penalty": 0.1,
            "preference_loss_weight": 1.0,
            "sft_loss_weight": 0.0,
            "preference_average_log_probs": True,
        }
    )
    assert out["sft_average_log_probs"] is True
    # explicit value wins over the fallback
    out2 = od["_build_dpo_loss_config"](
        {
            "reference_policy_kl_penalty": 0.1,
            "preference_loss_weight": 1.0,
            "sft_loss_weight": 0.0,
            "preference_average_log_probs": True,
            "sft_average_log_probs": False,
        }
    )
    assert out2["sft_average_log_probs"] is False


# ---------------------------------------------------------------------------
# setup() guards (fire before grpo_setup, so reachable in the stubbed namespace)
# ---------------------------------------------------------------------------
def _setup_master_config(num_generations=4, num_prompts=8, gbs=8, precision=None):
    return {
        "grpo": {
            "num_generations_per_prompt": num_generations,
            "num_prompts_per_step": num_prompts,
        },
        "policy": {
            "train_global_batch_size": gbs,
            "generation": {"vllm_cfg": {"precision": precision} if precision else {}},
        },
    }


def test_setup_requires_at_least_two_generations(od):
    with pytest.raises(AssertionError, match="num_generations_per_prompt >= 2"):
        od["setup"](_setup_master_config(num_generations=1), None, None, None)


def test_setup_requires_gbs_equals_num_prompts(od):
    with pytest.raises(AssertionError, match="train_global_batch_size == "):
        od["setup"](_setup_master_config(num_prompts=8, gbs=4), None, None, None)


def test_setup_rejects_fp8_rollout(od):
    with pytest.raises(AssertionError, match="fp8"):
        od["setup"](_setup_master_config(precision="fp8"), None, None, None)


# ---------------------------------------------------------------------------
# reference-update snapshot contract (surrogate for the worker methods)
# ---------------------------------------------------------------------------
def _megatron_style_snapshot(model):
    """Mirror megatron_policy_worker.update_reference_model's per-tensor CPU detach/copy."""
    snapshot = {}
    for name, item in model.state_dict().items():
        if isinstance(item, torch.Tensor):
            item = item.detach().to(device="cpu", copy=True)
        snapshot[name] = item
    return snapshot


def _dtensor_style_snapshot(model):
    """Mirror dtensor_policy_worker.update_reference_model (get_cpu_state_dict semantics).

    The contract is a detached CPU copy of the live weights; pin_memory (a CUDA-only
    perf detail of get_cpu_state_dict) is not part of the observable contract.
    """
    return {
        name: item.detach().to(device="cpu", copy=True)
        for name, item in model.state_dict().items()
    }


@pytest.mark.parametrize("snapshot", [_megatron_style_snapshot, _dtensor_style_snapshot])
def test_reference_update_snapshot_contract(snapshot):
    """Lock the snapshot contract BOTH worker update_reference_model methods satisfy.

    A detached, CPU copy of the current weights, where a later re-snapshot reflects
    the updated live weights (so ref==policy right after an update -> ln2 anchor).
    """
    model = torch.nn.Linear(4, 3)
    reference = snapshot(model)

    # snapshot is detached + on CPU
    assert not reference["weight"].requires_grad
    assert reference["weight"].device.type == "cpu"

    # mutate the live model: the old snapshot must NOT track the change (it's a copy)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    assert not torch.equal(reference["weight"], model.state_dict()["weight"].cpu())

    # re-snapshot (the "update"): reference now equals the new live weights
    reference = snapshot(model)
    assert torch.equal(reference["weight"], model.state_dict()["weight"].cpu())
