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
"""MPOLossFn unit tests (pure-tensor, CPU, no torch.distributed).

Exercises the real classes from this checkout: the stock
``nemo_rl.algorithms.loss.loss_functions`` and ``nemo_rl_apertus.mpo_loss``.
Plain-dict configs are used throughout; ``MPOLossFn`` normalizes them through
:class:`~nemo_rl_apertus.mpo_loss.MPOConfig`, and the one direct ``DPOLossFn``
construction wraps in the stock ``DPOLossConfig``.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import nemo_rl.algorithms.loss.loss_functions as stock_mod
from nemo_rl_apertus import mpo_loss as mpo_mod

DPOLossFn = stock_mod.DPOLossFn
MPOLossFn = mpo_mod.MPOLossFn

LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_delta_state():
    mpo_mod.reset_delta_states()
    yield
    mpo_mod.reset_delta_states()


def make_cfg(**overrides):
    cfg = {
        "reference_policy_kl_penalty": 0.5,
        "preference_loss_weight": 1.0,
        "sft_loss_weight": 0.0,
        "preference_average_log_probs": False,
        "sft_average_log_probs": False,
    }
    cfg.update(overrides)
    return cfg


def make_fixture(num_pairs=3, seq_len=6, seed=0, requires_grad=True):
    """Interleaved (chosen, rejected) fixture matching the stock layout.

    Returns (policy_logprobs[2B, T-1], data, global_valid_seqs, global_valid_toks).
    """
    g = torch.Generator().manual_seed(seed)
    n = 2 * num_pairs
    policy_logprobs = -torch.rand((n, seq_len - 1), generator=g, dtype=torch.float64)
    ref_logprobs = -torch.rand((n, seq_len), generator=g, dtype=torch.float64)
    token_mask = torch.ones((n, seq_len), dtype=torch.float64)
    token_mask[:, :2] = 0.0  # prompt tokens masked out
    token_mask[0, -1] = 0.0  # one ragged response length
    sample_mask = torch.ones(n, dtype=torch.float64)
    data = {
        "input_ids": torch.randint(0, 100, (n, seq_len), generator=g),
        "reference_policy_logprobs": ref_logprobs,
        "token_mask": token_mask,
        "sample_mask": sample_mask,
    }
    policy_logprobs.requires_grad_(requires_grad)
    global_valid_seqs = sample_mask.sum()
    global_valid_toks = (token_mask[:, 1:] * sample_mask[:, None]).sum()
    return policy_logprobs, data, global_valid_seqs, global_valid_toks


def implicit_rewards_reference(policy_logprobs, data, beta, average_log_probs=False):
    """Independent re-derivation of r = beta * (log pi_theta - log pi_0)."""
    tm = data["token_mask"][:, 1:]
    ref = data["reference_policy_logprobs"][:, :-1]
    logratios = ((policy_logprobs - ref) * tm).sum(-1)
    if average_log_probs:
        logratios = logratios / tm.sum(-1).clamp(min=1)
    return beta * logratios


def quality_loss_reference(rewards, sample_mask, gvs, delta):
    """Independent re-derivation of the per-pair BCO quality term."""
    r_c, r_r = rewards[::2], rewards[1::2]
    q_c = -F.logsigmoid(r_c - delta) * sample_mask[::2]
    q_r = -F.logsigmoid(-(r_r - delta)) * sample_mask[1::2]
    return (q_c.sum() + q_r.sum()) / (gvs / 2 + 1e-8)


# ---------------------------------------------------------------------------
# 1. equivalence: quality_loss_weight = 0  =>  bitwise-stock
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sft_weight", [0.0, 0.5])
@pytest.mark.parametrize("avg_logprobs", [False, True])
def test_weight_zero_bitwise_equivalence(sft_weight, avg_logprobs):
    cfg = make_cfg(
        sft_loss_weight=sft_weight, preference_average_log_probs=avg_logprobs
    )
    lp_s, data, gvs, gvt = make_fixture(seed=7)
    lp_m = lp_s.detach().clone().requires_grad_(True)

    stock_loss, stock_metrics = DPOLossFn(stock_mod.DPOLossConfig(**cfg))(
        lp_s, data, gvs, gvt
    )
    mpo_loss, mpo_metrics = MPOLossFn(dict(cfg, quality_loss_weight=0.0))(
        lp_m, data, gvs, gvt
    )

    assert torch.equal(stock_loss, mpo_loss)  # bitwise
    for key, value in stock_metrics.items():
        assert mpo_metrics[key] == value, key  # exact, incl. shared "loss"
    # extra MPO metrics present
    for key in (
        "quality_loss",
        "delta",
        "delta_count",
        "implicit_rewards_chosen_mean",
        "implicit_rewards_rejected_mean",
    ):
        assert key in mpo_metrics

    # gradients bitwise-identical too
    stock_loss.backward()
    mpo_loss.backward()
    assert torch.equal(lp_s.grad, lp_m.grad)


# ---------------------------------------------------------------------------
# 2. quality-term math vs hand-computed fixtures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("avg_logprobs", [False, True])
def test_quality_term_hand_computed_fixed_delta(avg_logprobs):
    beta, delta = 0.5, 0.3
    cfg = make_cfg(
        preference_loss_weight=0.0,
        quality_loss_weight=1.0,
        quality_delta_init=delta,
        preference_average_log_probs=avg_logprobs,
    )
    # eval-style inputs (requires_grad=False): delta stays pinned at its seed
    lp, data, gvs, gvt = make_fixture(seed=11, requires_grad=False)

    loss, metrics = MPOLossFn(cfg)(lp, data, gvs, gvt)

    rewards = implicit_rewards_reference(lp, data, beta, avg_logprobs)
    expected = quality_loss_reference(rewards, data["sample_mask"], gvs, delta)
    assert metrics["delta"] == delta
    assert torch.allclose(loss, expected, rtol=1e-12, atol=1e-12)
    assert metrics["quality_loss"] == pytest.approx(expected.item(), rel=1e-12)
    # total respects the weights: w_p = 0, w_q = 1, w_sft = 0
    assert metrics["loss"] == pytest.approx(metrics["quality_loss"], rel=1e-12)


def test_quality_term_two_pair_explicit_numbers():
    """Fully hand-computed 2-pair fixture (no helper reuse)."""
    beta = 2.0
    # one response token each; reference_policy_logprobs[:, :-1] keeps column 0,
    # so the per-token reference values live in the FIRST column
    lp = torch.tensor([[-1.0], [-2.0], [-0.5], [-3.0]], dtype=torch.float64)
    ref = torch.tensor(
        [[-1.5, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]], dtype=torch.float64
    )
    data = {
        "input_ids": torch.zeros(4, 2, dtype=torch.long),
        "reference_policy_logprobs": ref,
        "token_mask": torch.tensor([[0.0, 1.0]] * 4, dtype=torch.float64),
        "sample_mask": torch.ones(4, dtype=torch.float64),
    }
    # rewards r = beta * (lp - ref): [1.0, -2.0, 1.0, -2.0]
    delta = 0.25
    cfg = make_cfg(
        reference_policy_kl_penalty=beta,
        preference_loss_weight=0.0,
        quality_loss_weight=1.0,
        quality_delta_init=delta,
    )
    loss, metrics = MPOLossFn(cfg)(
        lp,
        data,
        torch.tensor(4.0, dtype=torch.float64),
        torch.tensor(4.0, dtype=torch.float64),
    )

    sigma = lambda x: 1.0 / (1.0 + math.exp(-x))
    per_pair = -math.log(sigma(1.0 - delta)) - math.log(sigma(-(-2.0 - delta)))
    expected = 2 * per_pair / (2 + 1e-8)  # two identical pairs, gvs/2 = 2
    assert loss.item() == pytest.approx(expected, rel=1e-10)
    # masked_mean's +1e-8 normalization epsilon bounds the achievable precision
    assert metrics["implicit_rewards_chosen_mean"] == pytest.approx(1.0, rel=1e-8)
    assert metrics["implicit_rewards_rejected_mean"] == pytest.approx(-2.0, rel=1e-8)


# ---------------------------------------------------------------------------
# 3. STEP-1 invariant: policy == reference
# ---------------------------------------------------------------------------
def test_step1_invariant_delta_zero():
    w_pref, w_q, w_sft = 1.0, 1.0, 0.7
    cfg = make_cfg(
        preference_loss_weight=w_pref,
        sft_loss_weight=w_sft,
        quality_loss_weight=w_q,
    )
    _, data, gvs, gvt = make_fixture(seed=3)
    lp = data["reference_policy_logprobs"][:, :-1].detach().clone().requires_grad_(True)

    loss, metrics = MPOLossFn(cfg)(lp, data, gvs, gvt)

    # rel=1e-8 everywhere below: masked_mean normalizes by (gvs/2 + 1e-8)
    assert metrics["preference_loss"] == pytest.approx(LN2, rel=1e-8)
    assert metrics["delta"] == 0.0  # folding all-zero rewards keeps delta at 0
    assert metrics["quality_loss"] == pytest.approx(2 * LN2, rel=1e-8)
    assert metrics["implicit_rewards_chosen_mean"] == 0.0
    assert metrics["implicit_rewards_rejected_mean"] == 0.0

    # chosen NLL exactly as the stock dpo-mode NLL computes it
    tm = data["token_mask"][:, 1:]
    nll = -(lp.detach() * tm).sum(-1)
    sft_expected = nll[::2].sum() / (gvs / 2 + 1e-8)
    expected_total = w_pref * LN2 + w_q * 2 * LN2 + w_sft * sft_expected
    assert loss.item() == pytest.approx(expected_total.item(), rel=1e-8)


def test_step1_invariant_delta_nonzero_closed_form():
    d, c = 0.37, 10.0
    cfg = make_cfg(
        preference_loss_weight=1.0,
        quality_loss_weight=1.0,
        quality_delta_init=d,
        quality_delta_resume_count=c,
    )
    _, data, gvs, gvt = make_fixture(num_pairs=3, seed=4)
    lp = data["reference_policy_logprobs"][:, :-1].detach().clone().requires_grad_(True)

    _, metrics = MPOLossFn(cfg)(lp, data, gvs, gvt)

    n = float(data["sample_mask"].sum())  # 6 valid samples fold in as zeros
    delta_post = d * c / (c + n)
    assert metrics["delta"] == pytest.approx(delta_post, rel=1e-12)
    assert metrics["delta_count"] == pytest.approx(c + n, rel=1e-12)
    sigma = lambda x: 1.0 / (1.0 + math.exp(-x))
    expected_quality = -math.log(sigma(-delta_post)) - math.log(sigma(delta_post))
    # rel=1e-8: masked_mean normalizes by (gvs/2 + 1e-8)
    assert metrics["quality_loss"] == pytest.approx(expected_quality, rel=1e-8)


# ---------------------------------------------------------------------------
# 4. delta running mean: count-weighted merge (TRL semantics, no EMA)
# ---------------------------------------------------------------------------
def test_delta_count_weighted_merge_matches_union():
    cfg = make_cfg(quality_loss_weight=1.0)
    loss_fn = MPOLossFn(cfg)

    lp_a, data_a, gvs_a, gvt_a = make_fixture(num_pairs=3, seed=21)
    lp_b, data_b, gvs_b, gvt_b = make_fixture(num_pairs=2, seed=22)
    # make batch B partially masked so the batches carry different counts
    data_b["sample_mask"][2] = 0.0
    data_b["sample_mask"][3] = 0.0
    gvs_b = data_b["sample_mask"].sum()

    _, m_a = loss_fn(lp_a, data_a, gvs_a, gvt_a)
    _, m_b = loss_fn(lp_b, data_b, gvs_b, gvt_b)

    beta = cfg["reference_policy_kl_penalty"]
    r_a = implicit_rewards_reference(lp_a.detach(), data_a, beta)
    r_b = implicit_rewards_reference(lp_b.detach(), data_b, beta)
    valid = torch.cat(
        [
            r_a[data_a["sample_mask"] > 0],
            r_b[data_b["sample_mask"] > 0],
        ]
    )
    union_mean = valid.mean().item()
    assert m_b["delta"] == pytest.approx(union_mean, rel=1e-6)
    assert m_b["delta_count"] == pytest.approx(len(valid), rel=1e-12)

    # explicitly NOT an EMA: with unequal counts the unweighted average of the
    # two batch means differs from the count-weighted running mean
    mean_a = r_a.mean().item()
    mean_b = r_b[data_b["sample_mask"] > 0].mean().item()
    assert m_b["delta"] != pytest.approx((mean_a + mean_b) / 2, rel=1e-6)


def test_running_mean_state_mirrors_trl_formula():
    state = mpo_mod.RunningMeanState()
    batches = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([10.0]),
        torch.tensor([-4.0, 6.0]),
    ]
    mean, count = 0.0, 1e-24
    for xs in batches:
        state.update(xs.sum(), torch.tensor(float(xs.numel())), group=None)
        # TRL RunningMoments.update (mean/count lines)
        xs_mean, xs_count = xs.mean().item(), xs.numel()
        tot = count + xs_count
        mean += (xs_mean - mean) * xs_count / tot
        count = tot
        assert state.mean == pytest.approx(mean, rel=1e-7)
        assert state.count == pytest.approx(count, rel=1e-12)
    assert state.mean == pytest.approx(torch.cat(batches).mean().item(), rel=1e-6)


# ---------------------------------------------------------------------------
# 5. masking: invalid samples excluded from quality loss AND delta
# ---------------------------------------------------------------------------
def test_masked_samples_excluded_from_quality_and_delta():
    delta = 0.1
    cfg = make_cfg(
        preference_loss_weight=0.0,
        quality_loss_weight=1.0,
        quality_delta_init=delta,
        quality_delta_resume_count=1e6,  # pin delta: huge inertia
    )
    lp, data, _, gvt = make_fixture(num_pairs=3, seed=31)
    data["sample_mask"][2] = 0.0  # invalidate pair 1 (rows 2, 3)
    data["sample_mask"][3] = 0.0
    gvs = data["sample_mask"].sum()

    loss, metrics = MPOLossFn(cfg)(lp, data, gvs, gvt)

    beta = cfg["reference_policy_kl_penalty"]
    rewards = implicit_rewards_reference(lp.detach(), data, beta)
    # quality over valid pairs only (delta pinned by the huge resume count)
    expected = quality_loss_reference(
        rewards, data["sample_mask"], gvs, metrics["delta"]
    )
    assert loss.item() == pytest.approx(expected.item(), rel=1e-9)

    # delta update counted only the 4 valid samples
    assert metrics["delta_count"] == pytest.approx(1e6 + 4, rel=1e-12)

    # and the folded sum excluded masked rewards: reconstruct the update
    valid_mean = rewards[data["sample_mask"] > 0].mean().item()
    expected_delta = delta + (valid_mean - delta) * 4 / (1e6 + 4)
    assert metrics["delta"] == pytest.approx(expected_delta, rel=1e-9)


# ---------------------------------------------------------------------------
# 6. gradient directions: quality pushes r_chosen up, r_rejected down
# ---------------------------------------------------------------------------
def test_quality_gradient_directions():
    cfg = make_cfg(preference_loss_weight=0.0, quality_loss_weight=1.0)
    lp, data, gvs, gvt = make_fixture(num_pairs=3, seed=41)

    loss, _ = MPOLossFn(cfg)(lp, data, gvs, gvt)
    loss.backward()

    token_mask = data["token_mask"][:, 1:]
    grad = lp.grad
    # d loss / d logp = -(1 - sigmoid(r_c - delta)) * beta * mask  < 0 (chosen)
    #                 = +(1 - sigmoid(-(r_r - delta))) * beta * mask > 0 (rejected)
    chosen_grads = grad[::2][token_mask[::2] > 0]
    rejected_grads = grad[1::2][token_mask[1::2] > 0]
    assert (chosen_grads < 0).all(), "quality term must push chosen rewards up"
    assert (rejected_grads > 0).all(), "quality term must push rejected rewards down"
    # masked-out token positions receive no gradient
    assert (grad[token_mask == 0] == 0).all()


# ---------------------------------------------------------------------------
# 7. distributed fallback + train/eval gating
# ---------------------------------------------------------------------------
def test_distributed_fallback_local_update():
    assert not torch.distributed.is_initialized()
    assert mpo_mod._resolve_delta_group() is None
    assert mpo_mod._is_delta_writer(None)

    cfg = make_cfg(quality_loss_weight=0.2)
    lp, data, gvs, gvt = make_fixture(seed=51)
    _, metrics = MPOLossFn(cfg)(lp, data, gvs, gvt)  # must not crash
    rewards = implicit_rewards_reference(
        lp.detach(), data, cfg["reference_policy_kl_penalty"]
    )
    assert metrics["delta"] == pytest.approx(rewards.mean().item(), rel=1e-6)


def test_eval_mode_does_not_update_delta():
    cfg = make_cfg(
        quality_loss_weight=1.0,
        quality_delta_init=0.42,
        quality_delta_resume_count=5.0,
    )
    lp, data, gvs, gvt = make_fixture(seed=52, requires_grad=False)

    _, metrics = MPOLossFn(cfg)(lp, data, gvs, gvt)

    assert metrics["delta"] == 0.42
    assert metrics["delta_count"] == pytest.approx(5.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 8. worker-resident state + sidecar persistence
# ---------------------------------------------------------------------------
def test_state_survives_repickled_loss_instances():
    """Per-step re-pickling yields fresh MPOLossFn objects; delta must persist."""
    cfg = make_cfg(quality_loss_weight=1.0)
    lp, data, gvs, gvt = make_fixture(seed=61)

    _, m1 = MPOLossFn(cfg)(lp, data, gvs, gvt)
    lp_eval, data2, gvs2, gvt2 = make_fixture(seed=62, requires_grad=False)
    _, m2 = MPOLossFn(cfg)(lp_eval, data2, gvs2, gvt2)  # fresh instance, eval

    assert m2["delta"] == m1["delta"]
    assert m2["delta_count"] == m1["delta_count"]


def test_sidecar_roundtrip_and_seed_priority(tmp_path):
    sidecar = str(tmp_path / "mpo_delta.json")
    cfg = make_cfg(quality_loss_weight=1.0, quality_delta_state_path=sidecar)
    lp, data, gvs, gvt = make_fixture(seed=71)

    _, m1 = MPOLossFn(cfg)(lp, data, gvs, gvt)

    payload = mpo_mod.read_delta_sidecar(sidecar)
    assert payload is not None
    assert payload["mean"] == pytest.approx(m1["delta"], rel=1e-12)
    assert payload["count"] == pytest.approx(m1["delta_count"], rel=1e-12)

    # worker restart: registry cleared; sidecar outranks the config seeds
    mpo_mod.reset_delta_states()
    cfg_restart = dict(cfg, quality_delta_init=99.0, quality_delta_resume_count=1.0)
    lp_eval, data2, gvs2, gvt2 = make_fixture(seed=72, requires_grad=False)
    _, m2 = MPOLossFn(cfg_restart)(lp_eval, data2, gvs2, gvt2)
    assert m2["delta"] == pytest.approx(m1["delta"], rel=1e-12)
    assert m2["delta_count"] == pytest.approx(m1["delta_count"], rel=1e-12)


def test_sidecar_missing_or_corrupt_falls_back_to_config(tmp_path):
    missing = str(tmp_path / "absent.json")
    assert mpo_mod.read_delta_sidecar(missing) is None

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert mpo_mod.read_delta_sidecar(str(corrupt)) is None

    cfg = make_cfg(
        quality_loss_weight=1.0,
        quality_delta_state_path=str(corrupt),
        quality_delta_init=0.7,
        quality_delta_resume_count=3.0,
    )
    lp, data, gvs, gvt = make_fixture(seed=81, requires_grad=False)
    _, metrics = MPOLossFn(cfg)(lp, data, gvs, gvt)
    assert metrics["delta"] == 0.7
    assert metrics["delta_count"] == pytest.approx(3.0, rel=1e-12)


def test_delta_sidecar_path_helper():
    assert mpo_mod.delta_sidecar_path("/ckpts/run1") == "/ckpts/run1/mpo_delta.json"
