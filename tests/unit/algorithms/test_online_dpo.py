from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.online_dpo import (
    add_reference_logprobs_to_preference_batch,
    build_preference_datums_from_rollouts,
    collate_preference_datums,
    compute_gen_logprob_error_metrics,
    gather_generation_logprobs_for_pairs,
    strip_trailing_environment_messages,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _message_log(response_tokens: list[int], response_content: str):
    return [
        {
            "role": "user",
            "content": "question",
            "token_ids": torch.tensor([1, 2]),
        },
        {
            "role": "assistant",
            "content": response_content,
            "token_ids": torch.tensor(response_tokens),
        },
        {
            "role": "user",
            "content": "environment observation",
            "token_ids": torch.tensor([99]),
        },
    ]


def _rollout_batch(
    rewards: torch.Tensor,
    *,
    truncated: torch.Tensor | None = None,
) -> BatchedDataDict:
    batch = BatchedDataDict(
        {
            "message_log": [
                _message_log([3, 4], "first"),
                _message_log([5, 6, 7], "second"),
            ],
            "total_reward": rewards,
            "loss_multiplier": torch.tensor([1.0, 1.0]),
            "idx": [123, 123],
            "task_name": ["math", "math"],
        }
    )
    batch["truncated"] = (
        truncated if truncated is not None else torch.tensor([False, False])
    )
    return batch


def _two_pair_rollout_batch() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "message_log": [
                _message_log([3], "a"),
                _message_log([4], "b"),
                _message_log([5], "c"),
                _message_log([6], "d"),
            ],
            "total_reward": torch.tensor([0.0, 1.0, 0.2, 0.9]),
            "loss_multiplier": torch.ones(4),
            "idx": [0, 0, 1, 1],
            "task_name": ["math", "math", "math", "math"],
            "truncated": torch.tensor([False, False, False, False]),
        }
    )


def test_strip_trailing_environment_messages_keeps_final_assistant():
    stripped = strip_trailing_environment_messages(_message_log([3, 4], "answer"))

    assert [message["role"] for message in stripped] == ["user", "assistant"]
    assert stripped[-1]["content"] == "answer"


def test_build_preference_datums_orders_by_reward_and_strips_env_observation():
    datums, metrics = build_preference_datums_from_rollouts(
        _rollout_batch(torch.tensor([0.25, 0.75])),
        min_reward_margin=0.0,
        drop_truncated_pairs=True,
    )

    assert len(datums) == 1
    assert datums[0]["message_log_chosen"][-1]["content"] == "second"
    assert datums[0]["message_log_rejected"][-1]["content"] == "first"
    assert datums[0]["length_chosen"] == 5
    assert datums[0]["length_rejected"] == 4
    assert metrics["usable_pairs"] == 1.0
    assert metrics["reward_margin"] == 0.5


def test_build_preference_datums_respects_max_pairs_and_counts_discards():
    datums, metrics = build_preference_datums_from_rollouts(
        _two_pair_rollout_batch(),
        min_reward_margin=0.0,
        drop_truncated_pairs=True,
        max_pairs=1,
    )

    assert len(datums) == 1
    assert datums[0]["idx"] == 0
    assert metrics["generated_pairs"] == 2.0
    assert metrics["usable_pairs"] == 1.0
    assert metrics["discarded_pairs"] == 1.0


def test_build_preference_datums_drops_ties_within_margin():
    datums, metrics = build_preference_datums_from_rollouts(
        _rollout_batch(torch.tensor([0.25, 0.30])),
        min_reward_margin=0.1,
        drop_truncated_pairs=True,
    )

    assert datums == []
    assert metrics["tie_pairs"] == 1.0
    assert metrics["dropped_pairs"] == 1.0


def test_build_preference_datums_drops_truncated_pairs_when_enabled():
    datums, metrics = build_preference_datums_from_rollouts(
        _rollout_batch(
            torch.tensor([0.25, 0.75]),
            truncated=torch.tensor([False, True]),
        ),
        min_reward_margin=0.0,
        drop_truncated_pairs=True,
    )

    assert datums == []
    assert metrics["truncated_pairs"] == 1.0
    assert metrics["dropped_pairs"] == 1.0


def test_build_preference_datums_requires_truncated_key_when_dropping():
    batch = _rollout_batch(torch.tensor([0.25, 0.75]))
    del batch["truncated"]

    with pytest.raises(ValueError, match="truncated"):
        build_preference_datums_from_rollouts(
            batch,
            min_reward_margin=0.0,
            drop_truncated_pairs=True,
        )


def test_build_preference_datums_asserts_pair_idx_adjacency():
    batch = _rollout_batch(torch.tensor([0.25, 0.75]))
    batch["idx"] = [123, 456]

    with pytest.raises(AssertionError, match="adjacency"):
        build_preference_datums_from_rollouts(
            batch,
            min_reward_margin=0.0,
            drop_truncated_pairs=False,
        )


def test_collate_preference_datums_uses_dpo_interleaving_and_final_mask():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    datums, _ = build_preference_datums_from_rollouts(
        _rollout_batch(torch.tensor([0.25, 0.75])),
        min_reward_margin=0.0,
        drop_truncated_pairs=True,
    )

    batch = collate_preference_datums(
        datums,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=1,
    )

    assert batch["input_ids"].shape[0] == 2
    assert batch["sample_mask"].tolist() == [1.0, 1.0]
    assert batch["token_mask"][0].sum().item() == 3
    assert batch["token_mask"][1].sum().item() == 2


def test_add_reference_logprobs_rolls_reference_output():
    preference_batch = BatchedDataDict({"input_ids": torch.ones(2, 3)})
    policy = MagicMock()
    policy.get_reference_policy_logprobs.return_value = {
        "reference_logprobs": torch.tensor([[1, 2, 3], [4, 5, 6]])
    }

    add_reference_logprobs_to_preference_batch(
        preference_batch,
        policy,
        micro_batch_size=2,
    )

    assert preference_batch["reference_policy_logprobs"].tolist() == [
        [2, 3, 1],
        [5, 6, 4],
    ]
    policy.get_reference_policy_logprobs.assert_called_once_with(
        preference_batch,
        micro_batch_size=2,
        timer=None,
    )


# --- gen-vs-train logprob tripwire (pure-tensor, no GPU required) ----------


def _stripped_pair_log(response_tokens: list[int], gen_logprobs: list[float] | None):
    """A chosen/rejected message log as stored in a preference datum (env stripped)."""
    messages = [
        {"role": "user", "content": "q", "token_ids": torch.tensor([1, 2])},
        {
            "role": "assistant",
            "content": "a",
            "token_ids": torch.tensor(response_tokens),
        },
    ]
    if gen_logprobs is not None:
        messages[1]["generation_logprobs"] = torch.tensor(gen_logprobs)
    return messages


def test_compute_gen_logprob_error_metrics_known_values():
    # Position 0 is dropped by convention (grpo parity); token_mask covers
    # response tokens only. Masked-in abs errors: 0.2, 0.0, 0.5 (row 0) and
    # 1.0, 0.0 (row 1) -> mean 1.7/5, max 1.0. The 0.5 error planted at
    # position 0 of row 0 must not contribute.
    generation_logprobs = torch.tensor(
        [[0.0, -1.0, -2.0, -3.0], [0.0, -1.0, -1.0, -4.0]]
    )
    policy_logprobs = torch.tensor([[0.5, -1.2, -2.0, -3.5], [0.0, -1.0, -2.0, -4.0]])
    token_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

    metrics = compute_gen_logprob_error_metrics(
        generation_logprobs,
        policy_logprobs,
        token_mask=token_mask,
        sample_mask=torch.tensor([1.0, 1.0]),
    )

    assert metrics["gen_logprob_error_mean"] == pytest.approx(1.7 / 5)
    assert metrics["gen_logprob_error_max"] == pytest.approx(1.0)


def test_compute_gen_logprob_error_metrics_excludes_invalid_samples():
    generation_logprobs = torch.tensor(
        [[0.0, -1.0, -2.0, -3.0], [0.0, -1.0, -1.0, -4.0]]
    )
    policy_logprobs = torch.tensor([[0.5, -1.2, -2.0, -3.5], [0.0, -1.0, -2.0, -4.0]])
    token_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

    metrics = compute_gen_logprob_error_metrics(
        generation_logprobs,
        policy_logprobs,
        token_mask=token_mask,
        sample_mask=torch.tensor([1.0, 0.0]),  # row 1 invalid
    )

    assert metrics["gen_logprob_error_mean"] == pytest.approx(0.7 / 3)
    assert metrics["gen_logprob_error_max"] == pytest.approx(0.5)


def test_compute_gen_logprob_error_metrics_empty_mask_is_zero():
    generation_logprobs = torch.zeros(2, 4)
    policy_logprobs = torch.ones(2, 4)

    metrics = compute_gen_logprob_error_metrics(
        generation_logprobs,
        policy_logprobs,
        token_mask=torch.zeros(2, 4),
        sample_mask=torch.zeros(2),
    )

    assert metrics == {"gen_logprob_error_mean": 0.0, "gen_logprob_error_max": 0.0}


def test_gather_generation_logprobs_interleaves_zero_fills_and_pads():
    datums = [
        {
            "message_log_chosen": _stripped_pair_log([5, 6, 7], [-0.3, -0.4, -0.5]),
            "message_log_rejected": _stripped_pair_log([3, 4], [-0.1, -0.2]),
        }
    ]

    gathered = gather_generation_logprobs_for_pairs(datums, sequence_length=6)

    assert gathered.shape == (2, 6)
    # Row order matches preference_collate_fn: chosen first, then rejected.
    # Prompt tokens and right padding are zero-filled, grpo-style.
    assert torch.allclose(
        gathered[0], torch.tensor([0.0, 0.0, -0.3, -0.4, -0.5, 0.0])
    )
    assert torch.allclose(gathered[1], torch.tensor([0.0, 0.0, -0.1, -0.2, 0.0, 0.0]))


def test_gather_generation_logprobs_returns_none_when_absent():
    datums = [
        {
            "message_log_chosen": _stripped_pair_log([5, 6, 7], None),
            "message_log_rejected": _stripped_pair_log([3, 4], None),
        }
    ]

    assert gather_generation_logprobs_for_pairs(datums, sequence_length=6) is None


def test_gather_generation_logprobs_raises_on_mixed_presence():
    datums = [
        {
            "message_log_chosen": _stripped_pair_log([5, 6, 7], [-0.3, -0.4, -0.5]),
            "message_log_rejected": _stripped_pair_log([3, 4], None),
        }
    ]

    with pytest.raises(ValueError, match="inconsistent"):
        gather_generation_logprobs_for_pairs(datums, sequence_length=6)
