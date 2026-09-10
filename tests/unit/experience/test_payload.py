# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.reward_functions import RewardShapingConfig
from nemo_rl.algorithms.single_controller_utils.rewards import apply_grouped_alp
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.codec import materialize
from nemo_rl.data_plane.schema import (
    EPISODE_SUCCESS,
    INVALID_TOOL_CALL_MASK,
    MALFORMED_THINKING_MASK,
    SC_ROLLOUT_SCHEMA_FIELDS,
)
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.payload import pack_payload, record_to_train_batch


def _routes(start: int, count: int) -> torch.Tensor:
    token_routes = torch.arange(start, start + count, dtype=torch.int16).view(
        count, 1, 1
    )
    topk_offsets = torch.arange(2, dtype=torch.int16).view(1, 1, 2)
    return (token_routes + topk_offsets).expand(count, 2, 2).contiguous()


def _fallback_routes(count: int) -> torch.Tensor:
    return torch.arange(2, dtype=torch.int16).view(1, 1, 2).expand(count, 2, 2)


def _completion(
    route_start: int,
    reward: float,
    *,
    env_token_ids: tuple[int, ...] = (30,),
    with_routes: bool = True,
    mask_sample: bool | None = None,
    truncated: bool = False,
) -> Completion:
    message_log = [
        {
            "role": "user",
            "content": "prompt",
            "token_ids": torch.tensor([10, 11]),
            "routed_experts": _routes(route_start, 2),
        },
        {
            "role": "assistant",
            "content": "answer",
            "token_ids": torch.tensor([20, 21]),
            "generation_logprobs": torch.tensor([-0.1, -0.2]),
            "routed_experts": _routes(route_start + 2, 2),
        },
        {
            "role": "user",
            "content": "environment",
            "token_ids": torch.tensor(env_token_ids),
            "routed_experts": _fallback_routes(len(env_token_ids)),
        },
    ]
    if not with_routes:
        for message in message_log:
            message.pop("routed_experts")
    env_extras = (
        None
        if mask_sample is None
        else {"instance_config": {"mask_sample": mask_sample}}
    )
    return Completion(
        message_log=message_log,
        env_extras=env_extras,
        truncated=truncated,
        reward=reward,
    )


def _record(
    completions: list[Completion], *, loss_multiplier: float = 1.0
) -> PromptGroupRecord:
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11]),
            }
        ],
        extra_env_info=None,
        metadata={"task_name": "test"},
        completions=completions,
        rollout_metrics={},
        loss_multiplier=loss_multiplier,
    )


def test_record_to_train_batch_preserves_routed_experts_in_tq_payload() -> None:
    record = _record(
        [
            _completion(route_start=10, reward=1.0),
            _completion(
                route_start=30,
                reward=2.0,
                env_token_ids=(30, 31),
            ),
        ]
    )

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=False,
    )

    expected_routes = [
        torch.cat((_routes(10, 4), _fallback_routes(1))),
        torch.cat((_routes(30, 4), _fallback_routes(2))),
    ]
    assert train_batch["input_lengths"].tolist() == [5, 6]
    assert train_batch["routed_experts"].shape == (2, 6, 2, 2)
    assert torch.equal(
        train_batch["routed_experts"][0, :5],
        expected_routes[0],
    )
    assert torch.equal(
        train_batch["routed_experts"][1],
        expected_routes[1],
    )

    sample_ids, fields, tags = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
        prompt_idx=17,
    )
    assert sample_ids == ["group_g0", "group_g1"]
    assert "routed_experts" in fields
    packed_routes = fields["routed_experts"]
    assert packed_routes.is_nested
    packed_rows = list(packed_routes.unbind())
    assert torch.equal(packed_rows[0], expected_routes[0])
    assert torch.equal(packed_rows[1], expected_routes[1])
    no_violations = {
        "num_invalid_tool_calls": 0,
        "num_malformed_thinking": 0,
        "num_assistant_messages": 1,
        "num_routed_experts_backfilled": 0,
    }
    assert tags == [
        {"weight_version": 3, "prompt_idx": 17, **no_violations},
        {"weight_version": 3, "prompt_idx": 17, **no_violations},
    ]


def test_record_to_train_batch_preserves_message_violation_masks() -> None:
    invalid = _completion(route_start=10, reward=1.0)
    invalid.message_log[1]["is_invalid_tool_call"] = True

    malformed = _completion(
        route_start=30,
        reward=2.0,
        env_token_ids=(30, 31),
    )
    malformed.message_log[1]["has_malformed_thinking"] = True

    train_batch = record_to_train_batch(
        _record([invalid, malformed]),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=True,
    )

    assert train_batch[INVALID_TOOL_CALL_MASK].dtype == torch.bool
    assert train_batch[MALFORMED_THINKING_MASK].dtype == torch.bool
    assert train_batch[INVALID_TOOL_CALL_MASK][0, :5].tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert not train_batch[MALFORMED_THINKING_MASK][0, :5].any()
    assert not train_batch[INVALID_TOOL_CALL_MASK][1, :6].any()
    assert train_batch[MALFORMED_THINKING_MASK][1, :6].tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]

    _, fields, tags = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
        prompt_idx=17,
    )
    invalid_rows = list(fields[INVALID_TOOL_CALL_MASK].unbind())
    malformed_rows = list(fields[MALFORMED_THINKING_MASK].unbind())
    assert invalid_rows[0].tolist() == [False, False, True, True, False]
    assert malformed_rows[1].tolist() == [False, False, True, True, False, False]
    assert tags[0]["num_invalid_tool_calls"] == 1
    assert tags[1]["num_malformed_thinking"] == 1


def test_record_to_train_batch_preserves_clean_masks_when_enabled() -> None:
    train_batch = record_to_train_batch(
        _record([_completion(route_start=10, reward=1.0)]),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=True,
    )

    assert not train_batch[INVALID_TOOL_CALL_MASK].any()
    assert not train_batch[MALFORMED_THINKING_MASK].any()


def test_record_to_train_batch_omits_routed_experts_when_absent() -> None:
    completion = _completion(route_start=10, reward=1.0, with_routes=False)
    completion.message_log[1]["is_invalid_tool_call"] = True
    completion.message_log[1]["has_malformed_thinking"] = True
    record = _record([completion])

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=False,
    )
    assert "routed_experts" not in train_batch
    assert INVALID_TOOL_CALL_MASK not in train_batch
    assert MALFORMED_THINKING_MASK not in train_batch

    _, fields, _ = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
        prompt_idx=17,
    )
    assert "routed_experts" not in fields


def test_multimodal_packed_tensor_round_trips_through_tq_payload() -> None:
    completions = [
        _completion(route_start=10, reward=1.0, with_routes=False),
        _completion(route_start=30, reward=2.0, with_routes=False),
    ]
    media = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    completions[0].message_log[0]["pixel_values"] = PackedTensor(media, dim_to_pack=0)

    train_batch = record_to_train_batch(
        _record(completions),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=False,
    )
    assert isinstance(train_batch["pixel_values"], PackedTensor)

    _, fields, tags = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
        prompt_idx=17,
    )
    assert "pixel_values" in fields
    assert tags[0]["pixel_values__row_shapes"]["shapes"] == [[2, 4]]
    assert tags[1]["pixel_values__row_shapes"]["shapes"] == []

    restored = materialize(fields, tags=tags)
    restored_media = restored["pixel_values"]
    assert isinstance(restored_media, PackedTensor)
    assert len(restored_media) == 2
    assert restored_media.logical_segment_counts_by_row() == [1, 0]
    assert torch.equal(restored_media.as_tensor(), media)


def test_per_token_multimodal_field_is_packed_with_sequence_lengths() -> None:
    train_batch = {
        "input_lengths": torch.tensor([3, 2], dtype=torch.int32),
        "input_ids": torch.tensor([[10, 11, 12], [20, 21, 0]]),
        "token_type_ids": torch.tensor([[0, 1, 1], [0, 1, 0]]),
    }

    _, fields, tags = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
        prompt_idx=17,
    )

    assert [row.tolist() for row in fields["token_type_ids"].unbind()] == [
        [0, 1, 1],
        [0, 1],
    ]
    assert tags == [
        {"weight_version": 3, "prompt_idx": 17},
        {"weight_version": 3, "prompt_idx": 17},
    ]


def test_record_to_train_batch_carries_raw_masks_without_applying_them() -> None:
    record = _record(
        [
            _completion(
                route_start=10,
                reward=1.0,
                mask_sample=True,
            ),
            _completion(
                route_start=30,
                reward=2.0,
                mask_sample=False,
                truncated=True,
            ),
            _completion(route_start=50, reward=3.0),
        ]
    )

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=False,
    )

    assert torch.equal(train_batch["sample_mask"], torch.ones(3))
    assert torch.equal(
        train_batch["mask_sample"],
        torch.tensor([True, False, False]),
    )
    assert torch.equal(
        train_batch["truncated"],
        torch.tensor([False, True, False]),
    )

    _, fields, _ = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
        prompt_idx=17,
    )
    assert torch.equal(fields["mask_sample"], train_batch["mask_sample"])
    assert torch.equal(fields["truncated"], train_batch["truncated"])


def test_record_to_train_batch_broadcasts_prompt_loss_multiplier() -> None:
    record = _record(
        [
            _completion(route_start=10, reward=1.0),
            _completion(route_start=30, reward=2.0),
        ],
        loss_multiplier=0.25,
    )

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=False,
    )

    expected = torch.full((2,), 0.25)
    assert torch.equal(train_batch["sample_mask"], expected)

    _, fields, _ = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
        prompt_idx=17,
    )
    assert torch.equal(fields["sample_mask"], expected)


def _failed_completion() -> Completion:
    """A trajectory whose first generation raised: prompt only, no routes."""
    return Completion(
        message_log=[
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11]),
            }
        ],
        env_extras=None,
        truncated=False,
        reward=0.0,
    )


def test_record_to_train_batch_backfills_routes_for_failed_completion() -> None:
    """A group is packable when only some completions generated (and so have routes)."""
    record = _record([_completion(route_start=10, reward=1.0), _failed_completion()])

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=False,
    )

    assert train_batch["input_lengths"].tolist() == [5, 2]
    routes = train_batch["routed_experts"]
    assert routes.shape == (2, 5, 2, 2)
    assert torch.equal(routes[0, :5], torch.cat((_routes(10, 4), _fallback_routes(1))))
    # The completion that never generated gets the all--1 missing-route sentinel,
    # so Megatron routes those tokens with its own router.
    assert torch.equal(routes[1, :2], torch.full((2, 2, 2), -1, dtype=routes.dtype))
    # It is fully loss-masked either way.
    assert train_batch["token_mask"][1, :2].tolist() == [0, 0]

    _, fields, _ = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
        prompt_idx=17,
    )
    assert "routed_experts" in fields
    assert list(fields["routed_experts"].unbind())[1].shape == (2, 2, 2)


def test_pack_payload_stamps_violation_counts_on_tags() -> None:
    """Each flag lands in its own counter; a row that never generated counts zero."""
    completions = [
        _completion(route_start=10, reward=1.0),
        _completion(route_start=30, reward=1.0),
        _failed_completion(),
    ]
    completions[0].message_log[1]["is_invalid_tool_call"] = True
    completions[1].message_log[1]["has_malformed_thinking"] = True

    train_batch = record_to_train_batch(
        _record(completions),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=False,
    )
    _, fields, tags = pack_payload(
        train_batch,
        weight_version=7,
        group_id="g",
        prompt_idx=17,
    )

    assert "violation_counts" not in fields
    assert tags == [
        {
            "weight_version": 7,
            "prompt_idx": 17,
            "num_invalid_tool_calls": 1,
            "num_malformed_thinking": 0,
            "num_assistant_messages": 1,
            "num_routed_experts_backfilled": 0,
        },
        {
            "weight_version": 7,
            "prompt_idx": 17,
            "num_invalid_tool_calls": 0,
            "num_malformed_thinking": 1,
            "num_assistant_messages": 1,
            "num_routed_experts_backfilled": 0,
        },
        {
            "weight_version": 7,
            "prompt_idx": 17,
            "num_invalid_tool_calls": 0,
            "num_malformed_thinking": 0,
            "num_assistant_messages": 0,
            # _failed_completion() has one message missing routed_experts; the
            # other two completions in this group carry real routes, so
            # backfill_missing_routed_experts finds a template and fills it.
            "num_routed_experts_backfilled": 1,
        },
    ]


@pytest.mark.parametrize("history_has_logprobs", [False, True])
def test_payload_keeps_episode_success_separate_and_masks_all_generated_turns(
    history_has_logprobs,
):
    completion = _completion(route_start=10, reward=2.7, with_routes=False)
    completion.episode_success = 0.0
    # The original prompt contains an assistant message; it must not be charged.
    history = {
        "role": "assistant",
        "content": "old",
        "token_ids": torch.tensor([5, 6, 7]),
    }
    if history_has_logprobs:
        history["generation_logprobs"] = torch.zeros(3)
    completion.message_log.insert(0, history)
    completion.message_log.append(
        {
            "role": "assistant",
            "content": "second response",
            "token_ids": torch.tensor([40, 41, 42]),
            "generation_logprobs": torch.tensor([-0.1, -0.2, -0.3]),
        }
    )
    record = _record([completion])
    record.prompt.insert(0, history)
    batch = record_to_train_batch(
        record, pad_value_dict={"token_ids": 0}, include_message_violation_fields=False
    )
    assert batch["total_reward"].item() == torch.tensor(2.7).item()
    assert batch[EPISODE_SUCCESS].tolist() == [0.0]
    assert batch["token_mask"].sum().item() == 5  # two generated turns, 2 + 3
    assert batch["token_mask"][0, :5].sum().item() == 0  # complete prompt history
    _, fields, _ = pack_payload(batch, weight_version=1, group_id="q", prompt_idx=0)
    assert fields[EPISODE_SUCCESS].item() == 0.0
    again = record_to_train_batch(
        record, pad_value_dict={"token_ids": 0}, include_message_violation_fields=False
    )
    torch.testing.assert_close(again["token_mask"], batch["token_mask"])
    assert ("generation_logprobs" in history) == history_has_logprobs


def test_payload_missing_success_is_not_inferred_from_reward():
    batch = record_to_train_batch(
        _record([_completion(route_start=0, reward=1.0)]),
        pad_value_dict={"token_ids": 0},
        include_message_violation_fields=False,
    )
    assert torch.isnan(batch[EPISODE_SUCCESS]).all()


@pytest.mark.parametrize("success_field_present", [False, True])
def test_alp_payload_survives_checkpoint_without_prompt_tensor(
    tmp_path, success_field_present: bool
) -> None:
    completions = [
        _completion(route_start=0, reward=1.1, with_routes=False),
        _completion(route_start=0, reward=0.1, with_routes=False),
    ]
    completions[0].episode_success = 1.0
    completions[1].episode_success = 0.0
    batch = record_to_train_batch(
        _record(completions),
        pad_value_dict={"token_ids": 0},
        include_message_violation_fields=False,
    )
    sample_ids, fields, tags = pack_payload(
        batch, weight_version=3, group_id="restored", prompt_idx=0
    )
    assert set(fields.keys()) <= set(SC_ROLLOUT_SCHEMA_FIELDS)
    if not success_field_present:
        # Older success-free snapshots must fail explicitly, even with raw
        # rewards present; restoration cannot fabricate a correctness signal.
        del fields[EPISODE_SUCCESS]
    original = NoOpDataPlaneClient()
    original.register_partition(
        "rollout_data", list(SC_ROLLOUT_SCHEMA_FIELDS), 2, ["train"], grpo_group_size=2
    )
    original.put_samples(sample_ids, "rollout_data", fields=fields, tags=tags)
    checkpoint = tmp_path / "payload"
    original.save_checkpoint(checkpoint)
    restored = NoOpDataPlaneClient()
    restored.load_checkpoint(checkpoint)
    required_fields = ["total_reward", "token_mask", EPISODE_SUCCESS]
    if not success_field_present:
        with pytest.raises(KeyError, match="episode_success.*not yet produced"):
            restored.get_samples(sample_ids, "rollout_data", required_fields)
        return

    inputs = restored.get_samples(sample_ids, "rollout_data", required_fields)
    shaped = apply_grouped_alp(
        inputs["total_reward"],
        successes=inputs[EPISODE_SUCCESS],
        token_mask=inputs["token_mask"],
        sample_ids=sample_ids,
        group_size=2,
        cfg=RewardShapingConfig(enabled=True, alp_coef=0.5, max_response_length=10),
    )
    torch.testing.assert_close(shaped.rewards, torch.tensor([1.05, 0.05]))
    assert shaped.group_ids.tolist() == [[0], [0]]
    assert shaped.response_lengths.tolist() == [2.0, 2.0]
    torch.testing.assert_close(inputs["total_reward"], torch.tensor([1.1, 0.1]))
