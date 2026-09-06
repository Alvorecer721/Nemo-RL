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
"""ALP grouping, reward semantics, and setup guards for SingleController."""

import pytest
import torch

from nemo_rl.algorithms.reward_functions import RewardShapingConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    validate_single_controller_config,
)
from nemo_rl.algorithms.single_controller_utils.rewards import apply_grouped_alp
from tests.unit.single_controller.test_resiliency_config import _master_config


def _cfg(**kwargs):
    return RewardShapingConfig(
        enabled=True, alp_coef=0.5, max_response_length=10, **kwargs
    )


def test_alp_composite_rewards_and_interleaved_occurrence_groups():
    raw = torch.tensor([1.1, -0.2, 0.1, 0.1])
    out = apply_grouped_alp(
        raw,
        successes=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        token_mask=torch.tensor(
            [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1]]
        ),
        sample_ids=["a_g0", "b_g0", "a_g1", "b_g1"],
        group_size=2,
        cfg=_cfg(),
    )
    torch.testing.assert_close(out.rewards, torch.tensor([1.05, -0.2, 0.0, 0.1]))
    assert out.group_ids.tolist() == [[0], [1], [0], [1]]
    assert out.response_lengths.tolist() == [2, 2, 4, 4]
    # The raw task metric remains intact, including negative non-binary values.
    torch.testing.assert_close(raw, torch.tensor([1.1, -0.2, 0.1, 0.1]))


@pytest.mark.parametrize("successes", [[0.0, 0.0], [1.0, 1.0]])
def test_alp_zero_floor_and_all_correct_groups(successes):
    out = apply_grouped_alp(
        torch.tensor([0.1, 0.1]),
        successes=torch.tensor(successes),
        token_mask=torch.tensor([[0, 1, 0], [0, 1, 1]]),
        sample_ids=["q_g0", "q_g1"],
        group_size=2,
        cfg=_cfg(),
    )
    expected = [0.1, 0.1] if successes[0] == 0 else [0.05, 0.0]
    torch.testing.assert_close(out.rewards, torch.tensor(expected))


@pytest.mark.parametrize(
    "successes", [[0.2, 0.8], [-1.0, 1.0], [2.0, 1.0], [float("nan"), 0.0]]
)
def test_alp_rejects_scores_and_missing_episode_outcomes(successes):
    with pytest.raises(ValueError, match="binary episode_success"):
        apply_grouped_alp(
            torch.ones(2),
            successes=torch.tensor(successes),
            token_mask=torch.ones(2, 3),
            sample_ids=["q_g0", "q_g1"],
            group_size=2,
            cfg=_cfg(),
        )


@pytest.mark.parametrize(
    "sample_ids", [["a_g0", "b_g1"], ["a_g0", "a_g0"], ["a_g0", "a_g2"], ["a", "b"]]
)
def test_alp_rejects_partial_duplicate_and_invalid_groups(sample_ids):
    with pytest.raises(ValueError, match="[Rr]ollout"):
        apply_grouped_alp(
            torch.ones(2),
            successes=torch.ones(2),
            token_mask=torch.ones(2, 3),
            sample_ids=sample_ids,
            group_size=2,
            cfg=_cfg(),
        )


@pytest.mark.parametrize("group_size, count", [(1, 1), (2, 0)])
def test_alp_rejects_singleton_and_empty_batches(group_size: int, count: int) -> None:
    with pytest.raises(ValueError, match="nonempty complete groups of at least 2"):
        apply_grouped_alp(
            torch.ones(count),
            successes=torch.ones(count),
            token_mask=torch.ones(count, 3),
            sample_ids=[f"q_g{i}" for i in range(count)],
            group_size=group_size,
            cfg=_cfg(),
        )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("rewards", torch.ones(2, 1), "shape"),
        ("successes", torch.ones(1), "shape"),
        ("rewards", torch.tensor([float("inf"), 0.0]), "finite"),
        ("rewards", torch.tensor([float("nan"), 0.0]), "finite"),
        ("token_mask", torch.ones(2), "binary generated-token mask"),
        ("token_mask", torch.ones(1, 3), "binary generated-token mask"),
        (
            "token_mask",
            torch.tensor([[0.0, 0.5], [0.0, 1.0]]),
            "binary generated-token mask",
        ),
    ],
)
def test_alp_rejects_invalid_reward_and_token_inputs(
    field: str, value: torch.Tensor, message: str
) -> None:
    inputs = {
        "rewards": torch.ones(2),
        "successes": torch.ones(2),
        "token_mask": torch.ones(2, 3),
    }
    inputs[field] = value
    with pytest.raises(ValueError, match=message):
        apply_grouped_alp(
            **inputs, sample_ids=["q_g0", "q_g1"], group_size=2, cfg=_cfg()
        )


def test_single_controller_accepts_alp():
    cfg = _master_config()
    cfg.grpo.reward_shaping = _cfg()
    validate_single_controller_config(cfg)


@pytest.mark.parametrize("name", ["gdpo", "opd", "reinforce_plus_plus"])
def test_single_controller_rejects_alp_with_other_estimators(name):
    cfg = _master_config()
    cfg.grpo.reward_shaping = _cfg()
    cfg.grpo.adv_estimator.name = name
    with pytest.raises(ValueError, match="requires the GRPO advantage estimator"):
        validate_single_controller_config(cfg)


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf")])
def test_single_controller_rejects_invalid_alp(coefficient):
    cfg = _master_config()
    cfg.grpo.reward_shaping = _cfg()
    cfg.grpo.reward_shaping.alp_coef = coefficient
    with pytest.raises(ValueError, match="finite alp_coef"):
        validate_single_controller_config(cfg)


def test_single_controller_rejects_shadowed_penalties():
    cfg = _master_config()
    cfg.grpo.reward_shaping = _cfg(overlong_buffer_penalty=1.0)
    with pytest.raises(ValueError, match="cannot combine"):
        validate_single_controller_config(cfg)
