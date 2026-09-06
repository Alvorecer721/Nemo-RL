# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""ALP on complete SingleController prompt-occurrence groups."""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch

from nemo_rl.algorithms.reward_functions import RewardShapingConfig


def validate_alp_config(cfg: RewardShapingConfig) -> None:
    """Validate the normalized, zero-floor ALP configuration before allocation."""
    if cfg.alp_coef is None or not math.isfinite(cfg.alp_coef) or cfg.alp_coef < 0:
        raise ValueError("SingleController ALP requires a finite alp_coef >= 0")
    if cfg.max_response_length is None or cfg.max_response_length <= 0:
        raise ValueError("SingleController ALP requires max_response_length > 0")
    if any(
        value is not None
        for value in (
            cfg.overlong_buffer_length,
            cfg.overlong_buffer_penalty,
            cfg.stop_properly_penalty_coef,
        )
    ):
        raise ValueError(
            "SingleController ALP cannot combine overlong or stop-properly shaping; "
            "set overlong_buffer_length, overlong_buffer_penalty, and "
            "stop_properly_penalty_coef to null"
        )


@dataclass
class ALPResult:
    """Shaped rewards and per-row statistics for the same admitted samples."""

    rewards: torch.Tensor
    group_ids: torch.Tensor
    successes: torch.Tensor
    response_lengths: torch.Tensor


def apply_grouped_alp(
    rewards: torch.Tensor,
    *,
    successes: torch.Tensor,
    token_mask: torch.Tensor,
    sample_ids: Sequence[str],
    group_size: int,
    cfg: RewardShapingConfig,
) -> ALPResult:
    """Shape complete rollout groups before filtering/sharding or advantages.

    Sample IDs follow the producer's ``{group_id}_g{generation_index}`` contract.
    Group identity is the rollout occurrence, never prompt text or dataset index.
    Every generated policy token contributes length, including later turns; the
    producer's token mask excludes prompt history, observations, and padding.
    All G outcomes contribute to solve rate, before policy-side sample filtering.
    The raw reward may be non-binary; successes must be explicit binary outcomes.
    """
    validate_alp_config(cfg)
    n = len(sample_ids)
    if group_size < 2 or n == 0:
        raise ValueError("ALP requires nonempty complete groups of at least 2 rollouts")
    if rewards.shape != (n,) or successes.shape != (n,):
        raise ValueError("ALP rewards and episode successes must have shape [samples]")
    if not torch.isfinite(rewards).all():
        raise ValueError("ALP task rewards must be finite")
    if (
        not torch.isfinite(successes).all()
        or not ((successes == 0) | (successes == 1)).all()
    ):
        raise ValueError(
            "ALP requires a finite binary episode_success for every completion. "
            "Provide EnvironmentReturn.episode_successes or the Gym grader's "
            "episode_success field; binary_reward is only for binary episode rewards."
        )
    if (
        token_mask.ndim != 2
        or token_mask.shape[0] != n
        or not ((token_mask == 0) | (token_mask == 1)).all()
    ):
        raise ValueError("ALP requires a binary generated-token mask [samples, tokens]")

    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, sample_id in enumerate(sample_ids):
        group, separator, generation = sample_id.rpartition("_g")
        if not group or not separator or not generation.isdecimal():
            raise ValueError(f"Invalid ALP rollout sample ID: {sample_id!r}")
        groups[group].append((row, int(generation)))

    successes = successes.to(device=rewards.device, dtype=rewards.dtype)
    pass_rates = torch.empty_like(rewards)
    group_ids = torch.empty((n, 1), device=rewards.device, dtype=torch.long)
    for group_number, (group, rows) in enumerate(groups.items()):
        if sorted(generation for _, generation in rows) != list(range(group_size)):
            raise ValueError(
                f"ALP requires all {group_size} unique rollouts for {group!r}"
            )
        indices = [row for row, _ in rows]
        pass_rates[indices] = successes[indices].mean()
        group_ids[indices, 0] = group_number

    lengths = token_mask.sum(dim=-1).to(device=rewards.device, dtype=rewards.dtype)
    assert cfg.alp_coef is not None and cfg.max_response_length is not None
    shaped = rewards - cfg.alp_coef * pass_rates * lengths / cfg.max_response_length
    return ALPResult(shaped, group_ids, successes, lengths)
