# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import TypeVar

import torch
from pydantic import BaseModel

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class RewardShapingConfig(BaseModel, extra="allow"):
    """Configuration for reward function processing.

    This configuration enables custom reward shaping, currently supporting DAPO-style
    penalties for responses that exceed the maximum response length threshold.
    """

    enabled: bool = False

    # The length of the buffer to penalize responses that exceed the maximum response length threshold.
    # Responses of length greater than overlong_buffer_length + max_response_length will
    # receive the maximum penalty.
    overlong_buffer_length: int | None = None

    # The penalty for responses that exceed the maximum response length threshold.
    overlong_buffer_penalty: float | None = None

    # The maximum response length threshold. Responses exceeding this length will be penalized.
    max_response_length: int | None = None

    # Stop properly penalty: scale factor for rewards of truncated responses (0-1).
    # When set to 0, truncated responses get zero reward.
    # When set to 1, no penalty is applied (default behavior).
    stop_properly_penalty_coef: float | None = None

    # Adaptive Length Penalty coefficient (Xiang et al. 2025): reward -= alp_coef * pass_rate *
    # response_length / max_response_length. Difficulty-aware — harder prompts (lower pass rate)
    # are penalized less. Mutually exclusive with the DAPO overlong / stop-properly penalties.
    alp_coef: float | None = None

    # N-gram repetition penalty: reward -= ngram_penalty_coef * max(0, distinct-n rate - threshold).
    # Additive with whichever length penalty runs (applied before the early-returning branches).
    ngram_penalty_coef: float | None = None

    # Repetition rate at or below this threshold incurs no penalty (clean responses,
    # including long-but-clean CoT, are untouched); the penalty scales with the excess.
    ngram_penalty_threshold: float = 0.0


def assistant_token_parts(message_log) -> list[torch.Tensor]:
    """All assistant-turn token tensors of one sample's message log."""
    return [m["token_ids"] for m in message_log if m["role"] == "assistant"]


def ngram_rate(gen_ids: torch.Tensor, ngram_size: int) -> float:
    """Distinct-n-gram repetition rate of one sample's generated tokens."""
    if gen_ids.numel() < ngram_size:
        return 0.0
    windows = gen_ids.unfold(0, ngram_size, 1)
    return 1.0 - torch.unique(windows, dim=0).shape[0] / windows.shape[0]


def ngram_repetition_rates(
    per_sample_token_parts: list[list[torch.Tensor]],
    ngram_size: int = 4,
) -> torch.Tensor:
    """Per-sample distinct-n-gram repetition rate of the generated tokens.

    Single source of truth for the rate definition — used by the rollout
    quality metrics and as the reward-shaping fallback when a batch carries
    message_log but no precomputed ngram_repetition_rate tensor.
    """
    rates = torch.zeros(len(per_sample_token_parts))
    for idx, parts in enumerate(per_sample_token_parts):
        if not parts:
            continue
        gen_ids = torch.cat([torch.as_tensor(p).flatten() for p in parts])
        rates[idx] = ngram_rate(gen_ids, ngram_size)
    return rates


def _response_lengths(batch: BatchedDataDict) -> list[int]:
    """Per-sample assistant response lengths.

    Prefer the slim per-sample tensor (data-plane path: message_log lives in
    TQ, slice carries response_token_lengths). Fall back to scanning
    message_log for the legacy non-data-plane caller.
    """
    response_token_lengths = batch.get("response_token_lengths")
    if response_token_lengths is not None:
        if isinstance(response_token_lengths, torch.Tensor):
            return response_token_lengths.tolist()
        return list(response_token_lengths)
    response_lengths = []
    for message_log in batch["message_log"]:
        length = None
        for message in message_log:
            if message["role"] == "assistant":
                length = message["token_ids"].shape[0]
                break
        assert length is not None, "Assistant response not found during reward shaping"
        response_lengths.append(length)
    return response_lengths


def apply_reward_shaping(
    batch: BatchedDataDict,
    cfg: RewardShapingConfig,
    pass_rate: torch.Tensor | None = None,
) -> BatchedDataDict:
    """Process rewards by applying penalties for responses exceeding max_response_length. Currently, this function only supports DAPO reward shaping as illustrated in the DAPO paper : https://arxiv.org/pdf/2503.14476.

    Nonetheless, it can be potentially extended to support any custom reward logic.
    """
    rewards = batch["total_reward"]
    if not cfg.enabled:
        return batch

    # Preserve the pre-shaping reward so downstream consumers (e.g. DAPO
    # dynamic sampling) can filter prompt groups on the raw task metric
    # rather than on length-dependent shaped rewards.
    batch["unshaped_total_reward"] = rewards.clone()
    # N-gram repetition penalty applied first so it composes additively with whichever
    # length penalty (ALP / stop-properly / DAPO) returns early below.
    ngram_penalty_coef = cfg.ngram_penalty_coef
    if ngram_penalty_coef is not None:
        ngram_rates = batch.get("ngram_repetition_rate")
        if ngram_rates is None and "message_log" in batch:
            ngram_rates = ngram_repetition_rates(
                [assistant_token_parts(ml) for ml in batch["message_log"]]
            )
        if ngram_rates is None:
            raise ValueError(
                "reward_shaping.ngram_penalty_coef is set but the batch carries "
                "neither ngram_repetition_rate nor message_log to derive it from"
            )
        threshold = cfg.ngram_penalty_threshold
        excess = (
            ngram_rates.to(device=rewards.device, dtype=rewards.dtype) - threshold
        ).clamp(min=0.0)
        rewards = rewards - ngram_penalty_coef * excess
        batch["total_reward"] = rewards

    # Adaptive Length Penalty (Xiang et al. 2025): difficulty-aware length penalty.
    alp_coef = cfg.alp_coef
    if alp_coef is not None:
        assert pass_rate is not None, (
            "reward_shaping.alp_coef is set but pass_rate was not provided"
        )
        assert cfg.max_response_length, (
            "reward_shaping.alp_coef is set but max_response_length is not configured (must be > 0)"
        )
        shadowed = [
            k
            for k in (
                "stop_properly_penalty_coef",
                "overlong_buffer_length",
                "overlong_buffer_penalty",
            )
            if getattr(cfg, k) is not None
        ]
        if shadowed:
            print(
                f"[WARN] alp_coef is set, so the following penalties are ignored: {', '.join(shadowed)}.",
                flush=True,
            )
        ell_max = cfg.max_response_length
        resp_lengths = torch.tensor(
            _response_lengths(batch),
            dtype=rewards.dtype,
            device=rewards.device,
        )
        batch["total_reward"] = (
            rewards - alp_coef * pass_rate.to(rewards.device) * resp_lengths / ell_max
        )
        return batch

    # Apply stop properly penalty if configured
    if cfg.stop_properly_penalty_coef is not None:
        stop_properly_penalty_coef = cfg.stop_properly_penalty_coef
        assert 0 <= stop_properly_penalty_coef <= 1, (
            f"stop_properly_penalty_coef must be in [0, 1], got {stop_properly_penalty_coef}"
        )
        # Warn user that DAPO overlong parameters are ignored when stop_properly_penalty_coef is set
        ignored_params = []
        if cfg.overlong_buffer_length is not None:
            ignored_params.append("overlong_buffer_length")
        if cfg.overlong_buffer_penalty is not None:
            ignored_params.append("overlong_buffer_penalty")
        if cfg.max_response_length is not None:
            ignored_params.append("max_response_length")
        if ignored_params:
            print(
                f"[WARN] stop_properly_penalty_coef is set, so the following DAPO overlong "
                f"parameters are ignored: {', '.join(ignored_params)}. "
                f"Set stop_properly_penalty_coef=null to use DAPO overlong reward shaping instead.",
                flush=True,
            )
        truncated = batch.get("truncated")
        assert truncated is not None, "truncated field not found in batch"
        if isinstance(truncated, list):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=rewards.device)
        else:
            truncated = truncated.to(device=rewards.device)

        num_truncated = truncated.sum().item()
        if num_truncated > 0:
            original_rewards = rewards.clone()
            # For truncated samples, scale the reward by stop_properly_penalty_coef
            rewards = torch.where(
                truncated, rewards * stop_properly_penalty_coef, rewards
            )
            batch["total_reward"] = rewards
            print(
                f"[INFO] stop properly penalty applied: {num_truncated}/{len(truncated)} samples truncated, "
                f"coef={stop_properly_penalty_coef}, "
                f"original_reward_mean={original_rewards[truncated].mean().item():.4f}, "
                f"shaped_reward_mean={rewards[truncated].mean().item():.4f}",
                flush=True,
            )
        else:
            print(
                "[INFO] stop properly penalty: no truncated samples (truncation_rate=0)",
                flush=True,
            )

        return batch

    # DAPO reward shaping requires overlong_buffer_length, overlong_buffer_penalty, and max_response_length to be set.
    overlong_buffer_length = cfg.overlong_buffer_length
    overlong_buffer_penalty = cfg.overlong_buffer_penalty
    max_response_length = cfg.max_response_length
    dapo_fields = (
        overlong_buffer_length,
        overlong_buffer_penalty,
        max_response_length,
    )
    if any(field is None for field in dapo_fields):
        # A repetition-only config (ngram penalty, no length penalty at all) is
        # valid and already applied above; a partial DAPO trio stays an error.
        if ngram_penalty_coef is not None and all(
            field is None for field in dapo_fields
        ):
            return batch
        raise ValueError(
            "Reward function is enabled but only DAPO reward shaping is currently supported. Please ensure overlong_buffer_length, overlong_buffer_penalty, and max_response_length are properly configured."
        )

    assert overlong_buffer_penalty >= 0, f"{overlong_buffer_penalty=} must be >=0"
    # Calculate the expected response length
    expected_response_length = max_response_length - overlong_buffer_length

    response_lengths = _response_lengths(batch)

    assert len(response_lengths) == len(rewards), (
        "The number of messages in the batch must match the number of rewards"
    )

    updated_rewards = torch.zeros_like(rewards)
    for i, message_response_length in enumerate(response_lengths):
        # Calculate the exceed length and the corresponding reward penalty
        exceed_length = message_response_length - expected_response_length
        overlong_reward = min(
            -exceed_length / overlong_buffer_length * overlong_buffer_penalty, 0
        )
        updated_rewards[i] = rewards[i] + overlong_reward

    # Update the rewards in the batch
    batch["total_reward"] = updated_rewards

    return batch
