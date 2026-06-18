# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apertus format-reward GRPO environment.

Rewards structural well-formedness of the rollout, not content correctness, so it
can target the doom-loop / unclosed-thinking failure modes surfaced in the SFT
vibe test without requiring a judge or verifier:

  reward = 1.0
         - unclosed_thinking_penalty   if <|inner_prefix|> is in the response but <|inner_suffix|> is not
         - repetition_penalty * ratio  where ratio scales the top n-gram repeat count between rep_threshold and rep_full_weight_at
  reward = clip(reward, [clip_min, clip_max])

Token IDs (not text) drive the detectors: we re-tokenize the assistant's content
with the Apertus tokenizer (`add_special_tokens=False`) so special tokens round-trip
even if vLLM filtered them at decode.
"""

from collections import Counter
from typing import Any, NotRequired, TypedDict

import ray
import torch
from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

INNER_PREFIX = "<|inner_prefix|>"
INNER_SUFFIX = "<|inner_suffix|>"


class ApertusFormatEnvConfig(TypedDict):
    tokenizer_path: str
    rep_ngram: NotRequired[int]
    rep_threshold: NotRequired[int]
    rep_full_weight_at: NotRequired[int]
    unclosed_thinking_penalty: NotRequired[float]
    repetition_penalty: NotRequired[float]
    clip_min: NotRequired[float]
    clip_max: NotRequired[float]


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class ApertusFormatEnvironment(EnvironmentInterface[dict[str, Any]]):
    def __init__(self, cfg: ApertusFormatEnvConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"])
        self.inner_prefix_id = self.tokenizer.convert_tokens_to_ids(INNER_PREFIX)
        self.inner_suffix_id = self.tokenizer.convert_tokens_to_ids(INNER_SUFFIX)
        self.rep_n = int(cfg.get("rep_ngram", 5))
        # n-gram counts ≤ rep_threshold are "normal" (e.g. shared LaTeX env markers in a math derivation);
        # repetition_penalty grows linearly from rep_threshold and saturates at rep_full_weight_at.
        self.rep_threshold = int(cfg.get("rep_threshold", 3))
        self.rep_full_weight_at = int(cfg.get("rep_full_weight_at", 23))
        self.unclosed_thinking_penalty = float(cfg.get("unclosed_thinking_penalty", 0.5))
        self.repetition_penalty = float(cfg.get("repetition_penalty", 0.5))
        self.clip_min = float(cfg.get("clip_min", 0.0))
        self.clip_max = float(cfg.get("clip_max", 1.0))

    def shutdown(self) -> None:
        return None

    def _score(self, response_text: str) -> tuple[float, dict[str, float]]:
        ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]
        reward = 1.0
        unclosed = 0.0
        if self.inner_prefix_id in ids and self.inner_suffix_id not in ids:
            reward -= self.unclosed_thinking_penalty
            unclosed = 1.0

        top_count = 0
        if len(ids) >= self.rep_n:
            counter = Counter(
                tuple(ids[i : i + self.rep_n]) for i in range(len(ids) - self.rep_n + 1)
            )
            top_count = max(counter.values())
        denom = max(1, self.rep_full_weight_at - self.rep_threshold)
        rep_ratio = max(0.0, (top_count - self.rep_threshold) / denom)
        rep_ratio = min(1.0, rep_ratio)
        reward -= self.repetition_penalty * rep_ratio

        reward = max(self.clip_min, min(self.clip_max, reward))
        return reward, {
            "unclosed_thinking": unclosed,
            "top_ngram_count": float(top_count),
            "repetition_ratio": rep_ratio,
            "n_tokens": float(len(ids)),
        }

    def step(
        self,
        message_log_batch: list[list[dict[str, Any]]],
        metadata: list[dict[str, Any]],
    ) -> EnvironmentReturn[dict[str, Any]]:
        rewards: list[float] = []
        observations: list[dict[str, str]] = []
        for conversation in message_log_batch:
            assistant_text = "".join(
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            )
            reward, _ = self._score(assistant_text)
            rewards.append(reward)
            observations.append(
                {"role": "environment", "content": f"format_reward={reward:.3f}"}
            )

        rewards_t = torch.tensor(rewards, dtype=torch.float32).cpu()
        terminateds = torch.ones_like(rewards_t).cpu()
        next_stop_strings = [None] * len(message_log_batch)
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_t,
            terminateds=terminateds,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        metrics: dict[str, float | int] = {}
        if "total_reward" in batch.keys():
            rewards = batch["total_reward"]
            if hasattr(rewards, "float"):
                rewards = rewards.float().cpu().tolist()
            if rewards:
                metrics["format_reward/mean"] = float(sum(rewards) / len(rewards))
                metrics["format_reward/clean_rate"] = float(
                    sum(1.0 for r in rewards if r >= self.clip_max - 1e-6) / len(rewards)
                )
        return batch, metrics
