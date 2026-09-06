# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-turn benchmark reward with independent binary episode success."""

from __future__ import annotations

import itertools
from typing import Any, Literal

import ray
import torch
from pydantic import BaseModel, Field

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.bracket_math_reward import (
    AnswerMarker,
    BracketMathScore,
    score_marked_answer,
)
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.math_environment import MathEnvironmentMetadata
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers


class BracketMathEnvConfig(BaseModel, extra="forbid"):
    """Select the existing composite reward or binary evaluation outcome."""

    num_workers: int = Field(gt=0)
    reward: Literal["composite", "outcome"]
    answer_marker: AnswerMarker


@ray.remote  # pragma: no cover
class BracketMathVerifyWorker:
    def verify(
        self, responses: list[str], targets: list[str], marker: AnswerMarker
    ) -> list[BracketMathScore]:
        """Score each response with the shared benchmark reward function."""
        return [
            score_marked_answer(response, target, marker)
            for response, target in zip(responses, targets, strict=True)
        ]


@ray.remote(
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class BracketMathEnvironment(EnvironmentInterface[MathEnvironmentMetadata]):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = BracketMathEnvConfig.model_validate(cfg)
        self._worker_counter = itertools.count()
        self.workers = [
            BracketMathVerifyWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.cfg.num_workers)
        ]

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MathEnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[MathEnvironmentMetadata]:
        responses = [
            "".join(str(m["content"]) for m in messages if m["role"] == "assistant")
            for messages in message_log_batch
        ]
        targets = [item["ground_truth"] for item in metadata]
        start = next(self._worker_counter)
        futures = [
            self.workers[(start + i) % self.cfg.num_workers].verify.remote(
                response_chunk, target_chunk, self.cfg.answer_marker
            )
            for i, (response_chunk, target_chunk) in enumerate(
                zip(
                    chunk_list_to_workers(responses, self.cfg.num_workers),
                    chunk_list_to_workers(targets, self.cfg.num_workers),
                    strict=True,
                )
            )
        ]
        scores = [score for chunk in ray.get(futures) for score in chunk]
        if len(scores) != len(message_log_batch):
            raise ValueError("Verifier returned an incomplete batch")
        successes = torch.tensor(
            [score.outcome for score in scores], dtype=torch.float32
        )
        rewards = {"reward/correctness": successes}
        if self.cfg.reward == "composite":
            rewards.update(
                {
                    "reward/format": torch.tensor([s.format for s in scores]),
                    "reward/length_penalty": torch.tensor(
                        [s.length_penalty for s in scores]
                    ),
                }
            )
        return EnvironmentReturn(
            observations=[
                {
                    "role": "environment",
                    "content": "Environment: correct"
                    if s.outcome
                    else "Environment: incorrect",
                }
                for s in scores
            ],
            metadata=metadata,
            next_stop_strings=[None] * len(scores),
            rewards=rewards,
            terminateds=torch.ones_like(successes),
            answers=[s.extracted_answer for s in scores]
            if return_extracted_answer
            else None,
            episode_successes=successes,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        correct = batch["reward/correctness"] * batch["is_end"]
        metrics = {
            "accuracy": correct.float().mean().item(),
            "mean_reward": batch["rewards"].float().mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], correct
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
        }
        return batch, metrics
