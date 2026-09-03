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
"""Single-turn environment scoring marked final answers with the benchmark reward.

Same batching, worker fan-out and metadata contract as ``MathEnvironment``; the
reward is :func:`nemo_rl.environments.bracket_math_reward.score_marked_answer`,
reading the last ``[[[answer]]]`` or ``\\boxed{answer}`` span depending on
``answer_marker``.
The reward is composite (outcome + format bonus - length penalty), so the
accuracy metric is computed from the outcome component rather than from the
scalar reward.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal, NotRequired, TypedDict, Union

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.bracket_math_reward import AnswerMarker, score_marked_answer
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.math_environment import MathEnvironmentMetadata
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers


RewardMode = Literal["composite", "outcome"]


class BracketMathEnvConfig(TypedDict):
    num_workers: int
    # "composite" is the benchmark training reward (outcome + format bonus -
    # length penalty); "outcome" is the plain 0/1 correctness used for evaluation.
    reward: RewardMode
    answer_marker: AnswerMarker
    stop_strings: NotRequired[list[str] | None]


@ray.remote  # pragma: no cover
class BracketMathVerifyWorker:
    def __init__(self, reward_mode: RewardMode, answer_marker: AnswerMarker) -> None:
        self.reward_mode = reward_mode
        self.answer_marker = answer_marker

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        return_extracted_answer: bool = False,
        **kwargs: Any,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        scores = [
            score_marked_answer(response, ground_truth, self.answer_marker)
            for response, ground_truth in zip(pred_responses, ground_truths)
        ]
        if self.reward_mode == "outcome":
            rewards = [score.outcome for score in scores]
        else:
            rewards = [score.reward for score in scores]
        if return_extracted_answer:
            return rewards, [score.extracted_answer for score in scores]
        return rewards


@ray.remote(
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class BracketMathEnvironment(EnvironmentInterface[MathEnvironmentMetadata]):
    def __init__(self, cfg: BracketMathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.reward_mode: RewardMode = cfg["reward"]
        if self.reward_mode not in ("composite", "outcome"):
            raise ValueError(
                f"env.bracket_math.reward must be 'composite' or 'outcome', got {self.reward_mode!r}"
            )
        self.answer_marker: AnswerMarker = cfg["answer_marker"]
        if self.answer_marker not in ("bracket", "boxed"):
            raise ValueError(
                f"env.bracket_math.answer_marker must be 'bracket' or 'boxed', got {self.answer_marker!r}"
            )
        self._worker_counter = itertools.count()
        self.workers = [
            BracketMathVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(self.reward_mode, self.answer_marker)
            for _ in range(self.num_workers)
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
        assistant_response_batch = [
            "".join(
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            )
            for conversation in message_log_batch
        ]
        ground_truths = [g["ground_truth"] for g in metadata]
        worker_index = next(self._worker_counter) % self.num_workers
        futures = [
            self.workers[(worker_index + i) % self.num_workers].verify.remote(
                chunk, ground_truth_chunk, return_extracted_answer
            )
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(
                    chunk_list_to_workers(assistant_response_batch, self.num_workers),
                    chunk_list_to_workers(ground_truths, self.num_workers),
                )
            )
        ]
        results: list[float] = []
        extracted_answers: list[str | None] | None = (
            [] if return_extracted_answer else None
        )
        for worker_result in ray.get(futures):
            if return_extracted_answer:
                worker_scores, worker_answers = worker_result
                extracted_answers.extend(worker_answers)
            else:
                worker_scores = worker_result
            results.extend(worker_scores)
        rewards = torch.tensor(results, dtype=torch.float32).cpu()
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if reward >= 1.0
                else "Environment: incorrect",
            }
            for reward in results
        ]
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards,
            terminateds=torch.ones_like(rewards).cpu(),
            answers=extracted_answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        rewards = batch["rewards"] * batch["is_end"]
        correct = (rewards >= 1.0).float()
        response_lengths = (
            batch["generation_lengths"] - batch["prompt_lengths"]
        ).float()
        metrics = {
            "accuracy": correct.mean().item(),
            "mean_reward": rewards.mean().item(),
            "format_rate": ((rewards > 0) | (rewards >= 1.0)).float().mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], correct
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": (
                response_lengths[correct.bool()].mean().item()
                if correct.sum() > 0
                else 0
            ),
        }
        return batch, metrics
