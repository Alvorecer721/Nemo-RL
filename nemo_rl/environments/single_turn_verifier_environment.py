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

r"""Single-turn verifier environment that grades rollouts against a gold set.

Grades each rollout by extracting its final answer (``\\boxed{}`` first, then an
``Answer:`` line) and matching it against ``{ground_truth} U answer_variants`` with a
layered comparator: symbolic (math_verify) -> numeric (epsilon) -> normalized string.
Reward is 1.0 on any match else 0.0; episodes are single-turn (terminate immediately).
"""

import contextlib
import io
import logging
import re
from typing import Any, NotRequired, Optional, TypedDict, Union

import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.dapo_math_verifier import compute_score as dapo_math_verify
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers


class SingleTurnVerifierConfig(TypedDict):
    num_workers: int
    math_verify_impl: NotRequired[str | None]


class SingleTurnVerifierMetadata(TypedDict):
    ground_truth: str
    answer_variants: list[str]


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


def _extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract the last \boxed{...} expression from text."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    num_braces = 0
    while i < len(text):
        if text[i] == "{":
            num_braces += 1
        elif text[i] == "}":
            num_braces -= 1
            if num_braces == 0:
                return text[idx + 7 : i]
        i += 1

    return None


def _extract_answer_line(text: str) -> Optional[str]:
    """Extract answer from 'Answer: ...' pattern (case-insensitive)."""
    match = re.search(r"(?i)Answer\s*:\s*([^\n]+)", text)
    if match:
        return match.group(1).strip()
    return None


def _normalize_string(s: str) -> str:
    """Normalize string: lowercase, strip whitespace, keep alphanumerics only."""
    return "".join(c.lower() for c in s if c.isalnum())


@ray.remote  # pragma: no cover
class SingleTurnVerifyWorker:
    def __init__(self) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def verify(
        self,
        pred_responses: list[str],
        metadata_list: list[SingleTurnVerifierMetadata],
        return_extracted_answer: bool = False,
        **kwargs,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        """Verify single-turn rollouts against a gold set.

        Args:
            pred_responses: predicted responses from the LLM.
            metadata_list: per-sample {ground_truth, answer_variants}.
            return_extracted_answer: whether to also return extracted answers.

        Returns:
            scores (1.0 match / 0.0 no-match), and optionally the extracted answers.
        """
        results = []
        extracted_answers: list[str | None] = []

        for response, metadata in zip(pred_responses, metadata_list):
            ground_truth = metadata["ground_truth"]
            answer_variants = metadata.get("answer_variants", [])

            extracted_answer = None
            score = 0.0

            boxed = _extract_boxed_answer(response)
            if boxed is not None:
                extracted_answer = boxed
            else:
                answer_line = _extract_answer_line(response)
                if answer_line is not None:
                    extracted_answer = answer_line

            if extracted_answer is not None:
                gold_set = {ground_truth} | set(answer_variants)
                for gold in gold_set:
                    if self._matches(
                        extracted_answer, gold, kwargs.get("math_verify_impl")
                    ):
                        score = 1.0
                        break

            results.append(score)
            if return_extracted_answer:
                extracted_answers.append(extracted_answer)

        if return_extracted_answer:
            return results, extracted_answers
        return results

    def _matches(
        self, pred: str, gold: str, math_verify_impl: Optional[str] = None
    ) -> bool:
        """Layered comparison: symbolic (math_verify) -> numeric (epsilon) -> string."""
        if not math_verify_impl or math_verify_impl == "hf_math_verify":
            try:
                with _mute_output():
                    score, _ = self.verify_func(
                        [f"\\boxed{{{gold}}}"], [f"\\boxed{{{pred}}}"]
                    )
                    if float(score) > 0.1:
                        return True
            except (Exception, TimeoutException):
                pass

        if math_verify_impl == "dapo_math_verify":
            try:
                with _mute_output():
                    result = dapo_math_verify(f"\\boxed{{{pred}}}", gold)
                    if result.get("score", 0.0) > 0.0:
                        return True
            except (Exception, TimeoutException):
                pass

        try:
            if abs(float(pred) - float(gold)) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass

        if _normalize_string(pred) == _normalize_string(gold):
            return True

        return False


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class SingleTurnVerifierEnvironment(EnvironmentInterface[SingleTurnVerifierMetadata]):
    def __init__(self, cfg: SingleTurnVerifierConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        self.workers = [
            SingleTurnVerifyWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[SingleTurnVerifierMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[SingleTurnVerifierMetadata]:
        """Grade a batch of single-turn rollouts against their gold sets.

        Returns an EnvironmentReturn with rewards in [0, 1] and all-terminated flags.
        """
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        chunked_responses = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_metadata = chunk_list_to_workers(metadata, self.num_workers)

        futures = [
            self.workers[i].verify.remote(
                resp_chunk,
                meta_chunk,
                return_extracted_answer,
                math_verify_impl=self.cfg.get("math_verify_impl", "hf_math_verify"),
            )
            for i, (resp_chunk, meta_chunk) in enumerate(
                zip(chunked_responses, chunked_metadata)
            )
        ]

        worker_results = ray.get(futures)

        results: list[float] = []
        extracted_answers: list[str | None] | None = (
            [] if return_extracted_answer else None
        )

        for worker_result in worker_results:
            worker_scores = worker_result
            if return_extracted_answer:
                worker_scores, worker_answers = worker_result
                extracted_answers.extend(worker_answers)
            results.extend(worker_scores)

        observations = [
            {
                "role": "environment",
                "content": "correct" if r > 0.5 else "incorrect",
            }
            for r in results
        ]

        rewards = torch.tensor(results, dtype=torch.float32).cpu()
        done = torch.ones(len(message_log_batch), dtype=torch.float32).cpu()
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=extracted_answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Compute environment metrics over a global rollout batch."""
        rewards = (
            batch["rewards"] if batch["rewards"].ndim == 1 else batch["rewards"][:, 0]
        )

        rewards = rewards * batch["is_end"]
        if (rewards == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[rewards == 1]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            "accuracy": rewards.mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], rewards
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
