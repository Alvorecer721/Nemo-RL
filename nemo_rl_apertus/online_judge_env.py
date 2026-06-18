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
"""NeMo-RL environment wrapper for the pluggable online-DPO judge.

Kept separate from :mod:`nemo_rl_apertus.online_judge` (which stays stdlib-only)
so the judge core is unit-testable without importing Ray / torch / nemo_rl. This
module wraps any :class:`~nemo_rl_apertus.online_judge.Judge` as an
``EnvironmentInterface`` Ray actor so it slots into ``run_multi_turn_rollout`` via
``task_to_env``, and registers itself additively (no upstream registry edits).
"""

from __future__ import annotations

from typing import Any, Optional

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl_apertus.online_judge import build_judge, judge_inputs_from_conversation

# Identifiers used to register the judge environment additively (no upstream edit).
ONLINE_DPO_JUDGE_ENV_NAME = "online_dpo_judge"
ONLINE_DPO_JUDGE_ENV_FQN = "nemo_rl_apertus.online_judge_env.JudgeEnvironment"


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class JudgeEnvironment(EnvironmentInterface):
    """Wraps any :class:`Judge` as a NeMo-RL environment.

    ``step`` formats each conversation's prompt + assistant response, calls the
    judge over the whole batch concurrently, and returns the per-sample scores as
    ``rewards`` (single-turn: always terminated). Because it holds an abstract
    ``Judge`` built via :func:`build_judge`, switching judge backends is pure config.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.judge = build_judge(cfg)
        # Reasoning aspects need the policy's thinking delimiters, which the rollout strips
        # from the assistant content; when the entry point configured a `completion_tokenizer`
        # (the policy tokenizer), load it so `step` can re-decode the rollout completion
        # keeping special tokens. Absent -> the judge sees the (stripped) content (default).
        # The env actor runs in the driver's own venv (PY_EXECUTABLES.SYSTEM), so transformers
        # is available; the import stays lazy so this module loads without it.
        self.tokenizer = None
        completion_tokenizer = cfg.get("completion_tokenizer")
        if completion_tokenizer:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                completion_tokenizer, trust_remote_code=True
            )

    def step(
        self,
        message_log_batch: list[list[dict[str, Any]]],
        metadata: list[Optional[dict[str, Any]]],
    ) -> EnvironmentReturn:
        prompts: list[str] = []
        completions: list[str] = []
        images: list[Optional[list[str]]] = []
        for conversation, meta in zip(message_log_batch, metadata):
            # The judge scores the same span the DPO loss trains on (the last
            # assistant turn), with the preceding conversation as context. See
            # judge_inputs_from_conversation for the multi-turn / clean-prompt rules.
            prompt_text, completion, sample_images = judge_inputs_from_conversation(
                conversation, meta, tokenizer=self.tokenizer
            )
            prompts.append(prompt_text)
            completions.append(completion)
            images.append(sample_images)

        scores = self.judge.score(prompts, completions, images=images)

        rewards = torch.tensor(scores, dtype=torch.float32).cpu()
        terminateds = torch.ones_like(rewards).cpu()
        observations = [
            {"role": "environment", "content": f"judge_score={score:.4f}"}
            for score in scores
        ]
        next_stop_strings: list[None] = [None] * len(scores)
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=terminateds,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        rewards = batch["rewards"]
        metrics = {
            "judge_score_mean": rewards.float().mean().item()
            if rewards.numel() > 0
            else 0.0,
        }
        return batch, metrics


def register_online_dpo_judge_env() -> None:
    """Register the judge env additively (idempotent) — avoids editing upstream registries.

    Inserts the actor → Python-env mapping (``PY_EXECUTABLES.SYSTEM``; the judge only
    needs ``openai``/``httpx``, both in the base venv) and the ``ENV_REGISTRY`` entry,
    so ``create_env('online_dpo_judge', ...)`` resolves to :class:`JudgeEnvironment`.
    """
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import ENV_REGISTRY, register_env

    ACTOR_ENVIRONMENT_REGISTRY.setdefault(
        ONLINE_DPO_JUDGE_ENV_FQN, PY_EXECUTABLES.SYSTEM
    )
    if ONLINE_DPO_JUDGE_ENV_NAME not in ENV_REGISTRY:
        register_env(ONLINE_DPO_JUDGE_ENV_NAME, ONLINE_DPO_JUDGE_ENV_FQN)
