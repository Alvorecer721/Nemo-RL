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

"""Declare the uv extras used by Ray actors and image builds.

The runtime registry imports this table. Docker runs this file directly before
the full source or dependencies are installed, so it must remain stdlib-only and
must never import ``nemo_rl``. ``None`` means the driver's interpreter: these
actors stay in the runtime registry but do not need a prebuilt worker venv.

Build-time actor selection never changes runtime actor lookup.
"""

from __future__ import annotations

import argparse
import sys

ACTOR_ENVIRONMENTS: dict[str, list[str] | None] = {
    "nemo_rl.environments.bracket_math_environment.BracketMathEnvironment": None,
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": ["vllm"],
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": [
        "vllm"
    ],
    "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker": ["sglang"],
    "nemo_rl.models.generation.dynamo.dynamo_worker.DynamoVllmWorker": None,
    "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": ["fsdp"],
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": [
        "automodel"
    ],
    "nemo_rl.models.value.workers.dtensor_value_worker_v2.DTensorValueWorkerV2": [
        "automodel"
    ],
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": [
        "mcore"
    ],
    "nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker": ["mcore"],
    "nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker": [
        "trtllm"
    ],
    "nemo_rl.environments.math_environment.MathEnvironment": None,
    "nemo_rl.environments.math_environment.MathMultiRewardEnvironment": None,
    "nemo_rl.environments.vlm_environment.VLMEnvironment": None,
    "nemo_rl.environments.single_turn_verifier_environment.SingleTurnVerifierEnvironment": None,
    "nemo_rl.environments.code_environment.CodeEnvironment": None,
    "nemo_rl.environments.reward_model_environment.RewardModelEnvironment": None,
    "nemo_rl.environments.code_jaccard_environment.CodeJaccardEnvironment": None,
    "nemo_rl.environments.games.sliding_puzzle.SlidingPuzzleEnv": None,
    # The collector handles vLLM exceptions; the buffer handles its trajectory data.
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector": ["vllm"],
    "nemo_rl.algorithms.async_utils.ReplayBuffer": ["vllm"],
    # SyncRolloutActor needs transfer_queue from the vLLM environment and shares
    # same-node worker caches with the generation actors.
    "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor": ["vllm"],
    "nemo_rl.environments.tools.retriever.RAGEnvironment": None,
    "nemo_rl.environments.nemo_gym.NemoGym": ["nemo_gym"],
    "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker": [
        "modelopt",
        "vllm",
    ],
    "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker": [
        "modelopt",
        "vllm",
    ],
    "nemo_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker.DTensorQuantPolicyWorker": [
        "modelopt",
        "automodel",
    ],
    "nemo_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker_v2.DTensorQuantPolicyWorkerV2": [
        "modelopt",
        "automodel",
    ],
    "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker.MegatronQuantPolicyWorker": [
        "modelopt",
        "mcore",
    ],
}

# These existing controllers do not honor NEMO_RL_PY_EXECUTABLES_SYSTEM. Keep that
# runtime behavior while sharing their image selection and declared dependencies.
VLLM_CONTROLLER_ACTORS = frozenset(
    {
        "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
        "nemo_rl.algorithms.async_utils.ReplayBuffer",
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
    }
)


def _build_stage(extras: list[str]) -> str:
    """Select the layer that can finish an actor's third-party dependencies."""
    return "trtllm" if "trtllm" in extras else "deps"


def main(argv: list[str]) -> int:
    r"""Print sorted ``actor FQN\tstage\tuv extra flags`` build rows.

    Usage: actor_environments.py [--actors "actor FQN ..."] [stage] [skip extra ...]

    ``stage`` defaults to ``all``; ``deps`` and ``trtllm`` restrict the image layer.
    Skip extras are positional, matching the upstream Docker interface. For
    example, ``all vllm sglang`` also omits controller and ModelOpt
    actors whose names contain neither backend name but whose extras require it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actors", default="", help="space-separated actor FQNs; empty selects all"
    )
    parser.add_argument(
        "stage", choices=("all", "deps", "trtllm"), nargs="?", default="all"
    )
    parser.add_argument(
        "skip_extras", nargs="*", help="omit actors requiring any of these extras"
    )
    args = parser.parse_args(argv[1:])
    skip = set(args.skip_extras)
    declared = {
        extra for extras in ACTOR_ENVIRONMENTS.values() for extra in extras or ()
    }
    unknown = skip - declared
    if unknown:
        parser.error(f"unknown extras: {sorted(unknown)}")
    selected = set(args.actors.split()) if args.actors else set(ACTOR_ENVIRONMENTS)
    missing = selected - ACTOR_ENVIRONMENTS.keys()
    if missing:
        parser.error(f"selection contains unregistered actors: {sorted(missing)}")
    for actor in sorted(selected):
        extras = ACTOR_ENVIRONMENTS[actor]
        if extras is None or skip.intersection(extras):
            continue
        stage = _build_stage(extras)
        if args.stage != "all" and stage != args.stage:
            continue
        print(actor, stage, " ".join(f"--extra {extra}" for extra in extras), sep="\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
