# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use certified worker interpreters with the shared source checkout."""

import runpy

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.utils.venvs import (
    add_checkout_to_pythonpath,
    image_venv_python,
    image_venvs_enabled,
)


def configure_baked_workers() -> dict[str, str]:
    """Check every worker this bench launches against the image before Ray starts."""
    if not image_venvs_enabled():
        raise RuntimeError(
            "The bench runs on the image's worker venvs; export NEMO_RL_IMAGE_VENVS=1"
        )
    add_checkout_to_pythonpath({})
    return {
        actor: image_venv_python(get_actor_python_env(actor), actor)
        for actor in [
            "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
            "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
            "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
            "nemo_rl.environments.nemo_gym.NemoGym",
        ]
    }


if __name__ == "__main__":
    print({"certified_baked_workers": configure_baked_workers()}, flush=True)
    runpy.run_module("examples.run_grpo_single_controller", run_name="__main__")
