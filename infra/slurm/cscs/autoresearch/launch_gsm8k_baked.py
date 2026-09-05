# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use certified worker interpreters with the shared source checkout."""

import runpy
from pathlib import Path

import nemo_rl
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.utils.venvs import VENV_READY_MARKER, venv_is_current


def configure_baked_workers() -> dict[str, str]:
    """Validate the original image build command before selecting its Python."""
    source = str(Path(nemo_rl.__file__).resolve().parent.parent)
    selected = {}
    for actor in [
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
        "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
    ]:
        path = Path("/opt/ray_venvs") / actor
        command = ACTOR_ENVIRONMENT_REGISTRY[actor]
        if not command.startswith("uv ") or source not in command:
            raise ValueError(f"Unexpected worker build command for {actor}: {command}")
        image_command = command.replace(source, "/opt/nemo-rl")
        if not (path / "bin/python").is_file() or not venv_is_current(
            path / VENV_READY_MARKER, image_command
        ):
            raise RuntimeError(
                f"Certified worker dependency/build fingerprint mismatch: {actor}"
            )
        selected[actor] = str(path / "bin/python")
    ACTOR_ENVIRONMENT_REGISTRY.update(selected)
    return selected


if __name__ == "__main__":
    print({"certified_baked_workers": configure_baked_workers()}, flush=True)
    runpy.run_module("examples.run_grpo_single_controller", run_name="__main__")
