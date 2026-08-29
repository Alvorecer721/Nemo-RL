# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-fast contracts for the legacy/synchronous GRPO entrypoint."""

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.models.generation.interfaces import should_use_async_rollouts


def validate_grpo_entrypoint_contract(master_config: MasterConfig) -> None:
    """Reject configs that select a runtime ``run_grpo.py`` cannot honor."""
    async_config = master_config.grpo.async_grpo
    if async_config is None:
        raise ValueError(
            "examples/run_grpo.py requires grpo.async_grpo to be present. "
            "A null block selects the SingleController schema; launch it with "
            "examples/run_grpo_single_controller.py instead."
        )

    # MasterConfig accepts extension fields, so an SC block otherwise parses and
    # is silently ignored by both legacy and synchronous trainers.
    if getattr(master_config, "async_rl", None) is not None:
        raise ValueError(
            "async_rl.* is consumed only by examples/run_grpo_single_controller.py; "
            "examples/run_grpo.py would ignore it. Remove async_rl or use the "
            "SingleController entrypoint with grpo.async_grpo: null."
        )

    if not async_config.enabled:
        return

    if (master_config.data_plane or {}).get("enabled", False):
        raise ValueError(
            "Legacy async GRPO does not support data_plane.enabled=true. It uses "
            "the in-memory ReplayBuffer, while TransferQueue async training is "
            "owned by SingleController. Set data_plane.enabled=false, or use "
            "examples/run_grpo_single_controller.py with grpo.async_grpo: null."
        )

    generation_config = master_config.policy.get("generation")
    backend = generation_config.get("backend", "") if generation_config else ""
    if backend not in ("vllm", "megatron", "trtllm", "dynamo"):
        raise ValueError(
            "Legacy async GRPO supports vLLM, Megatron, TRT-LLM, and Dynamo "
            f"generation; got policy.generation.backend={backend!r}."
        )
    if not should_use_async_rollouts(generation_config):
        raise ValueError(
            "Legacy async GRPO requires an async generation engine. Enable the "
            "selected backend's async engine before launching."
        )

    unsupported: list[str] = []
    if master_config.grpo.use_dynamic_sampling:
        unsupported.append("grpo.use_dynamic_sampling")
    if master_config.grpo.reward_scaling.enabled:
        unsupported.append("grpo.reward_scaling.enabled")
    if master_config.grpo.reward_shaping.enabled:
        unsupported.append("grpo.reward_shaping.enabled")
    if master_config.data["use_multiple_dataloader"]:
        unsupported.append("data.use_multiple_dataloader")
    if unsupported:
        raise NotImplementedError(
            "Legacy async GRPO does not consume these enabled settings: "
            + ", ".join(unsupported)
            + ". Disable them or use synchronous GRPO."
        )
