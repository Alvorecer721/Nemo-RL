# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated production profile for the CSCS GLM-5.1 Router Replay gate.

This module owns the fork-specific certification envelope only. It deliberately
does not implement or wrap GRPO training; the runner delegates execution to
``examples.run_grpo`` after this profile accepts the resolved recipe.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@dataclass(frozen=True)
class GLM51R3Profile:
    """Resolved values emitted after the GLM production contract is validated."""

    num_nodes: int
    gpus_per_node: int
    rollout_nodes: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_tensor_parallel_size: int
    expert_parallel_size: int
    vllm_tensor_parallel_size: int
    vllm_expert_parallel_size: int
    max_total_sequence_length: int
    max_new_tokens: int
    max_trajectory_age_steps: int

    def describe(self) -> str:
        """Return the stable marker consumed by operators and terminal logs."""
        training_ranks = (self.num_nodes - self.rollout_nodes) * self.gpus_per_node
        dense_dp = training_ranks // (
            self.tensor_parallel_size * self.pipeline_parallel_size
        )
        expert_dp = (
            dense_dp
            * self.tensor_parallel_size
            // (self.expert_tensor_parallel_size * self.expert_parallel_size)
        )
        return (
            "glm51_r3_10step_config=OK "
            f"tp={self.tensor_parallel_size} "
            f"pp={self.pipeline_parallel_size} "
            f"etp={self.expert_tensor_parallel_size} "
            f"ep={self.expert_parallel_size} "
            f"dense_dp={dense_dp} expert_dp={expert_dp} "
            f"total_seq={self.max_total_sequence_length} "
            f"max_new={self.max_new_tokens} transport=legacy-async"
        )


def _require_equal(field: str, actual: Any, expected: Any) -> None:
    """Fail with an actionable field-level error when the profile drifts."""
    if actual != expected:
        raise ValueError(
            f"GLM-5.1 production profile requires {field}={expected!r}; "
            f"resolved {actual!r}"
        )


def load_glm51_r3_config(recipe: Path) -> MasterConfig:
    """Load and resolve a GLM recipe through NeMo-RL's canonical schema."""
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(recipe), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(
            f"Expected a mapping at the root of {recipe}; got {type(resolved)}"
        )
    return MasterConfig.model_validate(resolved)


def validate_glm51_r3_profile(config: MasterConfig) -> GLM51R3Profile:
    """Validate the exact runtime envelope certified by the 80-node gate."""
    megatron = config.policy["megatron_cfg"]
    generation = config.policy["generation"]
    vllm = generation["vllm_cfg"]
    async_grpo = config.grpo.async_grpo
    if async_grpo is None:
        raise ValueError(
            "GLM-5.1 production profile requires the legacy async GRPO schema"
        )
    if config.data_plane is None:
        raise ValueError("GLM-5.1 production profile requires data_plane configuration")

    profile = GLM51R3Profile(
        num_nodes=config.cluster["num_nodes"],
        gpus_per_node=config.cluster["gpus_per_node"],
        rollout_nodes=generation["colocated"]["resources"]["num_nodes"],
        tensor_parallel_size=megatron["tensor_model_parallel_size"],
        pipeline_parallel_size=megatron["pipeline_model_parallel_size"],
        expert_tensor_parallel_size=megatron["expert_tensor_parallel_size"],
        expert_parallel_size=megatron["expert_model_parallel_size"],
        vllm_tensor_parallel_size=vllm["tensor_parallel_size"],
        vllm_expert_parallel_size=vllm["expert_parallel_size"],
        max_total_sequence_length=config.policy["max_total_sequence_length"],
        max_new_tokens=generation["max_new_tokens"],
        max_trajectory_age_steps=async_grpo.max_trajectory_age_steps,
    )

    # These values are certification invariants, not fallback defaults. Changing
    # one requires a new matched scale gate and therefore must fail closed here.
    checks = (
        ("cluster.num_nodes", profile.num_nodes, 80),
        ("cluster.gpus_per_node", profile.gpus_per_node, 4),
        ("policy.generation.colocated.resources.num_nodes", profile.rollout_nodes, 8),
        ("policy.megatron_cfg.enabled", megatron["enabled"], True),
        (
            "policy.megatron_cfg.tensor_model_parallel_size",
            profile.tensor_parallel_size,
            2,
        ),
        (
            "policy.megatron_cfg.pipeline_model_parallel_size",
            profile.pipeline_parallel_size,
            18,
        ),
        (
            "policy.megatron_cfg.expert_tensor_parallel_size",
            profile.expert_tensor_parallel_size,
            1,
        ),
        (
            "policy.megatron_cfg.expert_model_parallel_size",
            profile.expert_parallel_size,
            16,
        ),
        ("policy.megatron_cfg.sequence_parallel", megatron["sequence_parallel"], True),
        ("policy.generation.backend", generation["backend"], "vllm"),
        (
            "policy.generation.vllm_cfg.tensor_parallel_size",
            profile.vllm_tensor_parallel_size,
            32,
        ),
        (
            "policy.generation.vllm_cfg.expert_parallel_size",
            profile.vllm_expert_parallel_size,
            32,
        ),
        (
            "policy.generation.refit_transport",
            generation["refit_transport"],
            "nccl_reshard",
        ),
        (
            "policy.router_replay.enabled",
            config.policy["router_replay"]["enabled"],
            True,
        ),
        ("policy.max_total_sequence_length", profile.max_total_sequence_length, 2048),
        ("policy.generation.max_new_tokens", profile.max_new_tokens, 1536),
        ("policy.generation.vllm_cfg.max_model_len", vllm["max_model_len"], 2048),
        ("grpo.max_num_steps", config.grpo.max_num_steps, 10),
        ("grpo.async_grpo.enabled", async_grpo.enabled, True),
        (
            "grpo.async_grpo.max_trajectory_age_steps",
            profile.max_trajectory_age_steps,
            1,
        ),
        ("data_plane.enabled", config.data_plane["enabled"], False),
        ("checkpointing.enabled", config.checkpointing["enabled"], False),
        ("checkpointing.save_optimizer", config.checkpointing["save_optimizer"], False),
    )
    for field, actual, expected in checks:
        _require_equal(field, actual, expected)

    return profile


def parse_args() -> argparse.Namespace:
    """Parse the standalone preflight command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Validate a recipe and print its stable production-profile marker."""
    args = parse_args()
    profile = validate_glm51_r3_profile(load_glm51_r3_config(args.config))
    print(profile.describe(), flush=True)


if __name__ == "__main__":
    main()
