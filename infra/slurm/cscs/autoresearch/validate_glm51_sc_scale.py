# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a short GLM-5.1 SingleController scaling experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from infra.slurm.cscs.autoresearch import validate_glm51_r3_10step as r3_validator
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@dataclass(frozen=True)
class GLM51ScaleProfile:
    """Resolved values emitted after the scale configuration is certified."""

    total_nodes: int
    gpus_per_node: int
    generation_nodes: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_tensor_parallel_size: int
    expert_parallel_size: int
    vllm_tensor_parallel_size: int
    vllm_expert_parallel_size: int
    max_total_sequence_length: int
    max_new_tokens: int
    sampler: str
    steps: int
    min_groups_for_streaming_train: int
    speculative_method: str
    speculative_tokens: int
    fused_linear_logprobs: bool
    fused_linear_logprobs_chunk_size: int
    overlap_param_gather: bool

    def describe(self) -> str:
        """Return the stable configuration marker consumed by operators."""
        training_ranks = (self.total_nodes - self.generation_nodes) * self.gpus_per_node
        dense_dp = training_ranks // (
            self.tensor_parallel_size * self.pipeline_parallel_size
        )
        expert_dp = training_ranks // (
            self.expert_tensor_parallel_size
            * self.expert_parallel_size
            * self.pipeline_parallel_size
        )
        vllm_dp = (
            self.generation_nodes * self.gpus_per_node
        ) // self.vllm_tensor_parallel_size
        return (
            "glm51_sc_scale_config=OK "
            f"tp={self.tensor_parallel_size} "
            f"pp={self.pipeline_parallel_size} "
            f"etp={self.expert_tensor_parallel_size} "
            f"ep={self.expert_parallel_size} "
            f"dense_dp={dense_dp} expert_dp={expert_dp} "
            f"total_seq={self.max_total_sequence_length} "
            f"max_new={self.max_new_tokens} "
            f"vllm_tp={self.vllm_tensor_parallel_size} vllm_dp={vllm_dp} "
            f"transport=transfer-queue sampler={self.sampler} steps={self.steps} "
            f"stream_min_groups={self.min_groups_for_streaming_train} "
            f"spec_method={self.speculative_method} "
            f"spec_tokens={self.speculative_tokens} "
            f"fused_logprobs={str(self.fused_linear_logprobs).lower()} "
            f"logprob_chunk={self.fused_linear_logprobs_chunk_size} "
            f"overlap_param_gather={str(self.overlap_param_gather).lower()}"
        )


def _require_equal(field: str, actual: Any, expected: Any) -> None:
    """Fail with an actionable field-level error when the profile drifts."""
    if actual != expected:
        raise ValueError(
            f"GLM-5.1 scale profile requires {field}={expected!r}; resolved {actual!r}"
        )


def _require(condition: bool, message: str) -> None:
    """Fail closed on a non-equality certification invariant."""
    if not condition:
        raise ValueError(f"GLM-5.1 scale profile requires {message}")


def load_scale_config(recipe: Path) -> MasterConfig:
    """Load and resolve a scale recipe through the SingleController schema."""
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(recipe), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(
            f"Expected a mapping at the root of {recipe}; got {type(resolved)}"
        )
    return MasterConfig.model_validate(resolved)


def validate_scale_config(
    config: MasterConfig,
    *,
    expected_total_nodes: int,
    expected_generation_nodes: int,
    expected_steps: int,
    expected_sampler: str,
    expected_min_groups_for_streaming_train: int,
    expected_speculative_tokens: int,
    expected_speculative_method: str,
    expected_fused_linear_logprobs: bool,
) -> GLM51ScaleProfile:
    """Validate the exact runtime envelope certified by the scale gate."""
    megatron = config.policy["megatron_cfg"]
    generation = config.policy["generation"]
    vllm = generation["vllm_cfg"]
    profile = GLM51ScaleProfile(
        total_nodes=config.cluster["num_nodes"],
        gpus_per_node=config.cluster["gpus_per_node"],
        generation_nodes=generation["colocated"]["resources"]["num_nodes"],
        tensor_parallel_size=megatron["tensor_model_parallel_size"],
        pipeline_parallel_size=megatron["pipeline_model_parallel_size"],
        expert_tensor_parallel_size=megatron["expert_tensor_parallel_size"],
        expert_parallel_size=megatron["expert_model_parallel_size"],
        vllm_tensor_parallel_size=vllm["tensor_parallel_size"],
        vllm_expert_parallel_size=vllm["expert_parallel_size"],
        max_total_sequence_length=config.policy["max_total_sequence_length"],
        max_new_tokens=generation["max_new_tokens"],
        sampler=config.async_rl.sampler.name,
        steps=config.grpo.max_num_steps,
        min_groups_for_streaming_train=(config.async_rl.min_groups_for_streaming_train),
        speculative_method=expected_speculative_method,
        speculative_tokens=expected_speculative_tokens,
        fused_linear_logprobs=megatron["use_fused_linear_logprobs"],
        fused_linear_logprobs_chunk_size=megatron["fused_linear_logprobs_chunk_size"],
        overlap_param_gather=megatron["distributed_data_parallel_config"][
            "overlap_param_gather"
        ],
    )

    # These are certification invariants, not fallback defaults. Changing one
    # requires a new matched scale gate and must fail closed before allocation use.
    checks = (
        ("cluster.num_nodes", profile.total_nodes, expected_total_nodes),
        ("cluster.gpus_per_node", profile.gpus_per_node, 4),
        (
            "policy.generation.colocated.resources.num_nodes",
            profile.generation_nodes,
            expected_generation_nodes,
        ),
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
        ("training_nodes", profile.total_nodes - profile.generation_nodes, 72),
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
        ("policy.max_total_sequence_length", profile.max_total_sequence_length, 4096),
        ("policy.generation.max_new_tokens", profile.max_new_tokens, 3584),
        ("policy.generation.vllm_cfg.max_model_len", vllm["max_model_len"], 4096),
        (
            "policy.generation.vllm_cfg.gpu_memory_utilization",
            vllm["gpu_memory_utilization"],
            0.60,
        ),
        ("grpo.max_num_steps", profile.steps, expected_steps),
        ("grpo.async_grpo", config.grpo.async_grpo, None),
        ("grpo.use_dynamic_sampling", config.grpo.use_dynamic_sampling, False),
        (
            "loss_fn.use_importance_sampling_correction",
            config.loss_fn.use_importance_sampling_correction,
            True,
        ),
        ("loss_fn.force_on_policy_ratio", config.loss_fn.force_on_policy_ratio, False),
        ("data_plane.enabled", config.data_plane["enabled"], True),
        ("data_plane.impl", config.data_plane["impl"], "transfer_queue"),
        ("data_plane.backend", config.data_plane["backend"], "simple"),
        ("async_rl.sampler.name", profile.sampler, expected_sampler),
        (
            "async_rl.sampler.max_staleness_versions",
            config.async_rl.sampler.max_staleness_versions,
            1,
        ),
        (
            "async_rl.min_groups_for_streaming_train",
            profile.min_groups_for_streaming_train,
            expected_min_groups_for_streaming_train,
        ),
        ("async_rl.max_inflight_prompts", config.async_rl.max_inflight_prompts, 32),
        ("async_rl.max_buffered_rollouts", config.async_rl.max_buffered_rollouts, 128),
        ("checkpointing.enabled", config.checkpointing["enabled"], False),
        (
            "policy.megatron_cfg.use_fused_linear_logprobs",
            profile.fused_linear_logprobs,
            expected_fused_linear_logprobs,
        ),
        (
            "policy.megatron_cfg.fused_linear_logprobs_chunk_size",
            profile.fused_linear_logprobs_chunk_size,
            256,
        ),
        (
            "policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather",
            profile.overlap_param_gather,
            not expected_fused_linear_logprobs,
        ),
    )
    for field, actual, expected in checks:
        _require_equal(field, actual, expected)

    generation_ranks = profile.generation_nodes * profile.gpus_per_node
    _require(
        generation_ranks % profile.vllm_tensor_parallel_size == 0,
        "generation GPU count to be divisible by the vLLM tensor parallel size; "
        f"resolved {generation_ranks} GPUs and TP={profile.vllm_tensor_parallel_size}",
    )

    speculative = generation.get("vllm_kwargs", {}).get("speculative_config")
    if expected_speculative_tokens:
        _require(speculative is not None, "speculative_config to be present")
        _require_equal(
            "policy.megatron_cfg.mtp_num_layers",
            megatron.get("mtp_num_layers"),
            0,
        )
        _require_equal(
            "policy.generation.vllm_kwargs.speculative_config.method",
            speculative.get("method"),
            expected_speculative_method,
        )
        _require_equal(
            "policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens",
            speculative["num_speculative_tokens"],
            expected_speculative_tokens,
        )
        # vLLM 0.25.1 DeepSeekMTP has no get_top_tokens() and rejects this
        # optimization during worker startup.
        _require_equal(
            "policy.generation.vllm_kwargs.speculative_config.use_local_argmax_reduction",
            speculative.get("use_local_argmax_reduction", False),
            False,
        )

        model_dir = Path(config.policy["model_name"])
        model_config = json.loads((model_dir / "config.json").read_text())
        _require_equal(
            "model.config.model_type", model_config["model_type"], "glm_moe_dsa"
        )
        _require(
            model_config["num_nextn_predict_layers"] >= 1,
            "model.config.num_nextn_predict_layers >= 1; "
            f"resolved {model_config['num_nextn_predict_layers']!r}",
        )
        mtp_start = model_config["num_hidden_layers"]
        weight_map = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        _require(
            any(name.startswith(f"model.layers.{mtp_start}.") for name in weight_map),
            f"MTP weights under model.layers.{mtp_start}.*",
        )
    else:
        _require_equal(
            "policy.generation.vllm_kwargs.speculative_config", speculative, None
        )
        _require_equal(
            "expected_speculative_method", expected_speculative_method, "none"
        )

    validate_single_controller_config(config)
    return profile


def _series(metrics: dict[str, Any], name: str, expected_steps: int) -> list[float]:
    values = metrics.get(name)
    if not isinstance(values, dict):
        raise ValueError(f"Missing metric series: {name}")
    steps = sorted((int(step), float(value)) for step, value in values.items())
    expected = list(range(1, expected_steps + 1))
    if [step for step, _ in steps] != expected:
        raise ValueError(
            f"{name} must contain exactly steps {expected}; "
            f"got {[step for step, _ in steps]}"
        )
    series = [value for _, value in steps]
    if not all(math.isfinite(value) for value in series):
        raise ValueError(f"{name} contains non-finite values: {series}")
    return series


def _validate_speculative_metrics(
    metrics: dict[str, Any], expected_steps: int, speculative_tokens: int
) -> dict[str, Any]:
    names = (
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "acceptance_length",
        "acceptance_rate",
    )
    spec = {
        name: _series(metrics, f"train/vllm/spec_{name}", expected_steps)
        for name in names
    }
    for step, (drafts, draft_tokens, accepted, length, rate) in enumerate(
        zip(*(spec[name] for name in names), strict=True), start=1
    ):
        if min(drafts, draft_tokens, accepted) < 0:
            raise ValueError(f"Negative speculative counter at step {step}")
        if accepted > draft_tokens:
            raise ValueError(
                f"Accepted speculative tokens exceed drafts at step {step}"
            )
        if not 0 <= rate <= 1:
            raise ValueError(
                f"Invalid speculative acceptance rate at step {step}: {rate}"
            )
        if not 1 <= length <= speculative_tokens + 1:
            raise ValueError(
                f"Invalid speculative acceptance length at step {step}: {length}"
            )
        expected_rate = accepted / draft_tokens if draft_tokens else 0.0
        expected_length = 1 + accepted / drafts if drafts else 1.0
        if not math.isclose(rate, expected_rate, rel_tol=1.0e-6, abs_tol=1.0e-9):
            raise ValueError(f"Inconsistent speculative acceptance rate at step {step}")
        if not math.isclose(length, expected_length, rel_tol=1.0e-6, abs_tol=1.0e-9):
            raise ValueError(
                f"Inconsistent speculative acceptance length at step {step}"
            )

    total_drafts = sum(spec["num_drafts"])
    total_draft_tokens = sum(spec["num_draft_tokens"])
    total_accepted = sum(spec["num_accepted_tokens"])
    if total_drafts <= 0 or total_draft_tokens <= 0:
        raise ValueError("Speculative decoding produced no draft traffic")
    if total_accepted <= 0:
        raise ValueError("Speculative decoding accepted no draft tokens")
    return {
        "tokens_per_draft": speculative_tokens,
        "total_drafts": total_drafts,
        "total_draft_tokens": total_draft_tokens,
        "total_accepted_tokens": total_accepted,
        "aggregate_acceptance_rate": total_accepted / total_draft_tokens,
        "aggregate_acceptance_length": 1 + total_accepted / total_drafts,
        "per_step": spec,
    }


def _validate_logprob_tail_metrics(
    metrics: dict[str, Any], expected_steps: int
) -> dict[str, Any]:
    """Validate SingleController's compact per-token Router Replay evidence."""
    names = (
        "valid_tokens",
        "mean_abs",
        "p95_abs",
        "p99_abs",
        "max_abs",
        "count_gt_0_5",
        "count_gt_1_0",
    )
    tails = {
        name: _series(metrics, f"train/logprob_tails/{name}", expected_steps)
        for name in names
    }

    per_step: dict[str, dict[str, float | int]] = {}
    for step, values in enumerate(
        zip(*(tails[name] for name in names), strict=True), start=1
    ):
        tokens, mean_abs, p95_abs, p99_abs, max_abs, gt_0_5, gt_1_0 = values
        integer_values = (tokens, gt_0_5, gt_1_0)
        if any(value < 0 or not value.is_integer() for value in integer_values):
            raise ValueError(
                f"Invalid log-probability tail counters at step {step}: "
                f"tokens={tokens}, count_gt_0_5={gt_0_5}, count_gt_1_0={gt_1_0}"
            )
        if tokens <= 0 or not (0 <= gt_1_0 <= gt_0_5 <= tokens):
            raise ValueError(
                f"Inconsistent log-probability tail counters at step {step}: "
                f"tokens={tokens}, count_gt_0_5={gt_0_5}, count_gt_1_0={gt_1_0}"
            )
        if not (0 <= p95_abs <= p99_abs <= max_abs) or mean_abs < 0:
            raise ValueError(
                f"Inconsistent log-probability tail magnitudes at step {step}: "
                f"mean={mean_abs}, p95={p95_abs}, p99={p99_abs}, max={max_abs}"
            )
        per_step[str(step)] = {
            "tokens": int(tokens),
            "mean_abs": mean_abs,
            "p95_abs": p95_abs,
            "p99_abs": p99_abs,
            "max_abs": max_abs,
            "count_gt_0_5": int(gt_0_5),
            "count_gt_1_0": int(gt_1_0),
        }

    nonfinite_name = "train/logprob_tails/count_nonfinite"
    if isinstance(metrics.get(nonfinite_name), dict):
        nonfinite = _series(metrics, nonfinite_name, expected_steps)
        if any(value != 0 for value in nonfinite):
            raise ValueError(
                f"Non-finite log-probability deltas were logged: {nonfinite}"
            )

    summary = {
        "per_step": per_step,
        "total_tokens": sum(int(value) for value in tails["valid_tokens"]),
        "count_gt_0_5": sum(int(value) for value in tails["count_gt_0_5"]),
        "count_gt_1_0": sum(int(value) for value in tails["count_gt_1_0"]),
    }
    return r3_validator.validate_logprob_tails(summary)


def validate_metrics(
    metrics: dict[str, Any], expected_steps: int, speculative_tokens: int = 0
) -> dict[str, Any]:
    if expected_steps < 1:
        raise ValueError("expected_steps must be positive")

    kl = _series(metrics, "train/gen_kl_error", expected_steps)
    token_mult = _series(metrics, "train/token_mult_prob_error", expected_steps)
    max_seq_mult = _series(metrics, "train/max_seq_mult_prob_error", expected_steps)
    js = _series(metrics, "train/js_divergence_error", expected_steps)
    loss = _series(metrics, "train/loss", expected_steps)
    reward = _series(metrics, "train/reward", expected_steps)
    advantage_min = _series(metrics, "train/advantages/min", expected_steps)
    advantage_max = _series(metrics, "train/advantages/max", expected_steps)
    grad_norm = _series(metrics, "train/grad_norm", expected_steps)

    if min(kl) < 0 or max(kl) >= r3_validator.GLM_KL_SAFETY_CEILING:
        raise ValueError(
            "Generation KL escaped the GLM safety envelope: "
            f"min={min(kl):.6g}, max={max(kl):.6g}"
        )
    token_mult_median = statistics.median(token_mult)
    if min(token_mult) < 1 or (
        token_mult_median >= r3_validator.R3_TOKEN_MULT_MEDIAN_CEILING
    ):
        raise ValueError(
            "Router Replay probability parity failed: "
            f"min={min(token_mult):.6g}, median={token_mult_median:.6g}"
        )
    # Bound the per-sequence mean of exp(abs(delta log p)) so one extreme token
    # cannot hide inside an otherwise clean sequence.
    if min(max_seq_mult) < 1 or max(max_seq_mult) >= 1.1:
        raise ValueError(
            "Router Replay sequence probability parity failed: "
            f"min={min(max_seq_mult):.6g}, max={max(max_seq_mult):.6g}, "
            "required 1 <= value < 1.1"
        )
    if min(js) < 0:
        raise ValueError(f"JS divergence cannot be negative: min={min(js):.6g}")

    signal_steps = sum(
        low < 0 < high for low, high in zip(advantage_min, advantage_max, strict=True)
    )
    nonzero_loss_steps = sum(abs(value) > 1.0e-12 for value in loss)
    nonzero_grad_steps = sum(value > 0 for value in grad_norm)
    required_signal_steps = min(8, expected_steps)
    if (
        min(signal_steps, nonzero_loss_steps, nonzero_grad_steps)
        < required_signal_steps
    ):
        raise ValueError(
            "Scaling run did not provide enough learning-signal steps: "
            f"advantages={signal_steps}, loss={nonzero_loss_steps}, "
            f"grad={nonzero_grad_steps}, required={required_signal_steps}, "
            f"total={expected_steps}"
        )
    if max(reward) <= 0:
        raise ValueError("No positive reward was observed")

    timing_names = (
        "timing/train/total_step_time",
        "timing/train/exposed_generation",
        "timing/train/policy_training",
        "timing/train/weight_sync",
        "timing/train/valid_tokens_per_sec_per_gpu",
    )
    timing = {
        name.removeprefix("timing/train/"): _series(metrics, name, expected_steps)
        for name in timing_names
    }
    summary = {
        "steps": expected_steps,
        "kl": {
            "min": min(kl),
            "max": max(kl),
            "mean": statistics.fmean(kl),
        },
        "token_mult_prob_error_median": token_mult_median,
        "max_seq_mult_prob_error": max(max_seq_mult),
        "learning_signal_steps": signal_steps,
        "nonzero_loss_steps": nonzero_loss_steps,
        "nonzero_grad_steps": nonzero_grad_steps,
        "positive_reward_steps": sum(value > 0 for value in reward),
        "timing": {
            name: {
                "values": values,
                "mean": statistics.fmean(values),
                "steady_state_mean": statistics.fmean(values[1:])
                if len(values) > 1
                else values[0],
            }
            for name, values in timing.items()
        },
    }
    if speculative_tokens:
        summary["speculative_decoding"] = _validate_speculative_metrics(
            metrics, expected_steps, speculative_tokens
        )
    summary["per_token_logprob_tails"] = _validate_logprob_tail_metrics(
        metrics, expected_steps
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--speculative-tokens", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    summary = validate_metrics(
        metrics, args.expected_steps, speculative_tokens=args.speculative_tokens
    )
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
