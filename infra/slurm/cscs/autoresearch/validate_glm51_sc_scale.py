# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a short GLM-5.1 SingleController scaling experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from infra.slurm.cscs.autoresearch import validate_glm51_r3_10step as r3_validator


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


def validate_metrics(
    metrics: dict[str, Any], expected_steps: int, speculative_tokens: int = 0
) -> dict[str, Any]:
    if expected_steps < 1:
        raise ValueError("expected_steps must be positive")

    kl = _series(metrics, "train/gen_kl_error", expected_steps)
    token_mult = _series(metrics, "train/token_mult_prob_error", expected_steps)
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
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--speculative-tokens", type=int, default=0)
    parser.add_argument("--train-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    summary = validate_metrics(
        metrics, args.expected_steps, speculative_tokens=args.speculative_tokens
    )
    original_expected_steps = r3_validator.EXPECTED_STEPS
    try:
        r3_validator.EXPECTED_STEPS = args.expected_steps
        summary["per_token_logprob_tails"] = r3_validator.validate_logprob_tails(
            r3_validator.summarize_logprob_tails(args.train_data_dir)
        )
    finally:
        r3_validator.EXPECTED_STEPS = original_expected_steps
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
