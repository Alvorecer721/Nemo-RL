# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from infra.slurm.cscs.autoresearch.validate_glm51_sc_scale import validate_metrics


def _metrics(steps: int = 10) -> dict[str, dict[str, float]]:
    indices = range(1, steps + 1)

    def series(value: float) -> dict[str, float]:
        return {str(step): value for step in indices}

    return {
        "train/gen_kl_error": series(0.0004),
        "train/token_mult_prob_error": series(1.01),
        "train/js_divergence_error": series(0.0001),
        "train/loss": series(-0.01),
        "train/reward": series(0.5),
        "train/advantages/min": series(-1.0),
        "train/advantages/max": series(1.0),
        "train/grad_norm": series(0.2),
        "timing/train/total_step_time": series(100.0),
        "timing/train/exposed_generation": series(60.0),
        "timing/train/policy_training": series(30.0),
        "timing/train/weight_sync": series(5.0),
        "timing/train/valid_tokens_per_sec_per_gpu": series(10.0),
    }


def test_validate_metrics_accepts_ten_green_steps() -> None:
    summary = validate_metrics(_metrics(), expected_steps=10)

    assert summary["steps"] == 10
    assert summary["learning_signal_steps"] == 10
    assert summary["timing"]["total_step_time"]["steady_state_mean"] == 100.0


def test_validate_metrics_accepts_eight_of_ten_learning_signal_steps() -> None:
    metrics = _metrics()
    for metric in (
        "train/loss",
        "train/advantages/min",
        "train/advantages/max",
        "train/grad_norm",
    ):
        metrics[metric]["9"] = 0.0
        metrics[metric]["10"] = 0.0

    summary = validate_metrics(metrics, expected_steps=10)

    assert summary["learning_signal_steps"] == 8
    assert summary["nonzero_loss_steps"] == 8
    assert summary["nonzero_grad_steps"] == 8


def test_validate_metrics_rejects_seven_of_ten_learning_signal_steps() -> None:
    metrics = _metrics()
    for metric in (
        "train/loss",
        "train/advantages/min",
        "train/advantages/max",
        "train/grad_norm",
    ):
        for step in (8, 9, 10):
            metrics[metric][str(step)] = 0.0

    with pytest.raises(ValueError, match="required=8"):
        validate_metrics(metrics, expected_steps=10)


@pytest.mark.parametrize(
    ("metric", "value", "match"),
    (
        ("train/gen_kl_error", 0.001, "KL escaped"),
        ("train/loss", 0.0, "Every scaling step"),
        ("train/grad_norm", 0.0, "Every scaling step"),
    ),
)
def test_validate_metrics_rejects_failed_gates(
    metric: str, value: float, match: str
) -> None:
    metrics = _metrics()
    metrics[metric] = {str(step): value for step in range(1, 11)}

    with pytest.raises(ValueError, match=match):
        validate_metrics(metrics, expected_steps=10)
