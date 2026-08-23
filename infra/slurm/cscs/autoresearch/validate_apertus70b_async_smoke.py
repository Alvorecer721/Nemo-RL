#!/usr/bin/env python3
"""Validate Apertus-1.5 70B async-GRPO terminal and learning evidence."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


REQUIRED_TAGS = {
    "valid_tokens": "train/global_valid_toks",
    "grad_norm": "train/grad_norm",
    "loss": "train/loss",
    "reward_min": "train/min_total_reward",
    "reward_max": "train/max_total_reward",
    "advantage_min": "train/advantages/min",
    "advantage_max": "train/advantages/max",
    "trajectory_age": "train/avg_trajectory_age",
    "step_time_s": "timing/train/total_step_time",
    "gen_kl_error": "train/gen_kl_error",
}


def _load_scalars(log_dir: Path) -> dict[str, list[tuple[float, int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    scalars: dict[str, list[tuple[float, int, float]]] = defaultdict(list)
    event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise AssertionError(f"No TensorBoard event files under {log_dir}")
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                scalars[tag].append((event.wall_time, event.step, event.value))
    return scalars


def _values(
    scalars: dict[str, list[tuple[float, int, float]]],
    tag: str,
    expected_steps: set[int],
) -> dict[int, float]:
    latest: dict[int, tuple[float, float]] = {}
    for wall_time, step, value in scalars.get(tag, []):
        if step not in latest or wall_time >= latest[step][0]:
            latest[step] = (wall_time, value)
    missing = expected_steps - latest.keys()
    if missing:
        raise AssertionError(f"{tag} missing steps {sorted(missing)}")
    return {step: latest[step][1] for step in sorted(expected_steps)}


def validate_smoke_metrics(
    scalars: dict[str, list[tuple[float, int, float]]],
    *,
    steps: int,
    max_gen_kl: float = 0.002,
) -> dict[str, dict[int, float]]:
    """Validate per-step learning evidence using NeMo-RL's emitted scalar tags."""
    expected_steps = set(range(1, steps + 1))
    metrics = {
        name: _values(scalars, tag, expected_steps)
        for name, tag in REQUIRED_TAGS.items()
    }
    for name in ("valid_tokens", "grad_norm", "step_time_s"):
        if not all(
            math.isfinite(value) and value > 0 for value in metrics[name].values()
        ):
            raise AssertionError(f"Invalid {name}: {metrics[name]}")
    if not all(math.isfinite(value) for value in metrics["loss"].values()):
        raise AssertionError(f"Invalid losses: {metrics['loss']}")
    if not any(abs(value) > 1e-12 for value in metrics["loss"].values()):
        raise AssertionError(f"Every loss is zero: {metrics['loss']}")
    if not all(
        math.isfinite(metrics["reward_min"][step])
        and math.isfinite(metrics["reward_max"][step])
        and metrics["reward_max"][step] - metrics["reward_min"][step] > 1e-8
        for step in expected_steps
    ):
        raise AssertionError("Every step must contain a nonzero reward range")
    if not all(
        math.isfinite(metrics["advantage_min"][step])
        and math.isfinite(metrics["advantage_max"][step])
        and metrics["advantage_max"][step] - metrics["advantage_min"][step] > 1e-8
        for step in expected_steps
    ):
        raise AssertionError("Every step must contain a nonzero advantage range")
    if not all(
        math.isfinite(value) and 0 <= value <= 1
        for value in metrics["trajectory_age"].values()
    ):
        raise AssertionError(f"Invalid trajectory ages: {metrics['trajectory_age']}")
    if not all(
        math.isfinite(value) and 0 <= value < max_gen_kl
        for value in metrics["gen_kl_error"].values()
    ):
        raise AssertionError(
            f"Generation KL must remain in [0, {max_gen_kl}): {metrics['gen_kl_error']}"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--harness-head", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-gen-kl", type=float, default=0.002)
    args = parser.parse_args()

    scalars = _load_scalars(args.log_dir)
    metrics = validate_smoke_metrics(
        scalars,
        steps=args.steps,
        max_gen_kl=args.max_gen_kl,
    )
    expected_steps = set(range(1, args.steps + 1))

    run_text = args.run_log.read_text(encoding="utf-8", errors="replace")
    if (
        f"Step {args.steps}/{args.steps}" not in run_text
        and f"Step: {args.steps}" not in run_text
    ):
        raise AssertionError(f"Run did not reach step {args.steps}")
    refits = run_text.count("Performing policy generation refit")
    if refits < args.steps:
        raise AssertionError(f"Expected at least {args.steps} refits, got {refits}")
    if "Policy generation refit completed successfully" not in run_text:
        raise AssertionError("No successful policy generation refit marker")

    payload = {
        "schema": "nemo-rl.apertus70b-async-smoke.v1",
        "status": "terminal-metrics-validated",
        "source_head": args.source_head,
        "harness_head": args.harness_head,
        "image": str(args.image.resolve(strict=True)),
        "run_id": args.run_id,
        "validated_steps": sorted(expected_steps),
        "refit_count": refits,
        "max_gen_kl": args.max_gen_kl,
        "required_tags": REQUIRED_TAGS,
        "values_by_step": {
            name: {str(step): value for step, value in values.items()}
            for name, values in metrics.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise AssertionError(f"Refusing to replace evidence: {args.output}")
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"validated_steps={sorted(expected_steps)}")
    print(f"refit_count={refits}")
    print(f"evidence={args.output}")
    print("apertus70b_async_smoke_metrics=OK")


if __name__ == "__main__":
    main()
