"""Summarize an Apertus 70B topology run after its final training step."""

import argparse
import json
import re
import statistics
from pathlib import Path


def percentile(values: list[float], percentile_value: int) -> float:
    assert values
    return statistics.quantiles(values, n=100, method="inclusive")[percentile_value - 1]


def distribution(values: list[float], prefix: str) -> dict[str, float]:
    assert values
    return {
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_p50": statistics.median(values),
        f"{prefix}_p95": percentile(values, 95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--preference-pairs", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()

    text = args.run_log.read_text(encoding="utf-8", errors="replace")
    metrics = json.loads(args.metrics.read_text())

    def ordered_values(key: str) -> list[float]:
        series = metrics.get(key, {})
        return [
            float(series[str(step)])
            for step in range(1, args.steps + 1)
            if str(step) in series
        ]

    policy_times = ordered_values("timing/train/policy_training")
    total_times = ordered_values("timing/train/total_step_time")
    valid_tps_gpu = ordered_values("timing/train/valid_tokens_per_sec_per_gpu")
    losses = ordered_values("train/loss")
    aggregate_flops = [
        float(value) for value in re.findall(r"Training FLOPS: ([0-9.]+) TFLOPS", text)
    ]
    native_mfu = [
        float(value)
        for value in re.findall(
            r"Training Model Floating Point Utilization: ([0-9.]+)%", text
        )
    ]

    assert len(policy_times) == args.steps, policy_times
    assert len(total_times) == args.steps, total_times
    assert len(valid_tps_gpu) == args.steps, valid_tps_gpu
    assert len(losses) == args.steps, losses
    assert len(aggregate_flops) == args.steps, aggregate_flops

    steady_start = 2
    steady_policy_times = policy_times[steady_start:]
    steady_total_times = total_times[steady_start:]
    steady_valid_tps_gpu = valid_tps_gpu[steady_start:]
    steady_aggregate_flops = aggregate_flops[steady_start:]
    steady_tflops_per_gpu = [
        value / args.world_size for value in steady_aggregate_flops
    ]
    peak_bf16_tflops_per_gpu = 989.5
    external_mfu = [
        100.0 * value / peak_bf16_tflops_per_gpu for value in steady_tflops_per_gpu
    ]

    gpu_mem_keys = sorted(
        key for key in metrics if key.endswith(".mem_gb") and ".gpu." in key
    )
    gpu_util_keys = sorted(
        key for key in metrics if key.endswith(".util") and ".gpu." in key
    )
    per_gpu_peak_hbm = {
        key: max(float(value) for value in metrics[key].values())
        for key in gpu_mem_keys
        if metrics[key]
    }
    steady_gpu_util = [
        float(value)
        for key in gpu_util_keys
        for step, value in metrics[key].items()
        if str(step).isdigit() and int(step) >= 3
    ]

    summary: dict[str, object] = {
        "job_id": args.job_id,
        "topology": args.topology,
        "tp": args.tp,
        "pp": args.pp,
        "dp": 1,
        "cp": 1,
        "world_size": args.world_size,
        "steps": args.steps,
        "steady_state_steps": f"3-{args.steps}",
        "preference_pairs_per_step": args.preference_pairs,
        "effective_sequences_per_step": args.preference_pairs * 2,
        "max_sequence_length": args.sequence_length,
        "pipeline_1f1b_efficiency_estimate": args.preference_pairs
        / (args.preference_pairs + args.pp - 1),
        "loss_final_step": losses[-1],
        "external_mfu_basis_tflops_per_gpu": peak_bf16_tflops_per_gpu,
        "per_gpu_peak_hbm_gb": per_gpu_peak_hbm,
    }
    summary.update(distribution(steady_policy_times, "policy_training_seconds_steady"))
    summary.update(distribution(steady_total_times, "total_step_seconds_steady"))
    summary.update(
        distribution(
            steady_valid_tps_gpu,
            "valid_tokens_per_second_per_gpu_steady",
        )
    )
    summary.update(
        distribution(steady_tflops_per_gpu, "training_tflops_per_gpu_steady")
    )
    summary.update(distribution(external_mfu, "external_dense_bf16_mfu_percent_steady"))
    if len(native_mfu) == args.steps:
        summary.update(
            distribution(native_mfu[steady_start:], "native_mfu_percent_steady")
        )
    else:
        summary["native_mfu_percent_steady"] = None
    if per_gpu_peak_hbm:
        summary.update(
            distribution(list(per_gpu_peak_hbm.values()), "peak_hbm_gb_per_gpu")
        )
        summary["peak_hbm_gb_any_gpu"] = max(per_gpu_peak_hbm.values())
    if steady_gpu_util:
        summary.update(distribution(steady_gpu_util, "gpu_utilization_percent_steady"))

    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
