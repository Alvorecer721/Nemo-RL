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

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_STEPS = 10
# R3-on job 3171492 stayed between 0.0003615 and 0.0004061 for all ten
# steps. A 0.001 ceiling retains more than 2x headroom while rejecting the
# historical R3-off 0.00230-0.00271 regime.
GLM_KL_SAFETY_CEILING = 0.001
R3_TOKEN_MULT_MEDIAN_CEILING = 1.02
# R3-on job 3171492 had 4/1,291,712 valid tokens above 0.5 and none above
# 1.0. The historical R3-off run had 7,873 above 0.5 and 570 above 1.0.
R3_ABS_LOGPROB_GT_0_5_FRACTION_CEILING = 1.0e-4
# The 1024-token response envelope truncated 94.5-100% of each batch. The
# stronger rung is only representative if its larger envelope materially
# changes that regime rather than merely accumulating more truncated tokens.
TRUNCATION_RATE_MEAN_CEILING = 0.9
HISTORICAL_R3_OFF_KL = (
    0.0026317706797271967,
    0.0025273626670241356,
    0.0025749346241354942,
    0.0027088902425020933,
    0.002529832301661372,
    0.0024474062956869602,
    0.0025619089137762785,
    0.0023466208949685097,
    0.0022966070100665092,
    0.0024027915205806494,
)


def _series(metrics: dict[str, Any], name: str) -> list[float]:
    values = metrics.get(name)
    if not isinstance(values, dict):
        raise ValueError(f"Missing metric series: {name}")
    steps = sorted((int(step), float(value)) for step, value in values.items())
    expected = list(range(1, EXPECTED_STEPS + 1))
    if [step for step, _ in steps] != expected:
        raise ValueError(
            f"{name} must contain exactly steps {expected}; got {[step for step, _ in steps]}"
        )
    series = [value for _, value in steps]
    if not all(math.isfinite(value) for value in series):
        raise ValueError(f"{name} contains non-finite values: {series}")
    return series


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile from an empty series")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def summarize_logprob_tails(train_data_dir: Path) -> dict[str, Any]:
    per_step: dict[str, dict[str, float | int]] = {}
    total_tokens = 0
    total_gt_0_5 = 0
    total_gt_1_0 = 0

    for step in range(1, EXPECTED_STEPS + 1):
        matches = sorted(train_data_dir.glob(f"*/train_data_step{step}.jsonl"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one train_data_step{step}.jsonl below {train_data_dir}; "
                f"found {matches}"
            )

        errors: list[float] = []
        with matches[0].open() as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                fields = (
                    record["generation_logprobs"],
                    record["prev_logprobs"],
                    record["token_loss_mask"],
                    record["sample_loss_mask"],
                )
                lengths = [len(field) for field in fields]
                if len(set(lengths)) != 1:
                    raise ValueError(
                        f"Outer batch fields disagree in {matches[0]}:{line_number}: "
                        f"{lengths}"
                    )
                for generation, previous, token_mask, sample_mask in zip(
                    *fields, strict=True
                ):
                    if not sample_mask:
                        continue
                    inner_lengths = (len(generation), len(previous), len(token_mask))
                    if len(set(inner_lengths)) != 1:
                        raise ValueError(
                            f"Token fields disagree in {matches[0]}:{line_number}: "
                            f"{inner_lengths}"
                        )
                    errors.extend(
                        abs(float(generation_lp) - float(previous_lp))
                        for generation_lp, previous_lp, valid in zip(
                            generation, previous, token_mask, strict=True
                        )
                        if valid
                    )

        if not errors or not all(math.isfinite(value) for value in errors):
            raise ValueError(f"Step {step} has no finite valid-token logprob errors")
        errors.sort()
        gt_0_5 = sum(value > 0.5 for value in errors)
        gt_1_0 = sum(value > 1.0 for value in errors)
        per_step[str(step)] = {
            "tokens": len(errors),
            "mean_abs": statistics.fmean(errors),
            "p95_abs": _percentile(errors, 0.95),
            "p99_abs": _percentile(errors, 0.99),
            "max_abs": errors[-1],
            "count_gt_0_5": gt_0_5,
            "count_gt_1_0": gt_1_0,
        }
        total_tokens += len(errors)
        total_gt_0_5 += gt_0_5
        total_gt_1_0 += gt_1_0

    return {
        "per_step": per_step,
        "total_tokens": total_tokens,
        "count_gt_0_5": total_gt_0_5,
        "count_gt_1_0": total_gt_1_0,
    }


def validate_logprob_tails(summary: dict[str, Any]) -> dict[str, Any]:
    total_tokens = int(summary["total_tokens"])
    count_gt_0_5 = int(summary["count_gt_0_5"])
    count_gt_1_0 = int(summary["count_gt_1_0"])
    if total_tokens <= 0:
        raise ValueError("Per-token logprob evidence contains no valid tokens")
    fraction_gt_0_5 = count_gt_0_5 / total_tokens
    if count_gt_1_0:
        raise ValueError(
            "Router Replay left valid tokens with abs(delta log p) > 1.0: "
            f"count={count_gt_1_0}/{total_tokens}"
        )
    if fraction_gt_0_5 >= R3_ABS_LOGPROB_GT_0_5_FRACTION_CEILING:
        raise ValueError(
            "Router Replay per-token tail exceeded its safety envelope: "
            f"count_gt_0_5={count_gt_0_5}/{total_tokens} "
            f"({fraction_gt_0_5:.6g}), required < "
            f"{R3_ABS_LOGPROB_GT_0_5_FRACTION_CEILING}"
        )
    summary["fraction_gt_0_5"] = fraction_gt_0_5
    return summary


def validate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    kl = _series(metrics, "train/gen_kl_error")
    token_mult = _series(metrics, "train/token_mult_prob_error")
    js = _series(metrics, "train/js_divergence_error")
    loss = _series(metrics, "train/loss")
    reward = _series(metrics, "train/reward")
    advantage_min = _series(metrics, "train/advantages/min")
    advantage_max = _series(metrics, "train/advantages/max")
    truncation_rate = _series(metrics, "train/truncation_rate")

    if min(kl) < 0 or max(kl) >= GLM_KL_SAFETY_CEILING:
        raise ValueError(
            "Generation KL escaped the GLM safety envelope: "
            f"min={min(kl):.6g}, max={max(kl):.6g}"
        )
    if min(token_mult) < 1.0:
        raise ValueError(
            "token_mult_prob_error is exp(abs(delta log p)) and cannot be below 1; "
            f"observed {min(token_mult):.6g}"
        )
    token_mult_median = statistics.median(token_mult)
    if token_mult_median >= R3_TOKEN_MULT_MEDIAN_CEILING:
        raise ValueError(
            "Router Replay did not meet the established probability-parity gate: "
            f"median token_mult_prob_error={token_mult_median:.6g}, "
            f"required < {R3_TOKEN_MULT_MEDIAN_CEILING}"
        )
    if min(js) < 0:
        raise ValueError(f"JS divergence cannot be negative: min={min(js):.6g}")
    truncation_mean = statistics.fmean(truncation_rate)
    if min(truncation_rate) < 0 or max(truncation_rate) > 1:
        raise ValueError(
            "Truncation rate must stay in [0, 1]: "
            f"min={min(truncation_rate):.6g}, max={max(truncation_rate):.6g}"
        )
    if truncation_mean >= TRUNCATION_RATE_MEAN_CEILING:
        raise ValueError(
            "Response envelope still truncates too many trajectories for a "
            "representative learning run: "
            f"mean={truncation_mean:.6g}, required < "
            f"{TRUNCATION_RATE_MEAN_CEILING}"
        )

    signal_steps = sum(
        low < 0 < high for low, high in zip(advantage_min, advantage_max, strict=True)
    )
    nonzero_loss_steps = sum(abs(value) > 1.0e-12 for value in loss)
    if signal_steps < 8 or nonzero_loss_steps < 8:
        raise ValueError(
            "Ten-step run did not provide representative learning signal: "
            f"advantage_steps={signal_steps}, nonzero_loss_steps={nonzero_loss_steps}"
        )
    if max(reward) <= 0:
        raise ValueError("No positive reward was observed in any training step")

    baseline_mean = statistics.fmean(HISTORICAL_R3_OFF_KL)
    kl_mean = statistics.fmean(kl)
    return {
        "steps": EXPECTED_STEPS,
        "kl": {
            "min": min(kl),
            "max": max(kl),
            "mean": kl_mean,
            "median": statistics.median(kl),
            "final": kl[-1],
            "below_0_002_steps": sum(value < 0.002 for value in kl),
            "historical_r3_off_mean": baseline_mean,
            "mean_ratio_to_historical_r3_off": kl_mean / baseline_mean,
        },
        "token_mult_prob_error": {
            "min": min(token_mult),
            "max": max(token_mult),
            "median": token_mult_median,
        },
        "js_divergence_error": {
            "min": min(js),
            "max": max(js),
            "mean": statistics.fmean(js),
        },
        "learning_signal_steps": signal_steps,
        "nonzero_loss_steps": nonzero_loss_steps,
        "positive_reward_steps": sum(value > 0 for value in reward),
        "truncation_rate": {
            "min": min(truncation_rate),
            "max": max(truncation_rate),
            "mean": truncation_mean,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the GLM-5.1 R3 ten-step run")
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--train-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    summary = validate_metrics(metrics)
    summary["per_token_logprob_tails"] = validate_logprob_tails(
        summarize_logprob_tails(args.train_data_dir)
    )
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
