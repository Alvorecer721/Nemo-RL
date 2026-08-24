# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
HELPER = REPO_ROOT / "tests/test_suites/completion_helpers.bash"


def _run_completion_check(
    tmp_path: Path,
    *,
    recorded_step: int | None,
    max_steps: int = 10,
    run_index: int | str = 1,
    num_runs: int | str = 1,
    command_exit_code: int = 0,
) -> subprocess.CompletedProcess[str]:
    metrics_path = tmp_path / "metrics.json"
    metrics = {}
    if recorded_step is not None:
        metrics["train/loss"] = {str(recorded_step): 0.0}
    metrics_path.write_text(json.dumps(metrics))

    command = f'source "{HELPER}"; exit_if_max_steps_reached; exit {command_exit_code}'
    env = {
        **os.environ,
        "JSON_METRICS": str(metrics_path),
        "MAX_STEPS": str(max_steps),
        "NRL_RUN_INDEX": str(run_index),
        "NRL_NUM_RUNS": str(num_runs),
    }
    return subprocess.run(
        ["bash", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_final_run_fails_closed_below_target(tmp_path: Path) -> None:
    result = _run_completion_check(tmp_path, recorded_step=7)

    assert result.returncode == 1
    assert "train/loss only reached step 7/10" in result.stderr


def test_intermediate_run_may_finish_below_target(tmp_path: Path) -> None:
    result = _run_completion_check(tmp_path, recorded_step=7, run_index=1, num_runs=2)

    assert result.returncode == 0


def test_final_run_passes_at_target(tmp_path: Path) -> None:
    result = _run_completion_check(tmp_path, recorded_step=10)

    assert result.returncode == 0
    assert "Target step 10 reached, skipping run" in result.stdout


def test_missing_metrics_fail_closed_on_final_run(tmp_path: Path) -> None:
    result = _run_completion_check(tmp_path, recorded_step=None)

    assert result.returncode == 1
    assert "train/loss only reached step 0/10" in result.stderr


def test_original_failure_is_preserved(tmp_path: Path) -> None:
    result = _run_completion_check(tmp_path, recorded_step=7, command_exit_code=42)

    assert result.returncode == 42
    assert "train/loss only reached" not in result.stderr


def test_invalid_run_identity_fails_closed(tmp_path: Path) -> None:
    result = _run_completion_check(
        tmp_path, recorded_step=10, run_index="invalid", num_runs=2
    )

    assert result.returncode == 1
    assert "must be positive integers" in result.stderr


def test_launch_exports_run_identity() -> None:
    launcher = (REPO_ROOT / "tools/launch").read_text()

    assert "NRL_RUN_INDEX=$i" in launcher
    assert "NRL_NUM_RUNS=$NUM_RUNS" in launcher
