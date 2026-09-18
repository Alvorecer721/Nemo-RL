# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise launcher cleanup with real child processes, without a Ray cluster."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def run_shutdown(tmp_path, child, grace=1, step="12345.6"):
    source = (ROOT / "ray.sub").read_text()
    start = source.index("wait_for_ray_cluster() {")
    function = source[start : source.index("\n}\n", start) + 3]
    cancel = tmp_path / "scancel"
    cancel.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CANCEL_LOG"\n')
    cancel.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        CANCEL_LOG=str(tmp_path / "cancel.log"),
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            function
            + f'\n{child} &\npid=$!\nwait_for_ray_cluster "$pid" "{step}" {grace}\n',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result, tmp_path / "cancel.log"


def test_completed_child_preserves_status(tmp_path):
    result, cancel = run_shutdown(tmp_path, "(exit 7)")
    assert result.returncode == 7
    assert not cancel.exists()


def test_stuck_child_is_bounded_and_only_its_step_is_cancelled(tmp_path):
    result, cancel = run_shutdown(tmp_path, "sleep 600")
    assert result.returncode == 124
    assert cancel.read_text().strip() == "--signal=KILL 12345.6"
    assert "cleanup deadline" in result.stderr


def test_success_does_not_cancel_step(tmp_path):
    result, cancel = run_shutdown(tmp_path, "(exit 0)")
    assert result.returncode == 0
    assert not cancel.exists()


def test_invalid_identity_never_cancels_entire_job(tmp_path):
    result, cancel = run_shutdown(tmp_path, "sleep 600", step="12345")
    assert result.returncode == 124
    assert not cancel.exists()
    assert "Invalid Ray step identity" in result.stderr


def test_child_ignoring_term_is_killed(tmp_path):
    result, cancel = run_shutdown(tmp_path, "bash -c 'trap \"\" TERM; exec sleep 600'")
    assert result.returncode == 124
    assert cancel.exists()
