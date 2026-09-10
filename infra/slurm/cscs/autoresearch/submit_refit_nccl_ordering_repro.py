# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit the bounded two-node NCCL refit ordering probe matrix on CSCS."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--account", default="infra01")
    parser.add_argument("--partition", default="normal")
    parser.add_argument(
        "--reservation", default=os.environ.get("REFIT_REPRO_RESERVATION", "")
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--transfers-per-stage", type=int, default=32)
    parser.add_argument("--tensor-mib", type=int, default=8)
    parser.add_argument("--iteration-timeout-s", type=float, default=10.0)
    parser.add_argument("--coordination-timeout-s", type=float, default=30.0)
    parser.add_argument("--arm-timeout-s", type=int, default=600)
    parser.add_argument("--time", default="00:45:00")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _checked_output(command: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def main() -> None:
    """Validate the checkout and submit one explicit-environment Slurm job."""
    args = _parse_args()
    repo = Path(__file__).resolve().parents[4]
    environment = args.environment.resolve()
    if not environment.is_file():
        raise FileNotFoundError(f"container EDF does not exist: {environment}")
    head = _checked_output(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"], cwd=repo
    )
    status = _checked_output(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--ignore-submodules=all",
        ],
        cwd=repo,
    )
    if status:
        raise RuntimeError(f"tracked source is dirty:\n{status}")

    user = os.environ.get("USER") or str(os.getuid())
    run_root = Path(
        os.environ.get(
            "REFIT_REPRO_RUN_ROOT",
            f"/iopsstor/scratch/cscs/{user}/nemo_rl_refit_ordering/{head}",
        )
    )
    log_root = Path(
        os.environ.get(
            "REFIT_REPRO_LOG_ROOT",
            str(repo / ".tmp" / "slurm-logs" / "refit-ordering" / head),
        )
    )
    exports = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "REFIT_REPRO_REPO_DIR": str(repo),
        "REFIT_REPRO_EXPECTED_HEAD": head,
        "REFIT_REPRO_CONTAINER_ENV": str(environment),
        "REFIT_REPRO_RUN_ROOT": str(run_root),
        "REFIT_REPRO_ITERATIONS": str(args.iterations),
        "REFIT_REPRO_TRANSFERS_PER_STAGE": str(args.transfers_per_stage),
        "REFIT_REPRO_TENSOR_MIB": str(args.tensor_mib),
        "REFIT_REPRO_ITERATION_TIMEOUT_S": str(args.iteration_timeout_s),
        "REFIT_REPRO_COORDINATION_TIMEOUT_S": str(args.coordination_timeout_s),
        "REFIT_REPRO_ARM_TIMEOUT_S": str(args.arm_timeout_s),
    }
    if any("," in key + value or "\n" in key + value for key, value in exports.items()):
        raise ValueError(
            "Slurm export names and values must not contain commas or newlines"
        )

    command = [
        "sbatch",
        "--parsable",
        f"--account={args.account}",
        f"--partition={args.partition}",
        "--nodes=2",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        "--cpus-per-task=64",
        "--mem=200000M",
        "--exclusive",
        f"--time={args.time}",
        "--job-name=refit-ordering",
        f"--chdir={repo}",
        f"--output={log_root}/slurm_%j.out",
        f"--error={log_root}/slurm_%j.err",
        "--export=" + ",".join(f"{key}={value}" for key, value in exports.items()),
        str(repo / "infra/slurm/cscs/autoresearch/run_refit_nccl_ordering_repro.sh"),
    ]
    if args.reservation:
        command.insert(4, f"--reservation={args.reservation}")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return

    log_root.mkdir(parents=True, exist_ok=True)
    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_SPANK_", "_SLURM_SPANK_", "SLURM_CPU_BIND"))
    }
    job = (
        subprocess.check_output(command, cwd=repo, env=clean_environment, text=True)
        .strip()
        .split(";")[0]
    )
    if not job.isdecimal():
        raise RuntimeError(f"sbatch returned an invalid job id: {job!r}")
    print(f"submitted_refit_ordering_job={job}", flush=True)


if __name__ == "__main__":
    main()
