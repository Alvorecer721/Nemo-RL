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

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[3]
LAUNCHER = PROJECT_ROOT / "infra/slurm/cscs/submit_nemo_rl_dpo.slurm"


@pytest.fixture
def fake_slurm_bin(tmp_path: Path) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    srun_log = tmp_path / "srun.log"
    srun = bin_dir / "srun"
    srun.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$FAKE_SRUN_LOG"\n'
        'exit "${FAKE_SRUN_EXIT:?}"\n'
    )
    srun.chmod(0o755)

    sbatch_log = tmp_path / "sbatch.log"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$FAKE_SBATCH_LOG"\n')
    sbatch.chmod(0o755)

    return bin_dir, srun_log, sbatch_log


def _run_launcher(
    fake_slurm_bin: tuple[Path, Path, Path],
    srun_exit_code: int,
    *,
    auto_requeue: bool = True,
) -> tuple[subprocess.CompletedProcess[str], str, list[str]]:
    bin_dir, srun_log, sbatch_log = fake_slurm_bin
    env = os.environ.copy()
    env.update(
        {
            "AUTO_REQUEUE": str(auto_requeue).lower(),
            "FAKE_SBATCH_LOG": str(sbatch_log),
            "FAKE_SRUN_LOG": str(srun_log),
            "FAKE_SRUN_EXIT": str(srun_exit_code),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SLURM_CPUS_PER_TASK": "1",
            "SLURM_JOB_NAME": "test-dpo",
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    submissions = sbatch_log.read_text().splitlines() if sbatch_log.exists() else []
    srun_invocation = srun_log.read_text()
    return result, srun_invocation, submissions


def test_completed_training_does_not_resubmit(
    fake_slurm_bin: tuple[Path, Path, Path],
) -> None:
    result, srun_invocation, submissions = _run_launcher(fake_slurm_bin, 0)

    assert result.returncode == 0
    assert 'checkpointing.checkpoint_must_save_by="00:11:30:00"' in srun_invocation
    assert submissions == []


def test_timeout_resubmits_once(fake_slurm_bin: tuple[Path, Path, Path]) -> None:
    result, _, submissions = _run_launcher(fake_slurm_bin, 75)

    assert result.returncode == 0
    assert len(submissions) == 1
    assert submissions[0].startswith("--dependency=singleton --job-name=test-dpo ")


def test_timeout_does_not_resubmit_when_disabled(
    fake_slurm_bin: tuple[Path, Path, Path],
) -> None:
    result, _, submissions = _run_launcher(fake_slurm_bin, 75, auto_requeue=False)

    assert result.returncode == 75
    assert submissions == []


def test_training_failure_is_preserved_without_resubmission(
    fake_slurm_bin: tuple[Path, Path, Path],
) -> None:
    result, _, submissions = _run_launcher(fake_slurm_bin, 42)

    assert result.returncode == 42
    assert submissions == []
