# Copyright (c) 2026, the Apertus project.
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
"""Venv provisioning contract: readiness is the marker, claims are exclusive and expire.

bin/python existing is NOT readiness (uv creates it before packages land); a
venv is usable only once NEMO_RL_VENV_READY exists, written after `uv sync`
succeeds. Exactly one process builds (O_EXCL claim on STARTED_ENV_BUILDER);
waiters block on the marker with a timeout, and a claim older than the timeout
is expired as the residue of a killed build.
"""

import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import nemo_rl.utils.venvs as venvs_module
from nemo_rl.utils.venvs import VENV_READY_MARKER, create_local_venv

# The protocol under test lives in the task body, not in Ray scheduling.
_env_builder_fn = venvs_module._env_builder._function


@pytest.fixture(autouse=True)
def _isolated_venv_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_RL_VENV_DIR", str(tmp_path))
    create_local_venv.cache_clear()
    yield tmp_path
    create_local_venv.cache_clear()


def _fake_uv(venv_dir):
    """subprocess.run stand-in: `uv venv` materializes bin/python, the rest no-op."""

    def run(cmd, **kwargs):
        if cmd[:2] == ["uv", "venv"]:
            bin_dir = Path(cmd[-1]) / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "python").touch()
        return None

    return run


def test_create_local_venv_marks_ready_only_after_success(tmp_path):
    with patch.object(venvs_module.subprocess, "run", _fake_uv(tmp_path)):
        python_path = create_local_venv("uv run --locked", "demo.Worker")
    venv = tmp_path / "demo.Worker"
    assert python_path == str(venv / "bin" / "python")
    assert (venv / VENV_READY_MARKER).exists()


def test_create_local_venv_failure_leaves_no_marker(tmp_path):
    calls = {"n": 0}

    def failing_run(cmd, **kwargs):
        if cmd[:2] == ["uv", "venv"]:
            return _fake_uv(tmp_path)(cmd, **kwargs)
        raise RuntimeError("sync exploded")

    with (
        patch.object(venvs_module.subprocess, "run", failing_run),
        pytest.raises(RuntimeError),
    ):
        create_local_venv("uv run --locked", "demo.Worker")
    venv = tmp_path / "demo.Worker"
    assert (venv / "bin" / "python").exists()
    assert not (venv / VENV_READY_MARKER).exists()


def test_stale_marker_removed_at_build_start(tmp_path):
    venv = tmp_path / "demo.Worker"
    venv.mkdir(parents=True)
    (venv / VENV_READY_MARKER).touch()

    def killed_mid_sync(cmd, **kwargs):
        if cmd[:2] == ["uv", "venv"]:
            return _fake_uv(tmp_path)(cmd, **kwargs)
        raise KeyboardInterrupt

    with (
        patch.object(venvs_module.subprocess, "run", killed_mid_sync),
        pytest.raises(KeyboardInterrupt),
    ):
        create_local_venv("uv run --locked", "demo.Worker")
    assert not (venv / VENV_READY_MARKER).exists()


def test_env_builder_rebuilds_unmarked_venv(tmp_path):
    venv = tmp_path / "demo.Worker"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()  # partial: looks usable, is not

    with patch.object(venvs_module, "create_local_venv") as build:
        build.return_value = str(venv / "bin" / "python")
        result = _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)
    build.assert_called_once()
    assert result == str(venv / "bin" / "python")
    assert not (venv / "STARTED_ENV_BUILDER").exists()


def test_env_builder_early_returns_marked_venv(tmp_path):
    venv = tmp_path / "demo.Worker"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()
    (venv / VENV_READY_MARKER).touch()

    with patch.object(venvs_module, "create_local_venv") as build:
        result = _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)
    build.assert_not_called()
    assert result == str(venv / "bin" / "python")


def test_waiter_times_out_on_held_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("NRL_VENV_BUILD_TIMEOUT_SECS", "2")
    venv = tmp_path / "demo.Worker"
    venv.mkdir(parents=True)
    (venv / "STARTED_ENV_BUILDER").touch()

    with pytest.raises(TimeoutError, match="rm -rf"):
        _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)


def test_waiter_raises_when_builder_dies_without_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("NRL_VENV_BUILD_TIMEOUT_SECS", "30")
    venv = tmp_path / "demo.Worker"
    venv.mkdir(parents=True)
    claim = venv / "STARTED_ENV_BUILDER"
    claim.touch()

    threading.Timer(1.5, claim.unlink).start()
    with pytest.raises(RuntimeError, match="without completing"):
        _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)


def test_waiter_returns_when_marker_appears(tmp_path, monkeypatch):
    monkeypatch.setenv("NRL_VENV_BUILD_TIMEOUT_SECS", "30")
    venv = tmp_path / "demo.Worker"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()
    claim = venv / "STARTED_ENV_BUILDER"
    claim.touch()

    def finish_build():
        (venv / VENV_READY_MARKER).touch()
        claim.unlink()

    threading.Timer(1.5, finish_build).start()
    result = _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)
    assert result == str(venv / "bin" / "python")


def test_stale_claim_expired_and_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setenv("NRL_VENV_BUILD_TIMEOUT_SECS", "60")
    venv = tmp_path / "demo.Worker"
    venv.mkdir(parents=True)
    claim = venv / "STARTED_ENV_BUILDER"
    claim.touch()
    stale = time.time() - 3600
    os.utime(claim, (stale, stale))

    with patch.object(venvs_module, "create_local_venv") as build:
        build.return_value = str(venv / "bin" / "python")
        result = _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)
    build.assert_called_once()
    assert result == str(venv / "bin" / "python")
    assert not claim.exists()
