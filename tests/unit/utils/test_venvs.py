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
succeeds. The marker carries the dependency fingerprint it was built from, so a
venv that predates a uv.lock/pyproject.toml change is rebuilt instead of served
stale. Exactly one process builds (O_EXCL claim on STARTED_ENV_BUILDER);
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
from nemo_rl.utils.venvs import (
    VENV_READY_MARKER,
    create_local_venv,
    make_actor_runtime_env,
    pin_uv_to_path,
)

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
        # cmd[0] is whatever uv the UV pin resolved to, so key off the subcommand.
        if cmd[1:2] == ["venv"]:
            bin_dir = Path(cmd[-1]) / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "python").touch()
        return None

    return run


def _mark_ready(venv: Path, py_executable: str = "uv run --locked") -> None:
    """Mark a venv ready the way a completed build does."""
    (venv / VENV_READY_MARKER).write_text(
        venvs_module._dependency_fingerprint(py_executable)
    )


def test_create_local_venv_marks_ready_only_after_success(tmp_path):
    with patch.object(venvs_module.subprocess, "run", _fake_uv(tmp_path)):
        python_path = create_local_venv("uv run --locked", "demo.Worker")
    venv = tmp_path / "demo.Worker"
    assert python_path == str(venv / "bin" / "python")
    assert (venv / VENV_READY_MARKER).exists()


def test_base_sync_retains_actor_extra_until_exact_worker_sync(tmp_path):
    calls = []

    def record_run(cmd, **kwargs):
        calls.append(cmd)
        return _fake_uv(tmp_path)(cmd, **kwargs)

    with patch.object(venvs_module.subprocess, "run", record_run):
        create_local_venv("uv run --locked --extra vllm", "demo.Worker")

    assert calls[1][1:3] == ["sync", "--inexact"]
    assert "--inexact" not in calls[2]
    assert calls[2][1:6] == ["run", "--exact", "--locked", "--extra", "vllm"]


def test_non_uv_worker_keeps_exact_base_sync(tmp_path):
    calls = []

    def record_run(cmd, **kwargs):
        calls.append(cmd)
        return _fake_uv(tmp_path)(cmd, **kwargs)

    with patch.object(venvs_module.subprocess, "run", record_run):
        create_local_venv("python -V", "demo.Worker")

    assert calls[1][1] == "sync"
    assert "--inexact" not in calls[1]
    assert calls[2] == [
        "python",
        "-V",
        "echo",
        f"Finished creating venv {tmp_path}/demo.Worker",
    ]


def test_actor_runtime_env_prepends_pinned_uv_to_path(monkeypatch, tmp_path):
    uv_executable = tmp_path / "uv"
    uv_executable.touch(mode=0o755)
    monkeypatch.setenv("UV", str(uv_executable))
    monkeypatch.setenv("PATH", "/users/example/.local/bin:/usr/bin")

    with patch(
        "nemo_rl.distributed.ray_actor_environment_registry.get_actor_python_env",
        return_value="/opt/actor-venv/bin/python",
    ):
        runtime_env = make_actor_runtime_env("demo.Worker")

    assert runtime_env["py_executable"] == "/opt/actor-venv/bin/python"
    assert runtime_env["env_vars"]["UV"] == str(uv_executable)
    assert runtime_env["env_vars"]["PATH"].split(os.pathsep) == [
        str(tmp_path),
        "/users/example/.local/bin",
        "/usr/bin",
    ]
    assert runtime_env["env_vars"]["VIRTUAL_ENV"] == "/opt/actor-venv"
    assert runtime_env["env_vars"]["UV_PROJECT_ENVIRONMENT"] == "/opt/actor-venv"

    # Ray prepends the actor venv after applying runtime_env. Gym calls the
    # helper again inside its actor before spawning component subprocesses.
    monkeypatch.setenv("PATH", "/opt/actor-venv/bin:/users/example/.local/bin")
    pin_uv_to_path()
    assert os.environ["PATH"].split(os.pathsep) == [
        str(tmp_path),
        "/opt/actor-venv/bin",
        "/users/example/.local/bin",
    ]


def test_pin_uv_to_path_rejects_missing_explicit_uv(tmp_path):
    env_vars = {
        "UV": str(tmp_path / "missing-uv"),
        "PATH": "/usr/bin",
    }

    with pytest.raises(FileNotFoundError, match="UV executable"):
        pin_uv_to_path(env_vars)


def test_create_local_venv_failure_leaves_no_marker(tmp_path):
    calls = {"n": 0}

    def failing_run(cmd, **kwargs):
        if cmd[1:2] == ["venv"]:
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
        if cmd[1:2] == ["venv"]:
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
    _mark_ready(venv)

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
        _mark_ready(venv)
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


@pytest.fixture
def project_dependencies(tmp_path, monkeypatch):
    """Point the fingerprint at a stand-in project whose lockfile tests can edit."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(venvs_module, "git_root", str(root))
    venvs_module._dependency_fingerprint.cache_clear()
    yield root
    venvs_module._dependency_fingerprint.cache_clear()


def test_marker_records_the_dependency_fingerprint(tmp_path, project_dependencies):
    with patch.object(venvs_module.subprocess, "run", _fake_uv(tmp_path)):
        create_local_venv("uv run --locked", "demo.Worker")

    marker = tmp_path / "demo.Worker" / VENV_READY_MARKER
    assert marker.read_text() == venvs_module._dependency_fingerprint("uv run --locked")


def test_env_builder_rebuilds_when_dependencies_change(tmp_path, project_dependencies):
    venv = tmp_path / "demo.Worker"
    with patch.object(venvs_module.subprocess, "run", _fake_uv(tmp_path)):
        _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)
        built_from = (venv / VENV_READY_MARKER).read_text()

        (project_dependencies / "uv.lock").write_text("version = 2\n")
        venvs_module._dependency_fingerprint.cache_clear()
        # A later job is a fresh process, so it does not inherit the build cache.
        create_local_venv.cache_clear()

        _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)

    rebuilt_from = (venv / VENV_READY_MARKER).read_text()
    assert rebuilt_from != built_from
    assert rebuilt_from == venvs_module._dependency_fingerprint("uv run --locked")


def test_env_builder_rebuilds_when_worker_command_changes(
    tmp_path, project_dependencies
):
    venv = tmp_path / "demo.Worker"
    old_command = "uv run --locked --extra mcore"
    new_command = "uv run --locked --extra vllm"

    with patch.object(venvs_module.subprocess, "run", _fake_uv(tmp_path)):
        _env_builder_fn(old_command, "demo.Worker", node_idx=0)
        built_from = (venv / VENV_READY_MARKER).read_text()

        _env_builder_fn(new_command, "demo.Worker", node_idx=0)

    rebuilt_from = (venv / VENV_READY_MARKER).read_text()
    assert rebuilt_from != built_from
    assert rebuilt_from == venvs_module._dependency_fingerprint(new_command)


def test_dependency_fingerprint_normalizes_worker_command(project_dependencies):
    assert venvs_module._dependency_fingerprint(
        "uv  run --locked  --extra vllm"
    ) == venvs_module._dependency_fingerprint("uv run --locked --extra vllm")


def test_dependency_fingerprint_resolves_checkout_aliases(
    tmp_path, project_dependencies
):
    alias = tmp_path / "project-alias"
    alias.symlink_to(project_dependencies, target_is_directory=True)
    real_command = f"uv run --locked --directory={project_dependencies}"
    alias_command = f"uv run --locked --directory={alias}"

    assert venvs_module._dependency_fingerprint(
        real_command
    ) == venvs_module._dependency_fingerprint(alias_command)


def test_waiter_rejects_a_venv_built_from_other_dependencies(
    tmp_path, project_dependencies, monkeypatch
):
    monkeypatch.setenv("NRL_VENV_BUILD_TIMEOUT_SECS", "30")
    venv = tmp_path / "demo.Worker"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()
    claim = venv / "STARTED_ENV_BUILDER"
    claim.touch()

    def finish_build_against_another_lock():
        (venv / VENV_READY_MARKER).write_text("fingerprint-of-a-different-lock")
        claim.unlink()

    threading.Timer(1.5, finish_build_against_another_lock).start()
    with pytest.raises(RuntimeError, match="different dependencies"):
        _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)
