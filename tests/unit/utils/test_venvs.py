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
venv whose resolved environment no longer matches the lock is rebuilt instead of
served stale. Exactly one process builds (O_EXCL claim on STARTED_ENV_BUILDER);
waiters block on the marker with a timeout, and a claim older than the timeout
is expired as the residue of a killed build.
"""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import nemo_rl.utils.venvs as venvs_module
from nemo_rl.utils.venvs import (
    VENV_READY_MARKER,
    add_checkout_to_pythonpath,
    add_hf_modules_cache_to_pythonpath,
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


@pytest.fixture(autouse=True)
def resolved_requirements(monkeypatch):
    """Stand in for `uv export`; a test edits "text" to bump a dependency."""
    exported = {"text": "demo==1.0\n", "uv": "uv 1.0.0"}
    monkeypatch.setattr(
        venvs_module,
        "_export_requirements",
        lambda extras: exported["text"] + "".join(f"{e}==1.0\n" for e in extras),
    )
    monkeypatch.setattr(venvs_module, "_uv_version", lambda: exported["uv"])
    venvs_module._resolved_environment.cache_clear()
    yield exported
    venvs_module._resolved_environment.cache_clear()


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
    venvs_module._mark_venv_ready(venv / VENV_READY_MARKER, py_executable)


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


def test_non_uv_worker_command_runs_verbatim(tmp_path):
    calls = []

    def record_run(cmd, **kwargs):
        calls.append(cmd)
        return _fake_uv(tmp_path)(cmd, **kwargs)

    with patch.object(venvs_module.subprocess, "run", record_run):
        create_local_venv("python -V", "demo.Worker")

    assert calls[1][1:3] == ["sync", "--inexact"]
    assert calls[2] == [
        "python",
        "-V",
        "echo",
        f"Finished creating venv {tmp_path}/demo.Worker",
    ]


@pytest.mark.parametrize("failure", [None, "sync", "inventory"])
def test_prebuilt_finalization_selects_actor_and_gates_readiness(tmp_path, failure):
    """Finalize the actor directly and mark ready only after recording its state."""
    actor = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
    command = "uv run --locked --extra vllm"
    worker = tmp_path / actor
    (worker / "bin").mkdir(parents=True)
    (worker / "bin/python").touch()
    dependency = worker / "dependency"
    dependency.write_text("prepared backend version")
    marker = worker / VENV_READY_MARKER
    marker.write_text("stale")

    def execute(cmd, **kwargs):
        if cmd[1] == "sync":
            assert "--frozen" in cmd and "--offline" in cmd
            assert "--extra" in cmd and "vllm" in cmd
            assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(worker)
            if failure == "sync":
                raise RuntimeError("dependency mismatch")
        elif cmd[1].endswith("venv_inventory.py"):
            assert cmd[0] == str(worker / "bin/python") and cmd[2] == "record"
            if failure == "inventory":
                raise RuntimeError("inventory write failed")
        else:
            raise AssertionError(f"Unexpected mutating command: {cmd}")
        assert kwargs["env"]["UV_OFFLINE"] == "1"
        assert kwargs["env"]["UV_LINK_MODE"] == "copy"

    with patch.object(venvs_module.subprocess, "run", execute):
        if failure:
            with pytest.raises(RuntimeError):
                venvs_module.finalize_prebuilt_venv(command, actor)
            assert not marker.exists()
        else:
            result = venvs_module.finalize_prebuilt_venv(command, actor)
            assert result == str(worker / "bin/python")
            assert venvs_module.venv_is_current(marker, command)
    assert dependency.read_text() == "prepared backend version"


def test_prebuilt_finalization_rejects_missing_worker(tmp_path):
    actor = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
    with pytest.raises(FileNotFoundError):
        venvs_module.finalize_prebuilt_venv("uv run --locked --extra vllm", actor)
    assert not (tmp_path / actor / VENV_READY_MARKER).exists()


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
    """Point the fingerprint at a stand-in project whose build settings tests can edit."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text('[tool.uv]\nlink-mode = "copy"\n')
    monkeypatch.setattr(venvs_module, "git_root", str(root))
    yield root


def test_marker_records_the_dependency_fingerprint(tmp_path, project_dependencies):
    with patch.object(venvs_module.subprocess, "run", _fake_uv(tmp_path)):
        create_local_venv("uv run --locked", "demo.Worker")

    marker = tmp_path / "demo.Worker" / VENV_READY_MARKER
    assert json.loads(marker.read_text()) == venvs_module._dependency_fingerprint(
        "uv run --locked"
    )


def test_env_builder_rebuilds_when_dependencies_change(
    tmp_path, project_dependencies, resolved_requirements
):
    venv = tmp_path / "demo.Worker"
    with patch.object(venvs_module.subprocess, "run", _fake_uv(tmp_path)):
        _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)
        built_from = (venv / VENV_READY_MARKER).read_text()

        resolved_requirements["text"] = "demo==2.0\n"
        venvs_module._resolved_environment.cache_clear()
        # A later job is a fresh process, so it does not inherit the build cache.
        create_local_venv.cache_clear()

        _env_builder_fn("uv run --locked", "demo.Worker", node_idx=0)

    rebuilt_from = (venv / VENV_READY_MARKER).read_text()
    assert rebuilt_from != built_from
    assert venvs_module.venv_is_current(venv / VENV_READY_MARKER, "uv run --locked")


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
    assert venvs_module.venv_is_current(venv / VENV_READY_MARKER, new_command)


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


def test_edits_that_install_nothing_new_keep_the_venv_current(
    tmp_path, project_dependencies
):
    venv = tmp_path / "demo.Worker"
    venv.mkdir()
    _mark_ready(venv)

    (project_dependencies / "pyproject.toml").write_text(
        '[tool.ruff]\nline-length = 100\n\n[tool.uv]\nlink-mode = "copy"\n'
    )
    venvs_module._resolved_environment.cache_clear()
    assert venvs_module.venv_is_current(venv / VENV_READY_MARKER, "uv run --locked")

    (project_dependencies / "pyproject.toml").write_text(
        '[tool.uv]\nlink-mode = "copy"\nno-build-isolation-package = ["demo"]\n'
    )
    venvs_module._resolved_environment.cache_clear()
    assert not venvs_module.venv_is_current(venv / VENV_READY_MARKER, "uv run --locked")


def test_resolved_environment_ignores_export_comments(
    project_dependencies, resolved_requirements
):
    resolved_requirements["text"] = "demo==1.0\n    # via alpha\n"
    with_alpha = venvs_module._resolved_environment(())
    resolved_requirements["text"] = "# exported by uv\ndemo==1.0\n    # via beta\n"
    venvs_module._resolved_environment.cache_clear()
    assert venvs_module._resolved_environment(()) == with_alpha


def test_command_extras_are_order_and_spelling_independent():
    assert venvs_module._command_extras(
        "uv run --locked --extra vllm --extra=nemo_gym --extra vllm --directory /x"
    ) == ("nemo_gym", "vllm")
    assert venvs_module._command_extras("uv run --locked") == ()


@pytest.fixture
def image_venvs(monkeypatch):
    monkeypatch.setenv("NEMO_RL_IMAGE_VENVS", "1")
    monkeypatch.setattr(
        venvs_module.ray,
        "nodes",
        lambda: pytest.fail("image venvs must not schedule a venv build"),
    )


def _image_venv(tmp_path, built_with: str | None) -> Path:
    venv = tmp_path / "demo.Worker"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()
    if built_with is not None:
        _mark_ready(venv, built_with)
    return venv


def test_image_venv_is_used_as_built_from_another_checkout(
    tmp_path, project_dependencies, image_venvs
):
    venv = _image_venv(
        tmp_path, "uv run --locked --extra vllm --directory /opt/nemo-rl"
    )
    launched = f"uv run --locked --extra vllm --directory {project_dependencies}"

    with patch.object(venvs_module.subprocess, "run", side_effect=AssertionError):
        python = venvs_module.create_local_venv_on_each_node(launched, "demo.Worker")

    assert python == str(venv / "bin" / "python")
    assert not venvs_module.venv_is_current(venv / VENV_READY_MARKER, launched)


def test_image_venv_rejects_another_resolved_environment(
    tmp_path, project_dependencies, image_venvs, resolved_requirements
):
    _image_venv(tmp_path, "uv run --locked --extra vllm")
    resolved_requirements["text"] = "demo==2.0\n"
    venvs_module._resolved_environment.cache_clear()

    with pytest.raises(RuntimeError, match="Overlay or rebuild the image"):
        venvs_module.create_local_venv_on_each_node(
            "uv run --locked --extra vllm", "demo.Worker"
        )


def test_image_venv_rejects_a_marker_without_a_resolved_environment(
    tmp_path, project_dependencies, image_venvs
):
    venv = _image_venv(tmp_path, None)
    (venv / VENV_READY_MARKER).write_text("0" * 64)

    with pytest.raises(RuntimeError, match="was marked by None"):
        venvs_module.create_local_venv_on_each_node("uv run --locked", "demo.Worker")


def test_image_venv_rejects_another_uv_before_comparing_environments(
    tmp_path, project_dependencies, image_venvs, resolved_requirements
):
    _image_venv(tmp_path, "uv run --locked")
    resolved_requirements["uv"] = "uv 2.0.0"

    with pytest.raises(
        RuntimeError, match="marked by uv 1.0.0 and this launch runs uv 2.0.0"
    ):
        venvs_module.create_local_venv_on_each_node("uv run --locked", "demo.Worker")


def test_image_venv_rejects_an_actor_the_image_does_not_carry(
    tmp_path, project_dependencies, image_venvs
):
    with pytest.raises(FileNotFoundError, match="no prebuilt venv for demo.Worker"):
        venvs_module.create_local_venv_on_each_node("uv run --locked", "demo.Worker")


def _editable_project(root: Path, relative: str, *, src: bool) -> None:
    project = root / relative
    project.mkdir(parents=True)
    if src:
        (project / "src").mkdir()
    (project / "pyproject.toml").touch()


@pytest.fixture
def editable_checkout(project_dependencies):
    (project_dependencies / "uv.lock").write_text(
        "version = 1\n"
        '[[package]]\nname = "demo"\nsource = { registry = "https://pypi.org/simple" }\n'
        '[[package]]\nname = "bridge"\nsource = { editable = "third/bridge" }\n'
        '[[package]]\nname = "flat"\nsource = { editable = "third/flat" }\n'
        '[[package]]\nname = "project"\nsource = { editable = "." }\n'
        '[[package]]\nname = "wheelhouse"\nsource = { directory = "third/wheels" }\n'
    )
    venvs_module._checkout_import_roots.cache_clear()
    yield project_dependencies
    venvs_module._checkout_import_roots.cache_clear()


def test_checkout_reaches_image_venvs_through_pythonpath(
    editable_checkout, image_venvs
):
    _editable_project(editable_checkout, "third/bridge", src=True)
    _editable_project(editable_checkout, "third/flat", src=False)
    flat = str(editable_checkout / "third/flat")

    result = add_checkout_to_pythonpath(
        {"PYTHONPATH": os.pathsep.join(["/hf/modules", flat])}
    )

    assert result["PYTHONPATH"].split(os.pathsep) == [
        str(editable_checkout),
        str(editable_checkout / "third/bridge/src"),
        flat,
        "/hf/modules",
    ]


def test_checkout_pythonpath_rejects_an_uninitialized_submodule(
    editable_checkout, image_venvs
):
    _editable_project(editable_checkout, "third/bridge", src=True)
    (editable_checkout / "third/flat").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="git submodule update"):
        add_checkout_to_pythonpath({})


def test_checkout_pythonpath_is_untouched_without_image_venvs(editable_checkout):
    env_vars = {"PYTHONPATH": "/project"}
    assert add_checkout_to_pythonpath(env_vars) is env_vars


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


def test_add_hf_modules_cache_to_pythonpath():
    result = add_hf_modules_cache_to_pythonpath(
        {
            "HF_MODULES_CACHE": "/hf/modules",
            "PYTHONPATH": f"/project{os.pathsep}/other",
        }
    )

    assert result["PYTHONPATH"].split(os.pathsep) == [
        "/hf/modules",
        "/project",
        "/other",
    ]


def test_add_hf_modules_cache_does_not_duplicate_pythonpath_entry():
    pythonpath = f"/project{os.pathsep}/hf/modules"

    result = add_hf_modules_cache_to_pythonpath(
        {"HF_MODULES_CACHE": "/hf/modules", "PYTHONPATH": pythonpath}
    )

    assert result["PYTHONPATH"] == pythonpath


def test_make_actor_runtime_env_builds_local_venv_for_uv_python_executable():
    """Mirrors the inline venv-creation logic that used to live in grpo.py."""
    with (
        patch(
            "nemo_rl.distributed.ray_actor_environment_registry.get_actor_python_env",
            return_value="uv run --group vllm",
        ) as mock_get_env,
        patch(
            "nemo_rl.utils.venvs.create_local_venv_on_each_node",
            return_value="/fake/venv/bin/python",
        ) as mock_create_venv,
    ):
        runtime_env = make_actor_runtime_env("some.module.SomeActor")

    mock_get_env.assert_called_once_with("some.module.SomeActor")
    mock_create_venv.assert_called_once_with(
        "uv run --group vllm", "some.module.SomeActor"
    )
    assert runtime_env["py_executable"] == "/fake/venv/bin/python"
    assert runtime_env["env_vars"]["VIRTUAL_ENV"] == "/fake/venv"
    assert runtime_env["env_vars"]["UV_PROJECT_ENVIRONMENT"] == "/fake/venv"
