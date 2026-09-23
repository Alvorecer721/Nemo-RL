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
"""Guards on ACTOR_ENVIRONMENTS, the actor -> uv extras table.

docker/Dockerfile runs nemo_rl/distributed/actor_environments.py as a script from
the dependency layer to decide which venvs to pre-build, and the runtime registry
imports the same dict. The CSCS builder runs the same script with an explicit
``--actors`` selection read from ``infra/slurm/cscs/profiles``. These tests keep
the readers honest.
"""

import ast
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from unittest.mock import patch

import pytest

from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS, _build_stage
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import (
    PY_EXECUTABLES,
    git_root,
    uv_py_executable,
)

MODULE_PATH = Path(git_root) / "nemo_rl" / "distributed" / "actor_environments.py"

# Read straight from the environment: since #4020 the flag is applied inside
# PY_EXECUTABLES and uv_py_executable, so there is no module constant to import.
USE_SYSTEM_EXECUTABLE = os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1"

CONTROLLERS = {"AsyncTrajectoryCollector", "ReplayBuffer", "SyncRolloutActor"}
SELECTION = {
    "nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker",
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
}


@pytest.mark.parametrize("actor_fqn", sorted(ACTOR_ENVIRONMENTS))
def test_actor_extras_is_none_or_a_list_of_str(actor_fqn):
    """Each value is None or a list of strings.

    This deliberately does NOT re-check that the extras are declared in
    pyproject.toml. Importing this module imports the registry, which runs
    _reject_undeclared_extras() at import time, so an undeclared extra raises
    during collection and no assertion here is ever reached. A type error is
    different: ("mcore",) is a tuple of a real extra, so the import-time check --
    which only does set arithmetic -- passes it, and this is what catches it.
    """
    extras = ACTOR_ENVIRONMENTS[actor_fqn]
    if extras is None:
        return
    assert isinstance(extras, list) and all(isinstance(e, str) for e in extras), (
        f"{actor_fqn}: value must be None or a list of extras, got {extras!r}"
    )


@pytest.mark.parametrize("actor_fqn", sorted(ACTOR_ENVIRONMENTS))
def test_actor_module_exists(actor_fqn):
    """The FQN still points at a module that exists.

    Parsed, not imported: most of these pull in vllm or megatron.
    """
    module_name, _, class_name = actor_fqn.rpartition(".")
    path = Path(git_root) / (module_name.replace(".", "/") + ".py")
    if not path.exists():
        path = Path(git_root) / module_name.replace(".", "/") / "__init__.py"
    assert path.exists(), f"{actor_fqn}: no module file for {module_name}"

    defined = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update(a.asname or a.name.split(".")[-1] for a in node.names)
    assert class_name in defined, f"{actor_fqn}: {class_name} not defined in {path}"


@pytest.mark.skipif(
    USE_SYSTEM_EXECUTABLE,
    reason="NEMO_RL_PY_EXECUTABLES_SYSTEM=1 puts every actor on the driver interpreter",
)
def test_registry_matches_py_executables():
    """The generated py_executable is the string the worker actually needs."""
    expected = {
        ("vllm",): PY_EXECUTABLES.VLLM,
        ("vllm", "nemo_gym"): PY_EXECUTABLES.VLLM_GYM,
        ("sglang",): PY_EXECUTABLES.SGLANG,
        ("fsdp",): PY_EXECUTABLES.FSDP,
        ("automodel",): PY_EXECUTABLES.AUTOMODEL,
        ("mcore",): PY_EXECUTABLES.MCORE,
        ("trtllm",): PY_EXECUTABLES.TRTLLM,
        ("nemo_gym",): PY_EXECUTABLES.NEMO_GYM,
    }
    for actor_fqn, extras in ACTOR_ENVIRONMENTS.items():
        got = ACTOR_ENVIRONMENT_REGISTRY[actor_fqn]
        if extras is None:
            assert got == PY_EXECUTABLES.SYSTEM, actor_fqn
        else:
            assert got == expected.get(tuple(extras), uv_py_executable(extras)), (
                actor_fqn
            )


def test_every_extras_py_executable_is_wired_to_an_actor():
    """A PY_EXECUTABLES constant naming extras must be used by some actor.

    This branch builds ACTOR_ENVIRONMENT_REGISTRY from ACTOR_ENVIRONMENTS instead of
    the literal dict main keeps in ray_actor_environment_registry.py. When main changes
    which extras an actor needs -- as #4009 did, moving the vLLM workers onto
    PY_EXECUTABLES.VLLM_GYM so token capture can import nemo_gym -- a rebase drops the
    literal dict and the change is silently lost. A new constant that no actor uses is
    the signature of exactly that miss.

    PY_EXECUTABLES.BASE is excluded: it names no extra and is not an actor environment.
    """
    unused = []
    for name in sorted(n for n in dir(PY_EXECUTABLES) if n.isupper()):
        value = getattr(PY_EXECUTABLES, name)
        if "--extra" not in value:
            continue
        if not any(
            uv_py_executable(extras) == value
            for extras in ACTOR_ENVIRONMENTS.values()
            if extras is not None
        ):
            unused.append(name)
    assert not unused, (
        f"PY_EXECUTABLES {unused} name extras but no actor in ACTOR_ENVIRONMENTS uses "
        "them. Either an actor's extras were not carried over from "
        "nemo_rl/distributed/ray_actor_environment_registry.py on main, or the "
        "constant is dead and should be deleted."
    )


def test_actor_environments_module_is_stdlib_only():
    """docker/Dockerfile runs this module from the dependency layer.

    Only pyproject.toml, uv.lock and a couple of nemo_rl files exist there, so an
    import of anything else -- especially anything from nemo_rl -- breaks the image
    build in its most expensive layer.
    """
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        roots = []
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        for root in roots:
            assert root in stdlib, (
                f"{MODULE_PATH.name} imports {root!r}, which is not in the standard "
                "library. This module must stay dependency-free -- docker/Dockerfile "
                "runs it from a layer where only pyproject.toml and uv.lock exist."
            )


def _run_cli(*args: str, script: Path = MODULE_PATH) -> subprocess.CompletedProcess:
    """Run the manifest the way the Dockerfile does: isolated, no site-packages."""
    return subprocess.run(
        [sys.executable, "-I", "-S", str(script), *args],
        capture_output=True,
        text=True,
        cwd="/tmp",
    )


def _rows(*args: str) -> dict[str, tuple[str, str]]:
    result = _run_cli(*args)
    assert result.returncode == 0, result.stderr
    rows = [line.split("\t") for line in result.stdout.splitlines()]
    assert rows
    assert all(len(row) == 3 for row in rows)
    assert rows == sorted(rows)
    assert len(rows) == len({row[0] for row in rows})
    return {actor: (stage, flags) for actor, stage, flags in rows}


def test_script_rejects_a_typo_instead_of_printing_nothing():
    """A mistyped stage, skip extra or actor must fail, not emit an empty list.

    The Dockerfile pipes this into a loop, so silently printing zero rows would
    build zero venvs. `test -s` catches that in the deps layer, but the guard is
    what makes the failure say which word was wrong.
    """
    for argv in (["badstage"], ["all", "notanextra"], ["--actors", "missing.Worker"]):
        proc = _run_cli(*argv)
        assert proc.returncode == 2, f"{argv} should be rejected, got {proc.returncode}"
        assert not proc.stdout.strip(), f"{argv} printed rows despite being invalid"
        assert proc.stderr
    assert "unregistered actors" in _run_cli("--actors", "missing.Worker").stderr


def test_registry_import_rejects_an_undeclared_extra(monkeypatch):
    """A typo'd extra must fail while IMPORTING the registry, not at venv creation.

    On the image-build path `prefetch_venvs.py` reports the per-actor error only
    at exit, so without this guard a typo is found one expensive layer too late.
    Importing the module fresh is the point -- calling the check directly would
    still pass if nothing invoked it at import.
    """
    import nemo_rl.distributed.actor_environments as table

    registry_path = (
        Path(git_root) / "nemo_rl" / "distributed" / "ray_actor_environment_registry.py"
    )

    def _import_registry_fresh(name):
        spec = importlib.util.spec_from_file_location(name, registry_path)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    _import_registry_fresh("_registry_clean")  # real table: must not raise

    monkeypatch.setattr(
        table,
        "ACTOR_ENVIRONMENTS",
        {
            **table.ACTOR_ENVIRONMENTS,
            "nemo_rl.fake.Worker": ["definitely_not_an_extra"],
        },
    )
    with pytest.raises(ValueError, match="definitely_not_an_extra"):
        _import_registry_fresh("_registry_typo")


def test_script_emits_the_stage_and_extra_flags_each_actor_needs():
    """The script lists exactly the venvs the runtime expects, with the right columns.

    The Dockerfile builds each venv from all three columns. Column 3 is what
    `uv sync $extras` consumes, and column 2 is what the deps layer branches on to
    leave the TRT-LLM venv base-only until its wheel exists. Comparing the whole
    dict covers the actor list too, since dict equality requires the same keys.
    An empty ``--actors`` selection means every actor, so both forms must agree.
    """
    expected = {
        fqn: (_build_stage(extras), " ".join(f"--extra {e}" for e in extras))
        for fqn, extras in ACTOR_ENVIRONMENTS.items()
        if extras is not None
    }
    rows = _rows("all")
    assert rows == expected
    assert rows == _rows("--actors", "")
    assert (
        rows[
            "nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker"
        ][0]
        == "trtllm"
    )


def test_explicit_actor_selection_is_independent_of_site_profiles():
    """``--actors`` selects exactly the named actors; skip extras still apply."""
    rows = _rows("--actors", " ".join(sorted(SELECTION)))
    assert set(rows) == SELECTION
    rows = _rows("--actors", " ".join(sorted(SELECTION)), "all", "vllm")
    assert {actor.rsplit(".", 1)[1] for actor in rows} == {"MegatronValueWorker"}


def test_script_skips_by_extra_not_by_name():
    """SKIP_VLLM_BUILD must drop actors that need vllm even without 'vllm' in the name."""
    rows = _rows("all", "vllm", "automodel")
    names = {actor.rsplit(".", 1)[1] for actor in rows}
    assert not names & CONTROLLERS
    assert "VllmQuantGenerationWorker" not in names
    assert "DTensorQuantPolicyWorker" not in names
    assert "MegatronPolicyWorker" in names
    assert all(
        "vllm" not in flags and "automodel" not in flags for _, flags in rows.values()
    )


def test_stage_selection_emits_only_trtllm():
    assert _rows("trtllm") == {
        "nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker": (
            "trtllm",
            "--extra trtllm",
        )
    }


def test_script_runs_from_dependency_layer_without_package():
    """A bare copy of the file, outside the package, must still emit the rows."""
    with tempfile.TemporaryDirectory() as directory:
        standalone = Path(directory) / "actor_environments.py"
        shutil.copyfile(MODULE_PATH, standalone)
        result = _run_cli(script=standalone)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == sum(
        1 for extras in ACTOR_ENVIRONMENTS.values() if extras is not None
    )


def test_docker_manifest_invocation_without_python_on_path():
    """The Dockerfile's manifest command must work with only the driver venv."""
    docker_lines = (
        (Path(git_root) / "docker/Dockerfile")
        .read_text()
        .replace("\\\n", "")
        .splitlines()
    )
    invocation = next(
        line
        for line in docker_lines
        if "nemo_rl/distributed/actor_environments.py" in line
        and not line.startswith("COPY")
    ).replace("/opt/actor_venvs.tsv", '"$manifest_output"')
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        prefix = fixture / "driver venv"
        venv.EnvBuilder(with_pip=False).create(prefix)
        empty_path = fixture / "empty-path"
        empty_path.mkdir()
        output = fixture / "actors.tsv"
        result = subprocess.run(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                "SKIP_EXTRAS=(trtllm)\n" + invocation,
            ],
            cwd=git_root,
            env={
                "PATH": str(empty_path),
                "UV_PROJECT_ENVIRONMENT": str(prefix),
                "NRL_ACTORS": "\n".join(sorted(SELECTION)),
                "manifest_output": str(output),
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        rows = [line.split("\t") for line in output.read_text().splitlines()]
    assert {row[0] for row in rows} == SELECTION
    assert all(row[1] == "deps" for row in rows)


def test_fingerprint_covers_the_actor_table():
    """Editing an actor's extras must invalidate the container fingerprint.

    Venvs at NEMO_RL_VENV_DIR are reused rather than rebuilt, and nothing prunes
    them (the base sync runs --inexact and `uv run` is inexact by default). So a
    changed extras list has to trip _check_container_fingerprint(), which is what
    tells the user to set NRL_FORCE_REBUILD_VENVS=true.
    """
    spec = importlib.util.spec_from_file_location(
        "_gen_fingerprint", Path(git_root) / "tools" / "generate_fingerprint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fingerprint = module.generate_fingerprint()
    assert "nemo_rl/distributed/actor_environments.py" in fingerprint, (
        "tools/generate_fingerprint.py must hash the actor -> extras table, or a "
        "changed actor environment leaves a stale venv with no warning"
    )
    # Check the value is this file's hash, not merely present and non-empty --
    # pointing the entry at some other file passes the weaker check.
    assert (
        fingerprint["nemo_rl/distributed/actor_environments.py"]
        == hashlib.md5(MODULE_PATH.read_bytes()).hexdigest()
    )


def test_actor_manifest_change_invalidates_fingerprint():
    spec = importlib.util.spec_from_file_location(
        "_gen_fingerprint_fixture", Path(git_root) / "tools" / "generate_fingerprint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        (fixture / "pyproject.toml").write_text("[project]\n")
        (fixture / "uv.lock").write_text("version = 1\n")
        actor_path = fixture / "nemo_rl/distributed/actor_environments.py"
        actor_path.parent.mkdir(parents=True)
        actor_path.write_text("ACTOR_ENVIRONMENTS = {'Worker': ['vllm']}\n")
        with patch.object(module, "get_repo_root", return_value=fixture):
            before = module.generate_fingerprint()
            actor_path.write_text("ACTOR_ENVIRONMENTS = {'Worker': ['mcore']}\n")
            after = module.generate_fingerprint()
    assert before != after
    assert before["pyproject.toml"] == after["pyproject.toml"]
    assert before["uv.lock"] == after["uv.lock"]
    assert (
        before["nemo_rl/distributed/actor_environments.py"]
        != after["nemo_rl/distributed/actor_environments.py"]
    )
