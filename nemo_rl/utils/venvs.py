# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import hashlib
import logging
import os
import shlex
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import ray
from ray.util import placement_group

dir_path = os.path.dirname(os.path.abspath(__file__))
git_root = os.path.abspath(os.path.join(dir_path, "../.."))
DEFAULT_VENV_DIR = os.path.join(git_root, "venvs")

# Written into a venv only after `uv sync` + the exec command succeed. bin/python
# alone is a false readiness signal on shared filesystems: `uv venv` creates it
# long before packages land, so a crashed build leaves a venv that looks usable.
# `uv sync` is convergent (it repairs a partial venv), so an unmarked venv is
# simply rebuilt in place — no deletion needed.
#
# The marker holds the dependency fingerprint it was built from, so the same
# rebuild-in-place path also covers a venv that predates a dependency change.
VENV_READY_MARKER = "NEMO_RL_VENV_READY"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _dependency_fingerprint(py_executable: str) -> str:
    """Digest the inputs a worker venv resolves from.

    `uv run --locked` re-syncs the driver's environment only; worker venvs are
    reused whenever they are marked ready, so a lockfile bump would otherwise
    leave them serving the previous resolution indefinitely (after the vLLM
    0.20 -> 0.25 bump, generation workers kept importing 0.20 until 0.25-only
    code failed at refit). `pyproject.toml` is digested alongside `uv.lock`
    because `[tool.uv]` build settings change the installed environment without
    changing the resolution. The normalized worker command is part of the
    fingerprint because it selects the environment's extras; two actors can
    share the same lockfile while requiring different installed packages.
    """
    digest = hashlib.sha256()
    digest.update(b"py_executable\0")
    digest.update(shlex.join(shlex.split(py_executable)).encode())
    digest.update(b"\0")
    for name in ("uv.lock", "pyproject.toml"):
        path = Path(git_root) / name
        if path.exists():
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _mark_venv_ready(ready_marker: Path, py_executable: str) -> None:
    # Written via rename so a concurrent reader never sees a half-written
    # fingerprint and rebuilds a venv that is actually current.
    tmp = ready_marker.with_name(f"{ready_marker.name}.{os.getpid()}.tmp")
    tmp.write_text(_dependency_fingerprint(py_executable))
    os.replace(tmp, ready_marker)


def venv_is_current(ready_marker: Path, py_executable: str) -> bool:
    """Whether a venv is both ready and built from the current dependencies."""
    try:
        return ready_marker.read_text().strip() == _dependency_fingerprint(
            py_executable
        )
    except OSError:
        return False


@lru_cache(maxsize=None)
def create_local_venv(
    py_executable: str, venv_name: str, force_rebuild: bool = False
) -> str:
    """Create a virtual environment using uv and execute a command within it.

    The output can be used as a py_executable for a Ray worker assuming the worker
    nodes also have access to the same file system as the head node.

    This function is cached to avoid multiple calls to uv to create the same venv,
    which avoids duplicate logging.

    Args:
        py_executable (str): Command to run with the virtual environment (e.g., "uv.sh run --locked")
        venv_name (str): Name of the virtual environment (e.g., "foobar.Worker")
        force_rebuild (bool): If True, force rebuild the venv even if it already exists

    Returns:
        str: Path to the python executable in the created virtual environment
    """
    # This directory is where virtual environments will be installed
    # It is local to the driver process but should be visible to all worker nodes
    # If this directory is not accessible from worker nodes (e.g., on a distributed
    # cluster with non-shared filesystems), you may encounter errors when workers
    # try to access the virtual environments
    #
    # You can override this location by setting the NEMO_RL_VENV_DIR environment variable

    NEMO_RL_VENV_DIR = os.path.normpath(
        os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    )
    logger.info(f"NEMO_RL_VENV_DIR is set to {NEMO_RL_VENV_DIR}.")

    # Create the venv directory if it doesn't exist
    os.makedirs(NEMO_RL_VENV_DIR, exist_ok=True)

    # Full path to the virtual environment
    venv_path = os.path.join(NEMO_RL_VENV_DIR, venv_name)

    # Force rebuild if requested
    if force_rebuild and os.path.exists(venv_path):
        logger.info(f"Force rebuilding venv at {venv_path}")
        shutil.rmtree(venv_path)

    logger.info(f"Creating new venv at {venv_path}")

    # Any build attempt invalidates readiness until it completes.
    ready_marker = Path(venv_path) / VENV_READY_MARKER
    ready_marker.unlink(missing_ok=True)

    # Honor a pinned uv (UV env var) instead of PATH resolution: a personal uv
    # earlier on PATH can be too old for the project's [tool.uv] fields and
    # fail the build.
    uv = os.environ.get("UV", "uv")

    # Create the virtual environment
    uv_venv_cmd = [uv, "venv", "--allow-existing", venv_path]
    subprocess.run(uv_venv_cmd, check=True)

    # Execute the command with the virtual environment
    env = os.environ.copy()
    # NOTE: UV_PROJECT_ENVIRONMENT is appropriate here only b/c there should only be
    #  one call to this in the driver. It is not safe to use this in a multi-process
    #  context.
    #  https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
    env["UV_PROJECT_ENVIRONMENT"] = venv_path

    # Split the py_executable into command and arguments
    exec_cmd = shlex.split(py_executable)
    if exec_cmd and exec_cmd[0] == "uv":
        exec_cmd[0] = uv
    # Command doesn't matter, since `uv` syncs the environment no matter the command.
    exec_cmd.extend(["echo", f"Finished creating venv {venv_path}"])

    # Always run uv sync first to ensure the build requirements are set (for --no-build-isolation packages)
    subprocess.run([uv, "sync", "--directory", git_root], env=env, check=True)
    subprocess.run(exec_cmd, env=env, check=True)

    _mark_venv_ready(ready_marker, py_executable)

    # Return the path to the python executable in the virtual environment
    python_path = os.path.join(venv_path, "bin", "python")
    return python_path


# Ray-based helper to create a virtual environment on each Ray node
@ray.remote(num_cpus=1)  # pragma: no cover
def _env_builder(
    py_executable: str, venv_name: str, node_idx: int, force_rebuild: bool = False
):
    # Check if another node is already building
    NEMO_RL_VENV_DIR = os.path.normpath(
        os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    )
    venv_path = Path(NEMO_RL_VENV_DIR) / venv_name
    python_path = venv_path / "bin" / "python"
    ready_marker = venv_path / VENV_READY_MARKER
    started_file = venv_path / "STARTED_ENV_BUILDER"
    build_timeout = float(os.environ.get("NRL_VENV_BUILD_TIMEOUT_SECS", "3600"))

    # Skip early return if force_rebuild is True
    if not force_rebuild and python_path.exists() and ready_marker.exists():
        if venv_is_current(ready_marker, py_executable):
            logger.info(f"Using existing venv at {venv_path}")
            return str(python_path)
        logger.info(
            f"Rebuilding venv at {venv_path}: it was built from different "
            f"dependencies or worker command"
        )

    # Sleep to stagger node startup
    time.sleep(1 * node_idx)

    # A claim older than the build timeout is from a killed build (walltime,
    # OOM): its finally-cleanup never ran. Expire it so this run can rebuild
    # instead of waiting forever. Expiry is rename-based so exactly one of two
    # concurrent expirers wins — stat-then-unlink could delete the claim a
    # faster process just re-created.
    try:
        if (
            started_file.exists()
            and time.time() - started_file.stat().st_mtime > build_timeout
        ):
            logger.warning(
                f"Node {node_idx}: expiring stale build claim on {venv_name} "
                f"(older than {build_timeout}s; a previous build was killed)"
            )
            expired = started_file.with_name(started_file.name + ".expired")
            os.rename(started_file, expired)
            expired.unlink(missing_ok=True)
    except FileNotFoundError:
        pass

    # Create the venv directory if needed
    venv_path.mkdir(parents=True, exist_ok=True)

    # Atomically claim the build (O_EXCL): exactly one process builds, even
    # across concurrently-launched jobs sharing NEMO_RL_VENV_DIR.
    try:
        os.close(os.open(started_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        # Another process is building; wait for the readiness marker.
        logger.info(
            f"Node {node_idx}: Another node is building {venv_name}, skipping..."
        )
        deadline = time.monotonic() + build_timeout
        while time.monotonic() < deadline:
            if venv_is_current(ready_marker, py_executable) and python_path.exists():
                return str(python_path)
            if not started_file.exists():
                # The builder writes the marker and then unlinks the claim;
                # observing the unlink first is a benign race, so give the
                # marker one more poll before declaring the build dead.
                time.sleep(1)
                if (
                    venv_is_current(ready_marker, py_executable)
                    and python_path.exists()
                ):
                    return str(python_path)
                if ready_marker.exists():
                    raise RuntimeError(
                        f"The builder of venv {venv_path} completed against "
                        f"different dependencies or worker command than this "
                        f"job. Concurrent jobs must not share "
                        f"NEMO_RL_VENV_DIR across a dependency change; let one "
                        f"job settle the venvs first."
                    )
                raise RuntimeError(
                    f"The builder of venv {venv_path} exited without completing "
                    f"(no {VENV_READY_MARKER}); see its logs for the build error."
                )
            time.sleep(1)
        raise TimeoutError(
            f"Timed out after {build_timeout}s waiting for venv {venv_path} to "
            f"become ready. If the building job was killed, remove the stale "
            f"claim with: rm -rf {venv_path}  (or raise NRL_VENV_BUILD_TIMEOUT_SECS)."
        )

    try:
        if force_rebuild and venv_path.exists():
            # Rebuild from scratch while holding the claim: drop readiness
            # first (iterdir order is arbitrary and a reader must never see a
            # marked venv mid-wipe), then clear everything except the claim
            # file so waiters keep their signal.
            logger.info(f"Force rebuilding venv at {venv_path}")
            ready_marker.unlink(missing_ok=True)
            for child in venv_path.iterdir():
                if child == started_file:
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        # Create the virtual environment on this node (already cleared above,
        # so never forward force_rebuild — its rmtree would drop the claim).
        return create_local_venv(py_executable, venv_name)
    finally:
        # Clean up the started file
        started_file.unlink(missing_ok=True)


def create_local_venv_on_each_node(py_executable: str, venv_name: str):
    """Create a virtual environment on each Ray node.

    Args:
        py_executable (str): Command to run with the virtual environment
        venv_name (str): Name of the virtual environment

    Returns:
        str: Path to the python executable in the created virtual environment
    """
    # Skip nodes with 0 CPUs (e.g. unschedulable head nodes) — including them
    # makes the STRICT_SPREAD placement group infeasible.
    nodes = [
        n
        for n in ray.nodes()
        if n.get("Alive", False) and n.get("Resources", {}).get("CPU", 0) > 0
    ]
    num_nodes = len(nodes)
    # Reserve one CPU on each node using a STRICT_SPREAD placement group
    bundles = [{"CPU": 1} for _ in range(num_nodes)]
    pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    force_rebuild = os.environ.get("NRL_FORCE_REBUILD_VENVS", "false").lower() == "true"
    # Launch one actor per node
    actors = [
        _env_builder.options(placement_group=pg).remote(
            py_executable, venv_name, i, force_rebuild
        )
        for i, _ in enumerate(nodes)
    ]
    # ensure setup runs on each node
    paths = ray.get([actor for actor in actors])
    # Normalize paths to handle double slashes and other path inconsistencies
    normalized_paths = [os.path.normpath(p) for p in paths]
    assert len(set(normalized_paths)) == 1, (
        f"All nodes should have the same venv, but got: {set(normalized_paths)}"
    )

    # Clean up the placement group
    ray.util.remove_placement_group(pg)
    # Return mapping from node IP to venv python path
    return paths[0]


def make_actor_runtime_env(actor_class_fqn: str) -> dict:
    """Build a Ray ``runtime_env`` for one of our registered actors.

    Resolves the actor's tier-specific py_executable via the registry,
    materializes a per-node venv when uv-managed, and packages it with
    ``VIRTUAL_ENV`` / ``UV_PROJECT_ENVIRONMENT`` env vars so workers see
    the same interpreter as the driver.

    Used by ReplayBuffer, AsyncTrajectoryCollector, and SyncRolloutActor
    — three actors that need the VLLM tier's venv on every node. Also
    used by the SGLang router and SGLang generation engines (SGLANG tier).
    """
    # Local import — venvs.py is dep-light; the registry imports
    # PY_EXECUTABLES which transitively pulls heavier deps.
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )

    py_exec = get_actor_python_env(actor_class_fqn)
    if py_exec.startswith("uv"):
        py_exec = create_local_venv_on_each_node(py_exec, actor_class_fqn)
    venv = os.path.dirname(os.path.dirname(py_exec))  # strip bin/python
    return {
        "py_executable": py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": venv,
            "UV_PROJECT_ENVIRONMENT": venv,
        },
    }
