# Image Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every image gate (import check, worker readiness marker, image qualification, overlay restamp, hermetic cache key) key on what the image bakes, so a source sync with no dependency change reuses the qualified image with no rebuild and no manual reasoning, and a dependency change is classified as reusable, overlay, or rebuild by a tool.

**Architecture:** One stdlib-only module, `tools/image_identity.py`, computes a schema 2 identity: a SHA-256 of `uv export --frozen` per actor environment, the pins of submodules the lock does not declare editable, and the pinned uv version. `tools/generate_fingerprint.py` becomes a shim over it so the Dockerfile, `nemo_rl/__init__.py`, and the slurm scripts keep their contract. The worker readiness digest, the qualification probe, the overlay restamp, and the hermetic manifest all switch to the same identity. A new `tools/image_compat.py` diffs an image's stamp against the source tree and prints the verdict; the fork CI gate runs it on every PR against a committed copy of the qualified image's stamp.

**Tech Stack:** Python 3.13 stdlib (`tomllib`, `hashlib`, `subprocess`, `json`), uv 0.11.28 (`uv export --frozen`), git plumbing, pytest, GitHub Actions, Slurm/Podman build scripts.

**Spec:** `docs/superpowers/specs/2026-09-17-image-identity.md`. The plan argues from it; read both.

## PR placement

Branch `design/2026-09-17-image-identity` from `origin/main` (`9957b377a`, before #42 lands; rebase onto main after #42 merges, no file overlaps expected: #42 touches none of the files below). One PR for Tasks 1 through 6. Task 7 (migration of the in-use image and the sync of upstream `d4b446886`) is a separate PR that depends on this one and on #42.

## Global Constraints

- `tools/image_identity.py`, `tools/generate_fingerprint.py`, and `tools/restamp_overlay_fingerprint.py` import only the standard library. The Dockerfile runs them before dependencies exist and `nemo_rl/__init__.py` runs the first at import.
- Never `import nemo_rl` from `tools/image_identity.py`; `nemo_rl/__init__.py` calls it at import and would recurse. Loaders use `importlib.util.spec_from_file_location` on the repo-relative path.
- The uv version is read from the line `ARG UV_VERSION=<x.y.z>` in `docker/Dockerfile`. No second copy anywhere.
- All gates fail closed: missing uv, wrong uv version, schema other than 2, or an unreadable input is an error, never a skipped check, when `NRL_CONTAINER=1` or when running in the Dockerfile.
- Commits are DCO-signed (`git commit -s`), no Co-Authored-By lines, conventional-commit subjects.
- Tests run with `uv run --group test python -m pytest`; on CSCS use the container pattern from the unit-test memory note. Tests that need uv locate it with `shutil.which("uv")` and skip when absent, as `tests/unit/tools/test_image_workers.py` does.
- Do not modify `nemo_rl/__init__.py` beyond the one exception clause in Task 2; the dict comparison stays upstream's.

## Assumptions

Copied from the spec; every task inherits them.

- Editable path dependencies (Megatron-Bridge with nested Megatron-LM, Automodel, Gym) run from the mounted checkout after the frozen worker re-sync. Megatron-LM's pybind11 helper is rebuilt in place by that re-sync, offline, as today.
- Worker syncs are `--offline --frozen`; a wheel missing from the image's uv cache is the only launch-time failure a lock change causes.
- `uv export --frozen` output is stable for a fixed uv version and differs across versions; identities are comparable only under the same version.
- Gym's per-server venvs under `/opt/gym_venvs` are out of scope.
- The identity covers every actor in `ACTOR_ENVIRONMENTS`, not only the image profile's subset, as the current fingerprint does.

## No backward compatibility

- Schema 1 stamps are rejected everywhere. In-use images get a committed schema 2 stamp and a launcher mount (Task 7); nothing reads schema 1.
- `NEMO_RL_BUILD_FINGERPRINT_B64` is removed; `NEMO_RL_SUBMODULE_PINS_B64` replaces it in both Dockerfiles and both build scripts.
- The hermetic manifest's `submodules` input is removed; existing hermetic manifests stop matching and the next base build reruns the hermetic stage once.
- Existing worker readiness markers are invalid once; the first launch re-syncs each prebuilt venv offline from the image cache.
- The fixtures in `tests/unit/tools/test_image_workers.py`, `test_overlay_fingerprint.py`, `test_image_build_manifest.py`, `tests/unit/test_version_check.py`, and `tests/unit/distributed/test_actor_environments.py::test_fingerprint_covers_the_actor_table` are rewritten for schema 2.

## File structure

- Create `tools/image_identity.py` — identity computation, submodule classification, cache, delta classification helpers.
- Modify `tools/generate_fingerprint.py` — shim: locate `image_identity.py`, print `compute()`; keep `SOURCE_FINGERPRINT_FILENAME` and the embedded-pins fallback.
- Modify `nemo_rl/utils/venvs.py:92-117` — `_dependency_fingerprint` digests the export for the command's extras.
- Modify `tools/check_image_workers.py:60-67` — schema 2 validation in the probe.
- Modify `tools/restamp_overlay_fingerprint.py` — compare `baked_submodules`, recompute in-container.
- Modify `tools/image_build_manifest.py:181-199, 33-44` — schema 2 fingerprint input, drop `recursive_submodules` from inputs.
- Modify `docker/Dockerfile:483, 508-521` and `docker/Dockerfile.overlay:169, 181-185` — `NEMO_RL_SUBMODULE_PINS_B64`, in-container identity.
- Modify `infra/slurm/cscs/build_nemo_rl_image.slurm:175, 295, 360` and `infra/slurm/cscs/build_nemo_rl_overlay_image.slurm:130, 239` — pass pins, drop the fingerprint arg.
- Create `tools/image_compat.py` — verdict CLI.
- Create `infra/slurm/cscs/environments/nemo_rl_vllm026_ncclext.fingerprint.json` (Task 7 fills the real value; Task 6 wires the path).
- Modify `.github/workflows/cicd-main.yml` — hosted `image-compat` job, added to `HOSTED_CHECKS_SUCCESS`.
- Modify `infra/slurm/cscs/README.md` (overlay section) and `docs/design-docs/dependency-management.md` (Container Version Checking section).
- Tests: create `tests/unit/tools/test_image_identity.py`, `tests/unit/tools/test_image_compat.py`; rewrite the five fixtures named above.

---

### Task 1: `tools/image_identity.py` and its tests

**Files:**
- Create: `tools/image_identity.py`
- Create: `tests/unit/tools/test_image_identity.py`

**Interfaces:**
- Produces:
  - `pinned_uv_version(repo_root: Path) -> str` — the `ARG UV_VERSION=` value from `docker/Dockerfile`; `ValueError` if absent.
  - `resolve_uv(repo_root: Path) -> str` — path of a uv whose `uv --version` equals the pinned version; searches `$UV`, then `/root/.local/bin/uv`, then `PATH`; `RuntimeError` naming what was found otherwise.
  - `export_text(repo_root: Path, extras: Sequence[str], uv: str) -> str` — the `uv export` output with `# via` comment lines removed.
  - `export_digest(repo_root, extras, uv) -> str` — SHA-256 hex of `export_text`.
  - `editable_submodules(repo_root: Path) -> set[str]` — submodule paths that `uv.lock` declares as `source = { editable = "<path>" }` (a nested editable path counts for the top-level submodule that contains it).
  - `submodule_pins(repo_root: Path) -> dict[str, str]` — `git submodule status` map (non-recursive), or the pins embedded in `.nemo_rl_source_fingerprint.json` when `.git` is absent; `RuntimeError` when neither exists.
  - `compute(repo_root: Path, *, pins: dict[str, str] | None = None) -> dict` — the schema 2 identity; reads `ACTOR_ENVIRONMENTS` by executing `nemo_rl/distributed/actor_environments.py` with `runpy.run_path` (it is stdlib-only by contract).
  - `cached_compute(repo_root: Path, cache_dir: Path, **kw) -> dict` — `compute` memoized on a SHA-256 of `uv.lock`, `pyproject.toml`, `actor_environments.py`, the pin map, and the uv version.
  - `CACHE_KEY_INPUTS = ("uv.lock", "pyproject.toml", "nemo_rl/distributed/actor_environments.py")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/tools/test_image_identity.py
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/image_identity.py"
UV = shutil.which("uv")

pytestmark = pytest.mark.skipif(UV is None, reason="uv is required to lock the fixture")


def _load():
    spec = importlib.util.spec_from_file_location("image_identity", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docker").mkdir(parents=True)
    (repo / "nemo_rl/distributed").mkdir(parents=True)
    (repo / "3rdparty/editable_pkg").mkdir(parents=True)
    (repo / "3rdparty/baked_pkg").mkdir(parents=True)
    version = subprocess.check_output([UV, "--version"], text=True).split()[1]
    (repo / "docker/Dockerfile").write_text(f"ARG UV_VERSION={version}\n")
    (repo / "3rdparty/editable_pkg/pyproject.toml").write_text(
        '[project]\nname = "editable-pkg"\nversion = "1.0"\n'
        "[build-system]\nrequires = ['setuptools']\nbuild-backend = 'setuptools.build_meta'\n"
    )
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "1.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["editable-pkg"]\n'
        "[project.optional-dependencies]\na = []\nb = []\n"
        "[tool.uv.sources]\neditable-pkg = { path = '3rdparty/editable_pkg', editable = true }\n"
    )
    (repo / "nemo_rl/distributed/actor_environments.py").write_text(
        "ACTOR_ENVIRONMENTS = {'x.A': ['a'], 'x.B': ['b'], 'x.Driver': None}\n"
    )
    monkeypatch.setenv("UV", UV)
    monkeypatch.setenv("UV_OFFLINE", "1")
    subprocess.run([UV, "--no-config", "lock", "--directory", str(repo)], check=True, capture_output=True)
    return repo


PINS = {"3rdparty/editable_pkg": "aaa", "3rdparty/baked_pkg": "bbb"}


def test_compute_is_deterministic_and_keyed_per_actor(fixture_repo):
    m = _load()
    first = m.compute(fixture_repo, pins=PINS)
    second = m.compute(fixture_repo, pins=PINS)
    assert first == second
    assert first["schema"] == 2
    assert set(first["actors"]) == {"x.A", "x.B"}
    assert first["actors"]["x.A"] != first["base"]


def test_lock_comment_reordering_does_not_change_identity(fixture_repo):
    m = _load()
    before = m.compute(fixture_repo, pins=PINS)
    lock = fixture_repo / "uv.lock"
    lock.write_text(lock.read_text() + "\n# trailing comment\n")
    assert m.compute(fixture_repo, pins=PINS) == before


def test_editable_pins_drop_out_and_unmentioned_pins_are_baked(fixture_repo):
    m = _load()
    identity = m.compute(fixture_repo, pins=PINS)
    assert identity["baked_submodules"] == {"3rdparty/baked_pkg": "bbb"}
    assert m.editable_submodules(fixture_repo) == {"3rdparty/editable_pkg"}


def test_wrong_uv_version_fails_closed(fixture_repo):
    m = _load()
    (fixture_repo / "docker/Dockerfile").write_text("ARG UV_VERSION=0.0.1\n")
    with pytest.raises(RuntimeError, match="0.0.1"):
        m.compute(fixture_repo, pins=PINS)


def test_cache_hits_until_an_input_changes(fixture_repo, tmp_path):
    m = _load()
    cache = tmp_path / "cache"
    first = m.cached_compute(fixture_repo, cache, pins=PINS)
    assert len(list(cache.iterdir())) == 1
    assert m.cached_compute(fixture_repo, cache, pins=PINS) == first
    (fixture_repo / "nemo_rl/distributed/actor_environments.py").write_text(
        "ACTOR_ENVIRONMENTS = {'x.A': ['a'], 'x.Driver': None}\n"
    )
    third = m.cached_compute(fixture_repo, cache, pins=PINS)
    assert set(third["actors"]) == {"x.A"}
    assert len(list(cache.iterdir())) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group test python -m pytest tests/unit/tools/test_image_identity.py -v`
Expected: FAIL with `FileNotFoundError` or `AttributeError` on `tools/image_identity.py`.

- [ ] **Step 3: Write the module**

```python
# tools/image_identity.py
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute the image identity: what an image bakes, not which files changed.

Stdlib only. Runs in the Dockerfile before dependencies exist, at
`import nemo_rl` inside containers, and from the worker readiness check. Never
imports nemo_rl.
"""

import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Sequence

SCHEMA = 2
SOURCE_FINGERPRINT_FILENAME = ".nemo_rl_source_fingerprint.json"
CACHE_KEY_INPUTS = (
    "uv.lock",
    "pyproject.toml",
    "nemo_rl/distributed/actor_environments.py",
)
_EXPORT_ARGS = (
    "export",
    "--frozen",
    "--no-dev",
    "--no-header",
    "--no-emit-project",
    "--format",
    "requirements-txt",
)


def pinned_uv_version(repo_root: Path) -> str:
    dockerfile = (repo_root / "docker/Dockerfile").read_text()
    match = re.search(r"^ARG UV_VERSION=(\S+)$", dockerfile, re.MULTILINE)
    if match is None:
        raise ValueError("docker/Dockerfile does not pin ARG UV_VERSION")
    return match.group(1)


def resolve_uv(repo_root: Path) -> str:
    pinned = pinned_uv_version(repo_root)
    candidates = [os.environ.get("UV"), "/root/.local/bin/uv", shutil.which("uv")]
    found = []
    for candidate in candidates:
        if not candidate or not os.access(candidate, os.X_OK):
            continue
        version = subprocess.run(
            [candidate, "--version"], capture_output=True, text=True, check=True
        ).stdout.split()[1]
        if version == pinned:
            return candidate
        found.append(f"{candidate} ({version})")
    raise RuntimeError(
        f"No uv {pinned} found; the identity requires the version pinned in "
        f"docker/Dockerfile. Found: {found or 'nothing'}"
    )


def export_text(repo_root: Path, extras: Sequence[str], uv: str) -> str:
    command = [uv, "--no-config", *_EXPORT_ARGS, "--directory", str(repo_root)]
    for extra in extras:
        command.extend(["--extra", extra])
    output = subprocess.run(
        command,
        env={**os.environ, "UV_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return "".join(
        line for line in output.splitlines(keepends=True)
        if not line.lstrip().startswith("#")
    )


def export_digest(repo_root: Path, extras: Sequence[str], uv: str) -> str:
    return hashlib.sha256(export_text(repo_root, extras, uv).encode()).hexdigest()


def editable_submodules(repo_root: Path) -> set[str]:
    lock = tomllib.loads((repo_root / "uv.lock").read_text())
    editable = set()
    for package in lock.get("package", []):
        path = package.get("source", {}).get("editable")
        if path:
            editable.add(Path(path).as_posix())
    return editable


def _git_submodule_pins(repo_root: Path) -> dict[str, str]:
    result = subprocess.run(
        ["git", f"--git-dir={repo_root}/.git", f"--work-tree={repo_root}",
         "submodule", "status"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    pins = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            pins[parts[1]] = parts[0].lstrip("+-U")
    return pins


def submodule_pins(repo_root: Path) -> dict[str, str]:
    if (repo_root / ".git").exists():
        return _git_submodule_pins(repo_root)
    embedded = repo_root / SOURCE_FINGERPRINT_FILENAME
    if embedded.exists():
        stamp = json.loads(embedded.read_text())
        pins = stamp.get("submodule_pins")
        if isinstance(pins, dict):
            return dict(pins)
    raise RuntimeError(
        "Cannot read submodule pins: no .git directory and no embedded "
        f"{SOURCE_FINGERPRINT_FILENAME} with a submodule_pins map"
    )


def _baked(pins: dict[str, str], editable: set[str]) -> dict[str, str]:
    baked = {}
    for path, sha in sorted(pins.items()):
        if not any(e == path or e.startswith(path + "/") for e in editable):
            baked[path] = sha
    return baked


def _actor_extras(repo_root: Path) -> dict[str, list[str] | None]:
    table = runpy.run_path(
        str(repo_root / "nemo_rl/distributed/actor_environments.py"),
        run_name="image_identity",
    )
    return table["ACTOR_ENVIRONMENTS"]


def compute(repo_root: Path, *, pins: dict[str, str] | None = None) -> dict:
    repo_root = Path(repo_root).resolve()
    uv = resolve_uv(repo_root)
    pins = submodule_pins(repo_root) if pins is None else pins
    actors = {
        actor: export_digest(repo_root, extras, uv)
        for actor, extras in sorted(_actor_extras(repo_root).items())
        if extras is not None
    }
    return {
        "schema": SCHEMA,
        "uv": pinned_uv_version(repo_root),
        "base": export_digest(repo_root, (), uv),
        "actors": actors,
        "baked_submodules": _baked(pins, editable_submodules(repo_root)),
        "submodule_pins": dict(sorted(pins.items())),
    }


def cache_key(repo_root: Path, pins: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in CACHE_KEY_INPUTS:
        digest.update(name.encode() + b"\0")
        digest.update((repo_root / name).read_bytes() + b"\0")
    digest.update(json.dumps(pins, sort_keys=True).encode() + b"\0")
    digest.update(pinned_uv_version(repo_root).encode())
    return digest.hexdigest()


def cached_compute(repo_root: Path, cache_dir: Path, *, pins=None) -> dict:
    repo_root = Path(repo_root).resolve()
    pins = submodule_pins(repo_root) if pins is None else pins
    entry = Path(cache_dir) / f"{cache_key(repo_root, pins)}.json"
    if entry.exists():
        return json.loads(entry.read_text())
    identity = compute(repo_root, pins=pins)
    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp = entry.with_name(f"{entry.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(identity, indent=2, sort_keys=True))
    os.replace(tmp, entry)
    return identity


def validate(identity: object, *, label: str) -> dict:
    if not isinstance(identity, dict) or identity.get("schema") != SCHEMA:
        raise ValueError(f"{label} fingerprint is not schema {SCHEMA}")
    for key in ("uv", "base", "actors", "baked_submodules"):
        if key not in identity:
            raise ValueError(f"{label} fingerprint lacks {key!r}")
    if not isinstance(identity["actors"], dict) or not identity["actors"]:
        raise ValueError(f"{label} fingerprint has no actor environments")
    return identity
```

The `submodule_pins` key carries the full pin map so a release tree without
`.git` can recover them (the embedded-pins fallback); only `baked_submodules`
takes part in comparisons. `validate` is what every gate calls.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group test python -m pytest tests/unit/tools/test_image_identity.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/image_identity.py tests/unit/tools/test_image_identity.py
git commit -s -m "build: compute the image identity from resolved environments"
```

---

### Task 2: `generate_fingerprint.py` shim, Dockerfile pins argument, import gate

**Files:**
- Modify: `tools/generate_fingerprint.py` (replace body)
- Modify: `nemo_rl/__init__.py:196-201` (one clause)
- Modify: `docker/Dockerfile:483, 508-521`
- Modify: `docker/Dockerfile.overlay:169`
- Modify: `infra/slurm/cscs/build_nemo_rl_image.slurm:175, 295, 360`
- Modify: `infra/slurm/cscs/build_nemo_rl_overlay_image.slurm:130, 239`
- Modify: `tests/unit/test_version_check.py`, `tests/unit/distributed/test_actor_environments.py:367-380`

**Interfaces:**
- Consumes: `image_identity.compute`, `image_identity.submodule_pins`.
- Produces: `python tools/generate_fingerprint.py` prints the schema 2 identity; env `NEMO_RL_SUBMODULE_PINS_B64` (base64 JSON `{path: sha}`) overrides git discovery.

- [ ] **Step 1: Rewrite the failing tests**

In `tests/unit/test_version_check.py` replace `test_cscs_build_bakes_valid_json_with_separate_hermetic_inputs` assertions on `BUILD_FINGERPRINT_B64` with:

```python
    assert (
        'SUBMODULE_PINS_B64=$("$HOST_PYTHON" tools/generate_fingerprint.py --pins-only '
        "| base64 -w0)" in build_script
    )
    assert "NEMO_RL_BUILD_FINGERPRINT_B64" not in build_script
```

Replace `test_generate_fingerprint_uses_embedded_submodules_without_git` with a test that writes `.nemo_rl_source_fingerprint.json` containing `{"schema": 2, "submodule_pins": {"3rdparty/x": "abc"}}` into a fixture repo built like Task 1's, runs the shim with `--pins-only`, and asserts it prints `{"3rdparty/x": "abc"}`. In `TestContainerFingerprintCheck` change every fingerprint literal to a schema 2 dict (`{"schema": 2, "uv": "0.11.28", "base": "b", "actors": {"x.A": "a"}, "baked_submodules": {}}`); the mismatch test changes one actor digest.

In `tests/unit/distributed/test_actor_environments.py::test_fingerprint_covers_the_actor_table` assert instead that adding an actor with a new extras list to a copy of the table changes `compute()["actors"]` keys (use the Task 1 fixture pattern; skip when uv is absent).

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --group test python -m pytest tests/unit/test_version_check.py tests/unit/distributed/test_actor_environments.py -k "fingerprint or version_check" -v`
Expected: FAIL on the missing `--pins-only` flag and the schema 1 output.

- [ ] **Step 3: Write the shim**

```python
# tools/generate_fingerprint.py  (module docstring and copyright header kept)
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path

SOURCE_FINGERPRINT_FILENAME = ".nemo_rl_source_fingerprint.json"


def get_repo_root() -> Path:
    return Path(__file__).parent.resolve().parent


def _identity_module():
    path = get_repo_root() / "tools/image_identity.py"
    spec = importlib.util.spec_from_file_location("image_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def requested_pins(module, repo_root: Path) -> dict[str, str]:
    encoded = os.environ.get("NEMO_RL_SUBMODULE_PINS_B64")
    if encoded:
        return json.loads(base64.b64decode(encoded, validate=True))
    return module.submodule_pins(repo_root)


def generate_fingerprint() -> dict:
    module = _identity_module()
    repo_root = get_repo_root()
    return module.compute(repo_root, pins=requested_pins(module, repo_root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins-only", action="store_true",
                        help="print only the submodule pin map (stdlib, no uv)")
    args = parser.parse_args()
    module = _identity_module()
    repo_root = get_repo_root()
    if args.pins_only:
        print(json.dumps(requested_pins(module, repo_root), sort_keys=True))
        return
    print(json.dumps(generate_fingerprint(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

`generate_fingerprint()` keeps its name because `check_image_workers.py`,
`image_build_manifest.py`, and the tests call it through `runpy`.

- [ ] **Step 4: Dockerfile and build scripts**

`docker/Dockerfile` line 483: `ARG NEMO_RL_BUILD_FINGERPRINT_B64` becomes `ARG NEMO_RL_SUBMODULE_PINS_B64`. Lines 508-521 become:

```dockerfile
# Stamp the image identity in-container with the pinned uv. A tracked-files-only
# build context has no .git, so the launcher supplies the submodule pins.
RUN NEMO_RL_SUBMODULE_PINS_B64="${NEMO_RL_SUBMODULE_PINS_B64:-}" \
    python tools/generate_fingerprint.py \
        | tee /opt/nemo_rl_container_fingerprint \
            > /opt/nemo-rl/.nemo_rl_source_fingerprint.json
```

`docker/Dockerfile.overlay` line 169: same rename; the restamp step is rewritten in Task 4.

`infra/slurm/cscs/build_nemo_rl_image.slurm` line 175:

```bash
SUBMODULE_PINS_B64=$("$HOST_PYTHON" tools/generate_fingerprint.py --pins-only | base64 -w0)
```

and lines 295 and 360: `--build-arg "NEMO_RL_SUBMODULE_PINS_B64=$SUBMODULE_PINS_B64"`. Same two edits in `build_nemo_rl_overlay_image.slurm` lines 130 and 239.

- [ ] **Step 5: Import gate exception clause**

`nemo_rl/__init__.py`, the `except Exception as e:` at the end of `_check_container_fingerprint`: add before it

```python
    except RuntimeError:
        raise
```

(it already exists for mismatches; confirm the identity module's `RuntimeError` for a missing uv reaches it: `runpy.run_path` propagates it). No other change.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --group test python -m pytest tests/unit/test_version_check.py tests/unit/distributed/test_actor_environments.py tests/unit/tools/test_image_identity.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tools/generate_fingerprint.py nemo_rl/__init__.py docker/Dockerfile docker/Dockerfile.overlay infra/slurm/cscs/build_nemo_rl_image.slurm infra/slurm/cscs/build_nemo_rl_overlay_image.slurm tests/unit/test_version_check.py tests/unit/distributed/test_actor_environments.py
git commit -s -m "build: stamp the schema 2 identity in-container from host-supplied pins"
```

---

### Task 3: worker readiness marker

**Files:**
- Modify: `nemo_rl/utils/venvs.py:92-117`
- Modify: `tests/unit/tools/test_image_workers.py:45-63, 98-110` (fixture)

**Interfaces:**
- Consumes: `image_identity.export_digest`, `image_identity.resolve_uv`.
- Produces: `_dependency_fingerprint(py_executable)` unchanged signature; the digest is `sha256("py_executable\0" + normalized command + "\0export\0" + export_digest(extras of the command))`.

- [ ] **Step 1: Failing test**

Add to `tests/unit/tools/test_image_workers.py` in the qualification fixture class:

```python
    def test_marker_survives_a_lock_change_with_identical_resolution(self):
        lock = self.source / "uv.lock"
        lock.write_text(lock.read_text() + "\n# reformatted\n")
        namespace = {}
        exec((self.source / "nemo_rl/utils/venvs.py").read_text(), namespace)
        self.assertTrue(namespace["venv_is_current"](self.marker, self.command))
```

The fixture already extracts `_normalized_worker_command`, `_dependency_fingerprint`, and `venv_is_current` from the real module by AST; extend the preamble it writes (line 57-62) with `import importlib.util, re, subprocess` and a `git_root`-relative loader identical to the shim's `_identity_module`, and copy `tools/image_identity.py` and a `docker/Dockerfile` with the fixture's uv version into `self.source` beside the existing copy of `generate_fingerprint.py`.

- [ ] **Step 2: Verify it fails**

Run: `uv run --group test python -m pytest tests/unit/tools/test_image_workers.py -k marker_survives -v`
Expected: FAIL (`venv_is_current` returns False because the raw lock bytes changed).

- [ ] **Step 3: Implement**

Replace lines 92-117 of `nemo_rl/utils/venvs.py`:

```python
def _extras_from_command(py_executable: str) -> tuple[str, ...]:
    tokens = shlex.split(py_executable)
    return tuple(
        tokens[index + 1]
        for index, token in enumerate(tokens[:-1])
        if token == "--extra"
    )


@lru_cache(maxsize=None)
def _image_identity():
    path = Path(git_root) / "tools/image_identity.py"
    spec = importlib.util.spec_from_file_location("image_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _dependency_fingerprint(py_executable: str) -> str:
    """Digest what a worker venv resolves to, not the bytes it resolved from.

    Worker venvs are reused whenever they are marked ready, so the marker has
    to change exactly when the frozen resolution for the command's extras
    changes. The normalized command is part of the digest because it selects
    the extras; two actors can share a lockfile and need different packages.
    """
    identity = _image_identity()
    digest = hashlib.sha256()
    digest.update(b"py_executable\0")
    digest.update(_normalized_worker_command(py_executable).encode())
    digest.update(b"\0export\0")
    digest.update(
        identity.export_digest(
            Path(git_root), _extras_from_command(py_executable),
            identity.resolve_uv(Path(git_root)),
        ).encode()
    )
    return digest.hexdigest()
```

Add `import importlib.util` to the module imports.

- [ ] **Step 4: Verify all image-worker tests pass**

Run: `uv run --group test python -m pytest tests/unit/tools/test_image_workers.py tests/unit/utils/test_venvs.py -v`
Expected: all pass (adjust `test_venvs.py` fixtures that wrote raw-digest markers, if any, to call `_dependency_fingerprint` instead of hand-hashing).

- [ ] **Step 5: Commit**

```bash
git add nemo_rl/utils/venvs.py tests/unit/tools/test_image_workers.py tests/unit/utils/test_venvs.py
git commit -s -m "fix(venvs): key worker readiness on the frozen resolution"
```

---

### Task 4: qualification probe, overlay restamp, hermetic manifest

**Files:**
- Modify: `tools/check_image_workers.py:60-67`
- Modify: `tools/restamp_overlay_fingerprint.py`
- Modify: `docker/Dockerfile.overlay:181-185`
- Modify: `tools/image_build_manifest.py:33-44, 181-199`
- Modify: `tests/unit/tools/test_overlay_fingerprint.py`, `tests/unit/tools/test_image_build_manifest.py`, `tests/unit/tools/test_image_workers.py` (fingerprint file contents)

**Interfaces:**
- Consumes: `image_identity.validate`.
- Produces: `restamp_overlay_fingerprint.py <container_fingerprint>` reads `NEMO_RL_SUBMODULE_PINS_B64`, refuses when any `baked_submodules` entry of the inherited stamp differs from the requested pins, and exits 0 without writing (the Dockerfile then restamps with the shim).

- [ ] **Step 1: Failing tests**

`test_overlay_fingerprint.py`: `BASE` becomes a schema 2 stamp with `baked_submodules = {"3rdparty/kernels": "abc123"}` and `submodule_pins = {"3rdparty/kernels": "abc123", "3rdparty/Gym": "def456"}`. `_run` passes `NEMO_RL_SUBMODULE_PINS_B64` instead of the old variable and only the container path. Cases: same pins → exit 0, files untouched; Gym pin changed → exit 0 (editable, not baked); kernels pin changed → exit non-zero with "baked"; missing kernels pin → non-zero; schema 1 stamp → non-zero with "schema".

`test_image_build_manifest.py`: the fixture's fake `generate_fingerprint.py` prints a schema 2 dict; assert `"submodules"` is not in `manifest["inputs"]` and `inputs["dependency_fingerprint"]["schema"] == 2`.

`test_image_workers.py`: the fixture's container fingerprint is the shim's output (already computed by subprocess at line 253-258); add a case writing `{"schema": 1}` and assert the probe fails with "schema 2".

- [ ] **Step 2: Verify they fail**

Run: `uv run --group test python -m pytest tests/unit/tools/test_overlay_fingerprint.py tests/unit/tools/test_image_build_manifest.py tests/unit/tools/test_image_workers.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`tools/check_image_workers.py` probe, replace lines 60-67:

```python
        built = json.loads(Path(fingerprint_path).read_text())
        identity = runpy.run_path(str(source / "tools/image_identity.py"))
        fingerprint = runpy.run_path(str(source / "tools/generate_fingerprint.py"))[
            "generate_fingerprint"
        ]()
        identity["validate"](built, label="container")
        identity["validate"](fingerprint, label="source")
        if actor not in fingerprint["actors"]:
            raise ValueError(f"Source fingerprint has no environment for {actor}")
        if built != fingerprint:
            raise ValueError("Container fingerprint does not match source fingerprint")
```

`tools/restamp_overlay_fingerprint.py`, whole `main`:

```python
def main() -> None:
    """Refuse an overlay whose baked submodule pins differ from the base image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container_fingerprint", type=Path)
    args = parser.parse_args()
    try:
        requested = json.loads(
            base64.b64decode(os.environ["NEMO_RL_SUBMODULE_PINS_B64"], validate=True)
        )
        inherited = json.loads(args.container_fingerprint.read_bytes())
        if inherited.get("schema") != 2:
            raise ValueError("inherited fingerprint is not schema 2")
        baked = inherited["baked_submodules"]
        differences = [
            f"  {path}: inherited={sha} requested={requested.get(path, 'missing')}"
            for path, sha in sorted(baked.items())
            if requested.get(path) != sha
        ]
        if differences:
            raise ValueError(
                "Inherited baked submodule pins differ from the requested source; "
                "rebuild the base image:\n" + "\n".join(differences)
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Overlay fingerprint validation failed: {error}") from error
```

`docker/Dockerfile.overlay` lines 181-185:

```dockerfile
# Baked pins are inherited from the base image; editable submodules run from
# the checkout. Validate before anything imports nemo_rl, then restamp.
RUN python tools/restamp_overlay_fingerprint.py /opt/nemo_rl_container_fingerprint \
    && python tools/generate_fingerprint.py \
        | tee /opt/nemo_rl_container_fingerprint \
            > /opt/nemo-rl/.nemo_rl_source_fingerprint.json
```

`tools/image_build_manifest.py`: in `generate_manifest`, replace the fingerprint validation (lines 190-198) with a `schema == 2` check plus non-empty `actors`, and delete the `"submodules": recursive_submodules(repo_root)` input line. Delete `recursive_submodules` if nothing else uses it (grep first).

- [ ] **Step 4: Verify they pass**

Run: `uv run --group test python -m pytest tests/unit/tools/test_overlay_fingerprint.py tests/unit/tools/test_image_build_manifest.py tests/unit/tools/test_image_workers.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tools/check_image_workers.py tools/restamp_overlay_fingerprint.py docker/Dockerfile.overlay tools/image_build_manifest.py tests/unit/tools/
git commit -s -m "build: gate qualification, overlay and hermetic cache on the image identity"
```

---

### Task 5: `tools/image_compat.py` verdict tool

**Files:**
- Create: `tools/image_compat.py`
- Create: `tests/unit/tools/test_image_compat.py`

**Interfaces:**
- Consumes: `image_identity.compute`, `image_identity.export_text`, `image_identity.validate`, `uv.lock` via `tomllib`.
- Produces: CLI `python tools/image_compat.py <image_fingerprint.json> [--repo-root .] [--json]`; exit 0 reusable, 10 overlay, 20 rebuild, 1 error. Library `classify(image: dict, source: dict, source_exports: dict[str, str], lock: dict) -> Verdict` where `Verdict = tuple[str, list[str]]` (verdict, reasons).

- [ ] **Step 1: Failing tests**

```python
# tests/unit/tools/test_image_compat.py (fixture setup like test_image_identity)
def test_identical_is_reusable(...)         # exit 0, "reusable"
def test_removed_actor_is_reusable(...)     # image has extra actor key → 0
def test_universal_wheel_bump_is_overlay(...)
    # fixture lock has a pure dep `six`; bump to another version whose
    # lock `wheels` all end in "-none-any.whl" → 10, reason names "six 1.16.0 -> 1.17.0"
def test_platform_wheel_is_rebuild(...)
    # dep whose lock wheels include "manylinux" → 20
def test_baked_pin_change_is_rebuild(...)   # → 20, reason names the submodule
def test_uv_mismatch_is_error(...)          # image "uv" differs → exit 1
def test_schema_1_is_error(...)
```

Use a fixture lock with two registry packages at fixed versions written by hand (a minimal valid `uv.lock` with `[[package]]` tables carrying `wheels = [{ url = "...-py3-none-any.whl", hash = "sha256:..." }]`), so no network is needed; the "source" export text is produced by `export_text` on the fixture and the "image" identity by editing digests.

- [ ] **Step 2: Verify they fail**

Run: `uv run --group test python -m pytest tests/unit/tools/test_image_compat.py -v`
Expected: FAIL, module missing.

- [ ] **Step 3: Implement**

```python
# tools/image_compat.py
"""Say whether a source tree can run on an image: reusable, overlay, or rebuild."""

import argparse
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

EXIT = {"reusable": 0, "overlay": 10, "rebuild": 20}
_REQ = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>\S+)")


def _identity(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        "image_identity", repo_root / "tools/image_identity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_export(text: str) -> dict[str, str]:
    packages = {}
    for line in text.splitlines():
        match = _REQ.match(line)
        if match:
            packages[match["name"].lower()] = match["version"]
        elif line.startswith("-e "):
            packages[line.split()[1]] = "editable"
    return packages


def universal(lock: dict, name: str, version: str) -> bool:
    for package in lock.get("package", []):
        if package["name"].lower() == name and package.get("version") == version:
            wheels = package.get("wheels", [])
            return bool(wheels) and all(
                w["url"].rsplit("/", 1)[-1].endswith("-none-any.whl") for w in wheels
            )
    return False


def classify(image: dict, source: dict, image_exports: dict[str, str],
             source_exports: dict[str, str], lock: dict):
    reasons = []
    if image["uv"] != source["uv"]:
        raise ValueError(f"uv version differs: image {image['uv']}, source {source['uv']}")
    verdict = "reusable"
    for path, sha in source["baked_submodules"].items():
        if image["baked_submodules"].get(path) != sha:
            reasons.append(f"baked submodule {path}: {image['baked_submodules'].get(path)} -> {sha}")
            verdict = "rebuild"
    for actor, digest in source["actors"].items():
        if image["actors"].get(actor) == digest:
            continue
        before = parse_export(image_exports.get(actor, ""))
        after = parse_export(source_exports[actor])
        for name in sorted(set(before) | set(after)):
            old, new = before.get(name), after.get(name)
            if old == new:
                continue
            if new is None:
                reasons.append(f"{actor}: {name} {old} removed")
                continue
            if new == "editable":
                continue
            if universal(lock, name, new):
                reasons.append(f"{actor}: {name} {old} -> {new} (universal wheel)")
                verdict = "overlay" if verdict == "reusable" else verdict
            else:
                reasons.append(f"{actor}: {name} {old} -> {new} (platform wheel or sdist)")
                verdict = "rebuild"
    return verdict, reasons
```

`main` loads the image stamp, requires `image_exports` beside it (the stamp
alone holds digests, not texts, so the committed environment file from Task 6
is a directory `X.fingerprint/` with `identity.json` and `exports/<actor>.txt`;
adjust the Task 6 layout accordingly), computes `source = identity.compute(...)`
and `source_exports` via `export_text` per actor, prints the verdict and
reasons (or `--json`), and exits with `EXIT[verdict]`.

- [ ] **Step 4: Verify they pass**

Run: `uv run --group test python -m pytest tests/unit/tools/test_image_compat.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/image_compat.py tests/unit/tools/test_image_compat.py
git commit -s -m "build: add the image compatibility verdict tool"
```

---

### Task 6: committed image stamp, assemble verification, CI job, docs

**Files:**
- Create: `infra/slurm/cscs/environments/nemo_rl_vllm026_ncclext.fingerprint/` (`identity.json`, `exports/*.txt`; placeholder content replaced in Task 7)
- Modify: `infra/slurm/cscs/assemble_nemo_rl_image.slurm:55-63`
- Modify: `.github/workflows/cicd-main.yml` (new job after `lint-check`, and `HOSTED_CHECKS_SUCCESS`)
- Modify: `infra/slurm/cscs/README.md:89-97`, `docs/design-docs/dependency-management.md:283-300`

- [ ] **Step 1: Assemble verification**

After the existing `/opt/nemo_rl_container_fingerprint > "$PODMAN_STORAGE_BASE/fingerprint.json"` extraction, add:

```bash
"$HOST_PYTHON" - "$PODMAN_STORAGE_BASE/fingerprint.json" "$ENV_FINGERPRINT_DIR/identity.json" <<'PY'
import json, sys
built, committed = (json.load(open(p)) for p in sys.argv[1:])
if built != committed:
    raise SystemExit("Committed environment stamp differs from the image stamp; refresh it")
PY
```

with `ENV_FINGERPRINT_DIR` derived from the environment TOML the launcher is assembling for.

- [ ] **Step 2: CI job**

Add after `lint-check`:

```yaml
  image-compat:
    name: Image compatibility verdict
    needs: [pre-flight]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { submodules: false }
      - name: Install the pinned uv
        run: |
          UV_VERSION=$(sed -n 's/^ARG UV_VERSION=//p' docker/Dockerfile)
          curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"
      - name: Compare against the qualified image
        run: |
          set +e
          python tools/image_compat.py infra/slurm/cscs/environments/nemo_rl_vllm026_ncclext.fingerprint \
            | tee -a "$GITHUB_STEP_SUMMARY"
          code=$?
          echo "verdict exit code: $code" >> "$GITHUB_STEP_SUMMARY"
          test $code -ne 1
```

Submodule pins come from `git submodule status` on a checkout without initialized submodules (gitlinks still resolve), so no submodule fetch is needed. The job fails only on an error, never on `overlay` or `rebuild`; the verdict is information for the PR. Add `needs.image-compat.result == 'success' &&` to `HOSTED_CHECKS_SUCCESS` and `image-compat` to the `needs` list of the quality-check job.

- [ ] **Step 3: Docs**

README overlay section: replace "checks that inherited submodule pins match" with "checks that inherited baked submodule pins match (editable submodules run from the checkout)"; add a paragraph "Deciding whether a sync needs an image" pointing to `tools/image_compat.py` and the committed stamp directory. `dependency-management.md` Container Version Checking: describe schema 2 in five lines and link the spec.

- [ ] **Step 4: Run the whole affected suite**

Run: `uv run --group test python -m pytest tests/unit/tools tests/unit/test_version_check.py tests/unit/distributed/test_actor_environments.py -v`
Expected: all pass.

- [ ] **Step 5: Commit and push, then dispatch the fork gate**

```bash
git add infra/slurm/cscs .github/workflows/cicd-main.yml docs/design-docs/dependency-management.md
git commit -s -m "ci: report the image compatibility verdict on every PR"
git push -u origin design/2026-09-17-image-identity
gh api -X POST repos/Alvorecer721/Nemo-RL/actions/workflows/cicd-main.yml/dispatches \
  -f ref=design/2026-09-17-image-identity -f 'inputs[test_to_run]=Lfast'
```

---

### Task 7 (separate PR): migrate the in-use image and sync upstream

**Files:**
- Modify: `infra/slurm/cscs/environments/nemo_rl_vllm026_ncclext.fingerprint/` (real stamp)
- Modify: `ray.sub:226-229` (mount)
- Sync branch: merge `upstream/main` (`d4b446886`) after #42.

- [ ] **Step 1: Stamp the in-use image once.** Check out the source commit recorded in the image label (`org.opencontainers.image.revision` on `nemo-rl-apertus-vllm-0.26.0-46466aaf4259-5ad7c46f0a6d.sqsh`, read with `podman inspect` or the assemble log), run `python tools/generate_fingerprint.py > identity.json` and `uv export` per actor into `exports/` inside a container of that image (so uv is 0.11.28), commit the directory.
- [ ] **Step 2: Mount it at launch.** In `ray.sub`, when `NRL_IMAGE_FINGERPRINT` is set, append `"$NRL_IMAGE_FINGERPRINT/identity.json:/opt/nemo_rl_container_fingerprint"` to `MOUNTS` the same way the uv cache override is appended. The autoresearch launch scripts export `NRL_IMAGE_FINGERPRINT` from the environment TOML name.
- [ ] **Step 3: Verify on the 8B smoke.** `AP_VARIANT=8b-smoke bash infra/slurm/cscs/autoresearch/submit_apertus_bench.sh` with the design branch mounted: the import gate passes, no worker re-sync beyond the one-time marker refresh, two updates complete.
- [ ] **Step 4: Sync.** Merge `upstream/main` onto main-after-#42, run `python tools/image_compat.py infra/slurm/cscs/environments/nemo_rl_vllm026_ncclext.fingerprint`. Expected verdict: `overlay` with reasons `gitpython 3.1.59 -> 3.1.62 (universal wheel)` for the `vllm`+`nemo_gym` actors and a new `RolloutReassemblerActor` entry. Build the overlay with `build_nemo_rl_overlay_image.slurm`, restamp the environment directory from it, run the 8B smoke, then the 70B two-update gate.

## Self-review

- Spec coverage: contract items 1 to 3 → Task 1; consumers table → Tasks 2, 3, 4; verdict tool → Task 5; committed stamp, assemble check, CI → Task 6; no-backward-compatibility migration → Task 7. Open items stay open.
- Type consistency: `compute` returns the dict with keys `schema, uv, base, actors, baked_submodules, submodule_pins`; `validate` checks the first five; `classify` reads `uv, actors, baked_submodules`; the restamp reads `schema, baked_submodules`; `--pins-only` prints the pin map only.
- Placeholder scan: Task 6 Step 1 leaves `ENV_FINGERPRINT_DIR` derivation to the implementer with the rule stated (TOML basename plus `.fingerprint`); Task 5 `main` is described rather than listed because its shape follows `classify` exactly and the exit table.
