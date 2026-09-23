#!/usr/bin/env python3
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

"""Import every selected image worker in its own interpreter, offline.

CPU builds validate locked dependencies, fingerprints and actor imports. Native
GPU imports and a CUDA tensor operation require a separate ``--require-gpu`` run.
Distributed generation/refit/resume qualification remains a separate test.
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

CONTAINER_FINGERPRINT_PATH = Path("/opt/nemo_rl_container_fingerprint")

PROBE = r"""
import importlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

actor, expected_prefix, extras, fingerprint_path, require_gpu, timeout = sys.argv[1:]
require_gpu = require_gpu == "1"
extras = extras.split()
fingerprint = None
locked_dependencies = False
deferred = []
if Path(sys.prefix).resolve() != Path(expected_prefix).resolve():
    raise RuntimeError(f"Wrong interpreter prefix: {sys.prefix}; expected {expected_prefix}")
if actor.startswith("nemo_rl."):
    # The normal import-time gate deliberately permits non-container development
    # and missing fingerprints. Image qualification must fail closed in both cases.
    spec = importlib.util.find_spec("nemo_rl")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate NeMo-RL source for fingerprint validation")
    source = Path(spec.origin).resolve().parent.parent
    try:
        built = json.loads(Path(fingerprint_path).read_text())
        fingerprint = runpy.run_path(str(source / "tools/generate_fingerprint.py"))[
            "generate_fingerprint"
        ]()
        required = {"pyproject.toml", "uv.lock", "nemo_rl/distributed/actor_environments.py"}
        for label, value in (("container", built), ("source", fingerprint)):
            if (not isinstance(value, dict) or not required.issubset(value)
                or any(not isinstance(k, str) or not isinstance(v, str)
                       or not v or v == "missing" for k, v in value.items())):
                raise ValueError(f"Invalid or incomplete {label} fingerprint")
        if built != fingerprint:
            raise ValueError("Container fingerprint does not match source fingerprint")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Fingerprint validation failed: {error}") from error

    from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
    from nemo_rl.utils.venvs import venv_is_current
    marker = Path(expected_prefix) / "NEMO_RL_VENV_READY"
    if not venv_is_current(marker, get_actor_python_env(actor)):
        raise RuntimeError(f"Missing or stale worker readiness marker: {marker}")

    command = [sys.executable, str(source / "nemo_rl/utils/venv_inventory.py"), "verify"]
    for extra in extras:
        command.extend(["--extra", extra])
    checked = subprocess.run(command, capture_output=True, text=True, timeout=int(timeout))
    if checked.returncode:
        raise RuntimeError("Locked dependency check failed: " + (checked.stdout + checked.stderr)[-6000:])
    locked_dependencies = True

# Top-level specs catch a missing deferred backend without importing its GPU
# initialization path. The inventory above verifies the frozen installation.
backend_modules = {
    "vllm": "vllm", "sglang": "sglang", "mcore": "megatron",
    "automodel": "nemo_automodel", "fsdp": "torch",
    "trtllm": "tensorrt_llm", "modelopt": "modelopt", "nemo_gym": "nemo_gym",
}
for extra in extras:
    backend = backend_modules.get(extra)
    if backend and importlib.util.find_spec(backend) is None:
        raise RuntimeError(f"Missing backend module {backend} for extra {extra}")
if "vllm" in extras:
    from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat
    ensure_vllm_source_compat()
cpu_imports = {
    "vllm": ("vllm",), "sglang": ("sglang",),
    "mcore": ("megatron.core", "megatron.bridge"),
    "automodel": ("nemo_automodel",), "fsdp": ("torch.distributed.tensor",),
    "modelopt": ("modelopt",), "nemo_gym": ("nemo_gym",),
}
native_imports = {
    "vllm": ("vllm._C_stable_libtorch",), "sglang": ("sgl_kernel",),
    "mcore": ("transformer_engine.pytorch",),
    "trtllm": ("tensorrt_llm", "tensorrt_llm.bindings", "tensorrt_llm.llmapi.llm_args"),
}
for extra in extras:
    for backend in cpu_imports.get(extra, ()):
        importlib.import_module(backend)
native = sorted({backend for extra in extras for backend in native_imports.get(extra, ())})
if require_gpu:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU qualification requires an available CUDA device")
    for backend in native:
        importlib.import_module(backend)
    if torch.ones(1, device="cuda").sum().item() != 1:
        raise RuntimeError("CUDA tensor operation returned an unexpected result")
    torch.cuda.synchronize()
else:
    deferred = [*native, "CUDA tensor operation"]
module, name = actor.rsplit(".", 1)
getattr(importlib.import_module(module), name)
versions = {}
for package in ("torch", "vllm", "sglang", "transformer-engine", "megatron-core",
                "megatron-bridge", "tensorrt-llm", "nemo-automodel", "nvidia-nccl-cu13"):
    try:
        versions[package] = version(package)
    except PackageNotFoundError:
        pass
print("NRL_WORKER_CHECK=" + json.dumps({
    "actor": actor, "prefix": sys.prefix, "python": sys.version,
    "offline": os.environ.get("UV_OFFLINE"), "versions": versions, "passed": True,
    "fingerprint": fingerprint, "locked_dependencies": locked_dependencies,
    "gpu_qualified": require_gpu, "deferred_checks": deferred,
}))
"""


def read_manifest(path: Path) -> list[tuple[str, str]]:
    workers = []
    seen = set()
    with path.open(newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) != 3:
                raise ValueError(f"Malformed worker manifest row: {row!r}")
            actor, stage, arguments = row
            if "." not in actor or not all(p.isidentifier() for p in actor.split(".")):
                raise ValueError(f"Invalid actor name: {actor!r}")
            if actor in seen or stage not in {"deps", "trtllm"}:
                raise ValueError(f"Duplicate actor or invalid stage: {row!r}")
            seen.add(actor)
            words = shlex.split(arguments)
            if len(words) % 2 or any(
                words[i] != "--extra" for i in range(0, len(words), 2)
            ):
                raise ValueError(f"Invalid extras: {arguments!r}")
            workers.append((actor, " ".join(words[1::2])))
    if not workers:
        raise ValueError("Worker manifest is empty")
    return workers


def check_worker(
    actor: str, extras: str, root: Path, timeout: int, *, require_gpu: bool = False
) -> dict:
    prefix = root / actor
    python = prefix / "bin/python"
    environment = dict(os.environ)
    for variable in (
        "NRL_IGNORE_VERSION_MISMATCH",
        "NRL_FORCE_REBUILD_VENVS",
        "NEMO_RL_PY_EXECUTABLES_SYSTEM",
    ):
        environment.pop(variable, None)
    environment.update(
        UV_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VIRTUAL_ENV=str(prefix),
        UV_PROJECT_ENVIRONMENT=str(prefix),
        PATH=f"{prefix / 'bin'}:{environment.get('PATH', '')}",
    )
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                PROBE,
                actor,
                str(prefix),
                extras,
                str(CONTAINER_FINGERPRINT_PATH),
                "1" if require_gpu else "0",
                str(timeout),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(f"exit {result.returncode}: {result.stderr[-6000:]}")
        records = [
            line.removeprefix("NRL_WORKER_CHECK=")
            for line in result.stdout.splitlines()
            if line.startswith("NRL_WORKER_CHECK=")
        ]
        if len(records) != 1:
            raise RuntimeError("Worker did not return exactly one qualification record")
        return json.loads(records[0])
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        return {
            "actor": actor,
            "passed": False,
            "gpu_qualified": False,
            "error": str(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/opt/actor_venvs.tsv"))
    parser.add_argument("--venv-root", type=Path, default=Path("/opt/ray_venvs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="require backend native imports and a CUDA tensor operation",
    )
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    workers = read_manifest(args.manifest)
    reports = [
        check_worker(
            actor, extras, args.venv_root, args.timeout, require_gpu=args.require_gpu
        )
        for actor, extras in workers
    ]
    report = {
        "passed": all(item["passed"] for item in reports),
        "gpu_qualified": all(item["gpu_qualified"] for item in reports),
        "scope": "locked dependencies, actor imports, and optional CUDA/native smoke check",
        "workers": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=args.output.parent, delete=False
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, args.output)
    for item in reports:
        print(f"{item['actor']}: {'PASS' if item['passed'] else item['error']}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
