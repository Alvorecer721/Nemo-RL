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

"""Generate and validate stdlib-only, content-addressed hermetic build manifests.

The key excludes HEAD and release-stage Docker instructions. Dependency metadata,
recursive submodule pins, dependency-stage scripts, selected actors and explicit
build arguments determine reuse. Runtime fingerprints remain a separate format.
The builder embeds this JSON at /opt/nemo-rl-hermetic-manifest.json.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict


class BuildInputs(TypedDict):
    """Inputs that determine the reusable dependency image."""

    dependency_fingerprint: dict[str, str]
    submodules: dict[str, str]
    files_sha256: dict[str, str]
    hermetic_recipe_sha256: str
    base_image: str
    platform: str
    profile: str
    build_args: dict[str, str]
    actor_rows: list[str]


class Manifest(TypedDict):
    """Self-validating versioned cache record embedded in the image."""

    schema_version: int
    cache_key: str
    inputs: BuildInputs


def canonical_json(value: object) -> str:
    """Serialize JSON independently of dictionary insertion order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_digest(inputs: BuildInputs) -> str:
    """Hash the canonical input document, with its schema domain."""
    payload = {"schema_version": 1, "inputs": inputs}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def git_output(repo_root: Path, *arguments: str) -> str:
    """Run a required Git query, including in linked worktrees."""
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def recursive_submodules(repo_root: Path) -> dict[str, str]:
    """Read initialized pins; refuse missing, conflicting or moved submodules."""
    pins = {}
    for line in git_output(
        repo_root, "submodule", "status", "--recursive"
    ).splitlines():
        if not line or line[0] != " ":
            raise ValueError(
                f"Submodule is not initialized at its recorded pin: {line}"
            )
        commit, path, *_ = line[1:].split()
        pins[path] = commit
    return pins


def dependency_recipe(repo_root: Path) -> str:
    """Extract all foundation and hermetic instructions before release assembly."""
    recipe = (repo_root / "docker/Dockerfile").read_text()
    boundary = re.search(
        r"^FROM\s+hermetic\s+AS\s+release(?:-core)?\s*$",
        recipe,
        re.MULTILINE | re.IGNORECASE,
    )
    if boundary is None:
        raise ValueError(
            "Dockerfile must have a FROM hermetic AS release[-core] boundary"
        )
    return recipe[: boundary.start()]


def dependency_files(repo_root: Path, recipe: str) -> dict[str, str]:
    """Hash build scripts and metadata, excluding research application sources.

    Workspace packages are installed from pinned submodules or their own package
    descriptors. Top-level tracked 3rdparty files also include build backends.
    Literal Python/shell inputs in the dependency recipe are included, so adding
    another build helper does not require maintaining a parallel checksum list.
    """
    names = {
        "pyproject.toml",
        "uv.lock",
        "tools/generate_fingerprint.py",
        "nemo_rl/distributed/actor_environments.py",
    }
    instructions = "\n".join(
        line for line in recipe.splitlines() if not line.lstrip().startswith("#")
    )
    names.update(
        re.findall(
            r"(?<![\w/])(?:tools|docker|nemo_rl)/[\w./-]+\.(?:py|sh)\b", instructions
        )
    )
    for name in git_output(repo_root, "ls-files", "-z").split("\0"):
        path = Path(name)
        if (
            name == ".gitmodules"
            or (
                name.startswith("research/")
                and path.name in {"pyproject.toml", "uv.lock", "setup.py", "setup.cfg"}
            )
            or (name.startswith("3rdparty/") and (repo_root / path).is_file())
        ):
            names.add(name)
    hashes = {}
    for name in sorted(names):
        path = repo_root / name
        if not path.is_file():
            raise ValueError(f"Dependency build input is missing: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def generate_manifest(
    repo_root: Path,
    *,
    base_image: str,
    platform: str,
    profile: str,
    build_args: dict[str, str],
) -> Manifest:
    """Generate a stable manifest from the checked-out build inputs."""
    if not re.search(r"@sha256:[0-9a-f]{64}$", base_image):
        raise ValueError("BASE_IMAGE must be pinned with an immutable @sha256 digest")
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", profile):
        raise ValueError(f"Invalid image profile label: {profile}")
    if platform not in {"linux/arm64", "linux/amd64"}:
        raise ValueError(f"Unsupported platform: {platform}")
    recipe = dependency_recipe(repo_root)
    fingerprint = json.loads(
        subprocess.run(
            [sys.executable, str(repo_root / "tools/generate_fingerprint.py")],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if (
        not isinstance(fingerprint, dict)
        or not fingerprint
        or any(
            not isinstance(value, str) or value == "missing"
            for value in fingerprint.values()
        )
    ):
        raise ValueError("Dependency fingerprint is incomplete")
    skip_extras = [
        extra
        for extra in ("vllm", "sglang", "trtllm")
        if build_args.get(f"SKIP_{extra.upper()}_BUILD")
    ]
    actor_output = subprocess.run(
        [
            sys.executable,
            str(repo_root / "nemo_rl/distributed/actor_environments.py"),
            "--actors",
            build_args.get("NRL_ACTORS", ""),
            "all",
            *skip_extras,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    actor_rows = actor_output.splitlines()
    if not actor_rows or any(len(row.split("\t")) != 3 for row in actor_rows):
        raise ValueError("Actor manifest must contain nonempty three-field TSV records")
    inputs: BuildInputs = {
        "dependency_fingerprint": fingerprint,
        "submodules": recursive_submodules(repo_root),
        "files_sha256": dependency_files(repo_root, recipe),
        "hermetic_recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
        "base_image": base_image,
        "platform": platform,
        "profile": profile,
        "build_args": dict(sorted(build_args.items())),
        "actor_rows": actor_rows,
    }
    return {"schema_version": 1, "cache_key": manifest_digest(inputs), "inputs": inputs}


def validate_manifest(manifest: Manifest) -> None:
    """Reject malformed documents and forged or stale self-digests."""
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "cache_key",
        "inputs",
    }:
        raise ValueError("Invalid hermetic manifest fields")
    if manifest["schema_version"] != 1 or not isinstance(manifest["inputs"], dict):
        raise ValueError("Unsupported hermetic manifest schema")
    if manifest["cache_key"] != manifest_digest(manifest["inputs"]):
        raise ValueError("Hermetic manifest digest does not match its inputs")


def verify_manifest(expected: Manifest, embedded: Manifest) -> None:
    """Accept reuse only for an intact manifest with exactly the expected inputs."""
    validate_manifest(expected)
    validate_manifest(embedded)
    if expected != embedded:
        raise ValueError(
            "Embedded hermetic manifest does not match current build inputs"
        )


def write_manifest(destination: Path, manifest: Manifest) -> None:
    """Atomically replace a JSON manifest, retaining the previous file on error."""
    validate_manifest(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(canonical_json(manifest) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def select_cache(requested: str, current_key: str, *, available: bool) -> str:
    """Select one allocation's target; explicit pins must match generated inputs."""
    if requested == "rebuild":
        return "hermetic"
    if requested != "auto":
        if not re.fullmatch(r"[0-9a-f]{64}", requested):
            raise ValueError(
                "HERMETIC_CACHE_TAG must be auto, rebuild, or a 64-character digest"
            )
        if requested != current_key:
            raise ValueError(
                "Explicit hermetic cache digest does not match current build inputs"
            )
        if not available:
            raise ValueError(
                "Explicit hermetic cache digest is absent from the registry"
            )
    return "release-core" if available else "hermetic"


def read_manifest(path: Path) -> Manifest:
    """Read and validate a manifest supplied to a CLI operation."""
    manifest = json.loads(path.read_text())
    validate_manifest(manifest)
    return manifest


def main() -> None:
    """Generate, inspect, verify or select a cache using JSON files."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    generate.add_argument("--base-image", required=True)
    generate.add_argument("--platform", default="linux/arm64")
    generate.add_argument("--profile", default="full", help="image profile label")
    generate.add_argument("--build-arg", action="append", default=[])
    generate.add_argument("--output", required=True, type=Path)
    digest = commands.add_parser("digest")
    digest.add_argument("manifest", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("expected", type=Path)
    verify.add_argument("embedded", type=Path)
    select = commands.add_parser("select")
    select.add_argument("manifest", type=Path)
    select.add_argument("--cache-tag", required=True)
    select.add_argument("--available", required=True, choices=["yes", "no"])
    options = parser.parse_args()
    try:
        if options.command == "generate":
            build_args = {}
            for assignment in options.build_arg:
                name, separator, value = assignment.partition("=")
                if (
                    not separator
                    or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
                    or name in build_args
                ):
                    raise ValueError(
                        f"Invalid or duplicate build argument: {assignment}"
                    )
                build_args[name] = value
            manifest = generate_manifest(
                options.repo_root.resolve(),
                base_image=options.base_image,
                platform=options.platform,
                profile=options.profile,
                build_args=build_args,
            )
            write_manifest(options.output, manifest)
            print(manifest["cache_key"])
        elif options.command == "digest":
            print(read_manifest(options.manifest)["cache_key"])
        elif options.command == "verify":
            verify_manifest(
                read_manifest(options.expected), read_manifest(options.embedded)
            )
        else:
            print(
                select_cache(
                    options.cache_tag,
                    read_manifest(options.manifest)["cache_key"],
                    available=options.available == "yes",
                )
            )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"image build manifest: {error}\n")


if __name__ == "__main__":
    main()
