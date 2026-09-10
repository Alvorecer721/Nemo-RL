#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable build-to-assembly handoff; no package or container operations."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TypedDict

from image_build_manifest import Manifest, validate_manifest, verify_manifest


class ReleaseReceipt(TypedDict):
    schema_version: int
    image_ref: str
    source_commit: str
    build_inputs_sha256: str
    vllm_version: str
    hermetic_manifest: Manifest


def source_identity(commit: str, key: str) -> str:
    return hashlib.sha256(f"commit\0{commit}\0hermetic\0{key}\0".encode()).hexdigest()


def create_receipt(
    manifest: Manifest, *, source_commit: str, image_ref: str, vllm_version: str
) -> ReleaseReceipt:
    receipt = ReleaseReceipt(
        schema_version=1,
        image_ref=image_ref,
        source_commit=source_commit,
        build_inputs_sha256=source_identity(source_commit, manifest["cache_key"]),
        vllm_version=vllm_version,
        hermetic_manifest=manifest,
    )
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: ReleaseReceipt) -> None:
    if not isinstance(receipt, dict) or set(receipt) != set(
        ReleaseReceipt.__annotations__
    ):
        raise ValueError("Invalid release receipt fields")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("Unsupported release receipt schema")
    patterns = {
        "image_ref": r"127\.0\.0\.1:5000/nemo-rl-(?:apertus|full)@sha256:[0-9a-f]{64}",
        "source_commit": r"[0-9a-f]{40}",
        "build_inputs_sha256": r"[0-9a-f]{64}",
        "vllm_version": r"[0-9]+\.[0-9]+\.[0-9]+",
    }
    for field, pattern in patterns.items():
        value = receipt[field]
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"Invalid release {field}: {value!r}")
    manifest = receipt["hermetic_manifest"]
    validate_manifest(manifest)
    inputs = manifest["inputs"]
    if inputs.get("platform") != "linux/arm64" or inputs.get("profile") not in {
        "apertus",
        "full",
    }:
        raise ValueError("Unsupported prepared image platform/profile")
    if not receipt["image_ref"].startswith(
        f"127.0.0.1:5000/nemo-rl-{inputs['profile']}@"
    ):
        raise ValueError("Prepared image repository does not match profile")
    fingerprint = inputs.get("dependency_fingerprint")
    required = {
        "pyproject.toml",
        "uv.lock",
        "nemo_rl/distributed/actor_environments.py",
    }
    if (
        not isinstance(fingerprint, dict)
        or not required.issubset(fingerprint)
        or any(
            not isinstance(k, str) or not isinstance(v, str) or not v or v == "missing"
            for k, v in fingerprint.items()
        )
    ):
        raise ValueError("Incomplete dependency fingerprint")
    expected = source_identity(receipt["source_commit"], manifest["cache_key"])
    if receipt["build_inputs_sha256"] != expected:
        raise ValueError("Release source and dependency identity do not match")


def read_receipt(path: Path) -> ReleaseReceipt:
    receipt = json.loads(path.read_text())
    validate_receipt(receipt)
    return receipt


def write_receipt(path: Path, receipt: ReleaseReceipt) -> None:
    """Publish atomically without replacing a different existing handoff."""
    validate_receipt(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
        temporary.chmod(0o644)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_receipt(path) != receipt:
                raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def image_basename(receipt: ReleaseReceipt) -> str:
    validate_receipt(receipt)
    source_id = f"{receipt['source_commit'][:12]}-{receipt['build_inputs_sha256'][:12]}"
    profile = receipt["hermetic_manifest"]["inputs"]["profile"]
    return f"nemo-rl-{profile}-vllm-{receipt['vllm_version']}-{source_id}"


def verify_image(
    receipt: ReleaseReceipt,
    inspection: list[dict],
    embedded_manifest: Manifest,
    runtime_fingerprint: dict[str, str],
) -> str:
    validate_receipt(receipt)
    if not isinstance(inspection, list) or len(inspection) != 1:
        raise ValueError("Expected exactly one prepared image")
    info = inspection[0]
    if (info.get("Os"), info.get("Architecture")) != ("linux", "arm64"):
        raise ValueError("Prepared image platform does not match receipt")
    if receipt["image_ref"] not in info.get("RepoDigests", []):
        raise ValueError("Prepared image digest does not match receipt")
    labels = info.get("Labels") or {}
    for label, expected in {
        "org.opencontainers.image.revision": receipt["source_commit"],
        "org.opencontainers.image.source-inputs": receipt["build_inputs_sha256"],
    }.items():
        if labels.get(label) != expected:
            raise ValueError(f"Prepared image label does not match receipt: {label}")
    verify_manifest(receipt["hermetic_manifest"], embedded_manifest)
    if (
        runtime_fingerprint
        != receipt["hermetic_manifest"]["inputs"]["dependency_fingerprint"]
    ):
        raise ValueError("Prepared image runtime fingerprint does not match receipt")

    image_id = info.get("Id")
    if not isinstance(image_id, str) or not re.fullmatch(r"[0-9a-f]{64}", image_id):
        raise ValueError("Prepared image has an invalid local image ID")
    return image_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--image-ref", required=True)
    create.add_argument("--vllm-version", required=True)
    create.add_argument("--output", type=Path, required=True)
    fields = sub.add_parser("fields")
    fields.add_argument("receipt", type=Path)
    verify = sub.add_parser("verify-image")
    verify.add_argument("receipt", type=Path)
    for name in ("inspection", "manifest", "fingerprint"):
        verify.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "create":
        result = create_receipt(
            json.loads(args.manifest.read_text()),
            source_commit=args.source_commit,
            image_ref=args.image_ref,
            vllm_version=args.vllm_version,
        )
        write_receipt(args.output, result)
        print(args.output)
    elif args.action == "fields":
        result = read_receipt(args.receipt)
        print(result["image_ref"])
        print(image_basename(result))
    else:
        image_id = verify_image(
            read_receipt(args.receipt),
            json.loads(args.inspection.read_text()),
            json.loads(args.manifest.read_text()),
            json.loads(args.fingerprint.read_text()),
        )
        print(image_id)


if __name__ == "__main__":
    main()
