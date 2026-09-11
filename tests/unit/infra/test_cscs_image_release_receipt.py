# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable handoff validation, independent of GPUs and container services."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"


@pytest.fixture
def receipt_tool(monkeypatch):
    path = TOOLS.parent / "infra/slurm/cscs/image_release_receipt.py"
    assert path.is_file(), "The build must publish an immutable assembly receipt"
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("image_release_receipt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def receipt(receipt_tool):
    from image_build_manifest import manifest_digest

    inputs = {
        "platform": "linux/arm64",
        "profile": "apertus",
        "dependency_fingerprint": {
            key: "a" * 32
            for key in (
                "pyproject.toml",
                "uv.lock",
                "nemo_rl/distributed/actor_environments.py",
            )
        },
    }
    manifest = {
        "schema_version": 1,
        "inputs": inputs,
        "cache_key": manifest_digest(inputs),
    }
    return receipt_tool.create_receipt(
        manifest,
        source_commit="b" * 40,
        image_ref="127.0.0.1:5000/nemo-rl-apertus@sha256:" + "c" * 64,
        vllm_version="0.26.0",
    )


def image_metadata(receipt):
    return [
        {
            "Id": "d" * 64,
            "Os": "linux",
            "Architecture": "arm64",
            "RepoDigests": [receipt["image_ref"]],
            "Labels": {
                "org.opencontainers.image.revision": receipt["source_commit"],
                "org.opencontainers.image.source-inputs": receipt[
                    "build_inputs_sha256"
                ],
            },
        }
    ]


def test_receipt_roundtrip_and_no_overwrite(receipt_tool, receipt, tmp_path):
    path = tmp_path / "receipt.json"
    receipt_tool.write_receipt(path, receipt)
    assert receipt_tool.read_receipt(path) == receipt
    assert path.stat().st_mode & 0o777 == 0o644
    receipt_tool.write_receipt(path, receipt)
    different = copy.deepcopy(receipt)
    different["image_ref"] = different["image_ref"][:-64] + "d" * 64
    with pytest.raises(FileExistsError):
        receipt_tool.write_receipt(path, different)
    assert receipt_tool.read_receipt(path) == receipt


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_ref", "127.0.0.1:5000/nemo-rl-apertus:latest"),
        ("image_ref", "127.0.0.1:5000/nemo-rl-apertus@sha256:abc"),
        ("source_commit", "main"),
        ("build_inputs_sha256", "d" * 64),
        ("schema_version", 2),
        ("vllm_version", "../../escape"),
    ],
)
def test_invalid_handoff_is_rejected(receipt_tool, receipt, field, value):
    receipt[field] = value
    with pytest.raises(ValueError):
        receipt_tool.validate_receipt(receipt)


def test_dependency_manifest_corruption_is_rejected(receipt_tool, receipt):
    receipt["hermetic_manifest"]["inputs"]["profile"] = "full"
    with pytest.raises(ValueError, match="digest"):
        receipt_tool.validate_receipt(receipt)


def test_prepared_image_metadata_matches(receipt_tool, receipt):
    manifest = receipt["hermetic_manifest"]
    fingerprint = manifest["inputs"]["dependency_fingerprint"]
    receipt_tool.verify_image(receipt, image_metadata(receipt), manifest, fingerprint)
    for key, value in [
        ("Architecture", "amd64"),
        ("RepoDigests", []),
        ("Labels", {"org.opencontainers.image.revision": "wrong"}),
    ]:
        inspect = image_metadata(receipt)
        inspect[0][key] = value
        with pytest.raises(ValueError):
            receipt_tool.verify_image(receipt, inspect, manifest, fingerprint)
    with pytest.raises(ValueError, match="fingerprint"):
        receipt_tool.verify_image(receipt, image_metadata(receipt), manifest, {})


def test_missing_and_partial_receipts_fail(receipt_tool, tmp_path):
    path = tmp_path / "receipt.json"
    with pytest.raises(FileNotFoundError):
        receipt_tool.read_receipt(path)
    path.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError):
        receipt_tool.read_receipt(path)
