# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect drift from a successfully frozen image-worker installation."""

import importlib.util
from importlib.metadata import PathDistribution
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[3] / "nemo_rl/utils/venv_inventory.py"


@pytest.fixture
def inventory():
    assert MODULE.is_file(), "Worker inventory helper is missing"
    spec = importlib.util.spec_from_file_location("venv_inventory", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dist(tmp_path, suffix="", name="fixture"):
    info = tmp_path / f"fixture{suffix}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
    (info / "WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-any\n")
    (info / "RECORD").write_text("fixture.py,sha256=abc,12\n")
    return info, PathDistribution(info)


def test_inventory_detects_metadata_provenance_and_selection_drift(
    inventory, tmp_path, monkeypatch
):
    info, dist = make_dist(tmp_path)
    monkeypatch.setattr(inventory, "distributions", lambda: [dist])
    before = inventory.snapshot(["vllm"])
    assert before == inventory.snapshot(["vllm"])
    assert before != inventory.snapshot(["mcore"])
    (info / "direct_url.json").write_text('{"url": "file:///other-source"}')
    assert before != inventory.snapshot(["vllm"])


def test_inventory_receipt_is_readable_and_rejects_later_drift(
    inventory, tmp_path, monkeypatch
):
    info, dist = make_dist(tmp_path)
    monkeypatch.setattr(inventory, "distributions", lambda: [dist])
    monkeypatch.setattr(inventory.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        inventory.sys, "argv", ["inventory", "record", "--extra", "vllm"]
    )
    inventory.main()
    receipt = tmp_path / inventory.INVENTORY_FILENAME
    assert receipt.stat().st_mode & 0o777 == 0o644
    monkeypatch.setattr(
        inventory.sys, "argv", ["inventory", "verify", "--extra", "vllm"]
    )
    inventory.main()
    (info / "METADATA").write_text("Name: fixture\nVersion: 2.0\n")
    with pytest.raises(RuntimeError, match="differs"):
        inventory.main()


@pytest.mark.parametrize("filename", ["METADATA", "WHEEL", "RECORD"])
def test_inventory_rejects_missing_metadata(inventory, tmp_path, monkeypatch, filename):
    info, dist = make_dist(tmp_path)
    (info / filename).unlink()
    monkeypatch.setattr(inventory, "distributions", lambda: [dist])
    with pytest.raises(ValueError):
        inventory.snapshot(["vllm"])


def test_inventory_rejects_duplicate_distributions(inventory, tmp_path, monkeypatch):
    _, first = make_dist(tmp_path)
    _, second = make_dist(tmp_path, "-duplicate")
    monkeypatch.setattr(inventory, "distributions", lambda: [first, second])
    with pytest.raises(ValueError, match="Duplicate"):
        inventory.snapshot(["vllm"])


def test_inventory_excludes_only_project_source(inventory, tmp_path, monkeypatch):
    _, dependency = make_dist(tmp_path)
    project, editable = make_dist(tmp_path, "-project", "nemo-rl")
    monkeypatch.setattr(inventory, "distributions", lambda: [dependency, editable])
    before = inventory.snapshot(["vllm"])
    (project / "METADATA").write_text("Name: nemo-rl\nVersion: 2.0\n")
    assert before == inventory.snapshot(["vllm"])
