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
"""Tests for OmniPreferenceDataset (manifest contract) and the splice processor.

Manifest-contract tests are tokenizer-free. Splice tests use the real tokenizer
(session-scoped fixture in conftest.py; ~200 s to load, once).
"""

import hashlib
import json

import pytest
import torch

from nemo_rl_apertus.omni_preference import (
    IMAGE_TOKEN_ID,
    OmniPreferenceDataset,
    build_processed_datasets,
    omni_preference_preprocessor,
)
from nemo_rl_apertus.tests.toy_store import build_toy_dataset

# ---------------------------------------------------------------------------
# manifest contract (no tokenizer needed)
# ---------------------------------------------------------------------------


def _make_dataset(root, **kwargs):
    return OmniPreferenceDataset(root, split="train", **kwargs)


def test_missing_manifest_is_unpublished_garbage(tmp_path):
    root = tmp_path / "ds"
    build_toy_dataset(root)
    (root / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest.json"):
        _make_dataset(root)


def test_unknown_token_dtype_refused(tmp_path):
    root = tmp_path / "ds"
    build_toy_dataset(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["token_dtype"] = "<i8"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="token_dtype"):
        _make_dataset(root)


def test_manifest_file_size_mismatch_names_the_file(tmp_path):
    root = tmp_path / "ds"
    build_toy_dataset(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rel = "media/tokens.000000.bin"
    manifest["files"] = {rel: (root / rel).stat().st_size + 1}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tokens.000000.bin"):
        _make_dataset(root)


def test_manifest_missing_file_names_the_file(tmp_path):
    root = tmp_path / "ds"
    build_toy_dataset(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"] = {"media/tokens.000099.bin": 123}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tokens.000099.bin"):
        _make_dataset(root)


def test_tokenizer_sha_checked_when_path_given(tmp_path):
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text('{"toy": true}')
    sha = hashlib.sha256((tok_dir / "tokenizer.json").read_bytes()).hexdigest()

    good = tmp_path / "good"
    build_toy_dataset(good, tokenizer_sha=sha)
    ds = _make_dataset(good, tokenizer_path=tok_dir)
    assert ds.manifest["tokenizer"]["sha256"] == sha

    bad = tmp_path / "bad"
    build_toy_dataset(bad, tokenizer_sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        _make_dataset(bad, tokenizer_path=tok_dir)


def test_non_fork_start_method_refused(tmp_path, monkeypatch):
    import nemo_rl_apertus.omni_preference as op

    root = tmp_path / "ds"
    build_toy_dataset(root)
    monkeypatch.setattr(op.multiprocessing, "get_start_method", lambda: "spawn")
    with pytest.raises(RuntimeError, match="fork"):
        _make_dataset(root)


# ---------------------------------------------------------------------------
# splice processor (real tokenizer)
# ---------------------------------------------------------------------------


@pytest.fixture()
def toy(tmp_path):
    meta = build_toy_dataset(tmp_path / "ds")
    return tmp_path / "ds", meta


def _process(ds, tokenizer, i, max_seq_length=4096):
    return omni_preference_preprocessor(
        ds[i], ds.task_spec, tokenizer, max_seq_length, i, media=ds.media
    )


def _contains_contiguous(haystack: list[int], needle: list[int]) -> bool:
    return any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def test_splice_replaces_marker_with_block(tokenizer, toy):
    root, meta = toy
    ds = _make_dataset(root)
    datum = _process(ds, tokenizer, 0)
    flat = torch.cat([m["token_ids"] for m in datum["message_log_chosen"]])
    assert int((flat == IMAGE_TOKEN_ID).sum()) == 0  # marker consumed
    assert flat.dtype == torch.int64
    # block appears contiguously in the chosen sequence
    assert _contains_contiguous(flat.tolist(), meta["blocks"]["img0"])
    # PreferenceDatumSpec contract: keys + lengths match downstream expectations
    assert datum["length_chosen"] == len(flat)
    assert datum["length_rejected"] == sum(
        len(m["token_ids"]) for m in datum["message_log_rejected"]
    )
    assert datum["loss_multiplier"] == 1.0
    assert datum["idx"] == 0


def test_multi_image_marker_order(tokenizer, toy):
    root, meta = toy
    ds = _make_dataset(root)
    datum = _process(ds, tokenizer, 1)
    flat = torch.cat([m["token_ids"] for m in datum["message_log_chosen"]]).tolist()
    b1, b2 = meta["blocks"]["img1"], meta["blocks"]["img2"]
    i1 = next(i for i in range(len(flat)) if flat[i : i + len(b1)] == b1)
    i2 = next(i for i in range(len(flat)) if flat[i : i + len(b2)] == b2)
    assert i1 < i2  # blocks land in marker order


def test_per_field_refs_side_blocks_stay_on_their_side(tokenizer, toy):
    """Row 2 has an image marker in the CHOSEN response (chosen_media_refs).

    The chosen sequence must splice it; the rejected sequence must not consume
    chosen's refs (per-field contract, spec view schema).
    """
    root, meta = toy
    ds = _make_dataset(root)
    datum = _process(ds, tokenizer, 2)
    chosen = torch.cat([m["token_ids"] for m in datum["message_log_chosen"]]).tolist()
    rejected = torch.cat(
        [m["token_ids"] for m in datum["message_log_rejected"]]
    ).tolist()
    assert _contains_contiguous(chosen, meta["blocks"]["img0"])  # prompt block
    assert _contains_contiguous(chosen, meta["blocks"]["img1"])  # chosen-side block
    assert _contains_contiguous(rejected, meta["blocks"]["img0"])  # prompt block
    assert not _contains_contiguous(rejected, meta["blocks"]["img1"])


def test_marker_ref_count_mismatch_raises(tokenizer, toy):
    root, meta = toy
    ds = _make_dataset(root)
    datum_dict = dict(ds[0])
    # two refs, but the prompt carries a single marker
    datum_dict["prompt_media_refs"] = [meta["ids"]["img0"], meta["ids"]["img1"]]
    with pytest.raises(ValueError, match="marker"):
        omni_preference_preprocessor(
            datum_dict, ds.task_spec, tokenizer, 4096, 0, media=ds.media
        )


def test_overlength_dies_small(tokenizer, toy):
    root, _ = toy
    ds = _make_dataset(root)
    datum = _process(ds, tokenizer, 1, max_seq_length=64)
    assert datum["loss_multiplier"] == 0.0
    for side in ("chosen", "rejected"):
        total = sum(len(m["token_ids"]) for m in datum[f"message_log_{side}"])
        assert total <= 64  # shrunk placeholder, not full length
        assert datum[f"length_{side}"] == total


def test_build_processed_datasets_end_to_end(tokenizer, toy):
    """Exercise the seam dpo.setup() consumes.

    AllTaskProcessedDataset invokes the partial-bound processor positionally as
    (entry, spec, tokenizer, max_seq_length, idx).
    """
    root, _ = toy
    train, val = build_processed_datasets(root, tokenizer, max_seq_length=4096)
    assert len(train) == 3 and len(val) == 1
    assert train.dataset.manifest["expected_min_model_vocab"] == 266440
    datum = train[0]
    assert set(datum) >= {
        "message_log_chosen",
        "message_log_rejected",
        "length_chosen",
        "length_rejected",
        "loss_multiplier",
        "idx",
    }
    for m in datum["message_log_chosen"]:
        assert m["token_ids"].dtype == torch.int64
