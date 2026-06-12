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
"""Tests for the on-disk media-store format and the copied MediaStoreReader.

The golden-store test is the cross-repo sync contract: the fixture under
``fixtures/golden_store/`` is generated with the producer's real
``MediaStoreWriter`` (vision_tokenization, Part A) and must be regenerated on
any ``schema_version`` bump. Until Part A lands, it is skipped.
"""

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from nemo_rl_apertus.tests.toy_store import build_toy_dataset

GOLDEN_STORE = Path(__file__).resolve().parent / "fixtures" / "golden_store"


def test_reader_roundtrip_tokens_and_raw(tmp_path):
    from nemo_rl_apertus.media_store import MediaStoreReader

    meta = build_toy_dataset(tmp_path / "ds")
    reader = MediaStoreReader([tmp_path / "ds" / "media"])
    for key in ("img0", "img1", "img2"):
        np.testing.assert_array_equal(
            reader.tokens(meta["ids"][key]), meta["blocks"][key]
        )
        assert reader.raw(meta["ids"][key]) == key.encode()


def test_reader_memmap_path_equals_ram_path(tmp_path):
    from nemo_rl_apertus.media_store import MediaStoreReader

    meta = build_toy_dataset(tmp_path / "ds")
    ram = MediaStoreReader([tmp_path / "ds" / "media"])
    mmapped = MediaStoreReader(
        [tmp_path / "ds" / "media"], load_to_ram_threshold_bytes=0
    )
    for mid in meta["ids"].values():
        np.testing.assert_array_equal(ram.tokens(mid), mmapped.tokens(mid))


def test_reader_refuses_unknown_dtype(tmp_path):
    from nemo_rl_apertus.media_store import MediaStoreReader

    build_toy_dataset(tmp_path / "ds")
    with pytest.raises(ValueError, match="token_dtype"):
        MediaStoreReader([tmp_path / "ds" / "media"], token_dtype="<i8")


def test_reader_rejects_duplicate_media_id_across_roots(tmp_path):
    from nemo_rl_apertus.media_store import MediaStoreReader

    build_toy_dataset(tmp_path / "a")
    build_toy_dataset(tmp_path / "b")  # same synthetic ids in both roots
    with pytest.raises(ValueError, match="duplicate media_id"):
        MediaStoreReader([tmp_path / "a" / "media", tmp_path / "b" / "media"])


def test_toy_store_units_are_token_elements(tmp_path):
    """The fixture itself must honor the units contract: offsets/lengths in elements."""
    meta = build_toy_dataset(tmp_path / "ds")
    media = tmp_path / "ds" / "media"
    rows = {
        r["media_id"]: r
        for r in pq.read_table(media / "media.000000.parquet").to_pylist()
    }
    arena = np.fromfile(media / "tokens.000000.bin", dtype="<i4")
    off = 0
    for key in ("img0", "img1", "img2"):
        row = rows[meta["ids"][key]]
        block = meta["blocks"][key]
        assert row["offset_elems"] == off  # elements, not bytes
        assert row["length_elems"] == len(block)
        np.testing.assert_array_equal(arena[off : off + len(block)], block)
        off += len(block)
    assert arena.size == off


@pytest.mark.skipif(
    not GOLDEN_STORE.exists(),
    reason="golden store fixture not generated yet (requires Part A's MediaStoreWriter)",
)
def test_golden_store_roundtrip():
    """Open a store sealed by the PRODUCER's writer with the COPIED reader.

    Conformance is checked through the format's own redundancy, so no expected
    values need to be committed alongside the fixture:
      - content addressing: sha256(raw bytes) == media_id (spec invariant 2),
        which transitively proves the raw byte coordinates are correct;
      - units: reader.tokens() must equal a manual element-unit slice of the
        token bin (spec units contract).
    """
    from nemo_rl_apertus.media_store import MediaStoreReader

    reader = MediaStoreReader([GOLDEN_STORE])
    assert len(reader.index) > 0

    for pq_file in sorted(GOLDEN_STORE.glob("media.*.parquet")):
        shard = int(pq_file.name.split(".")[1])
        arena = np.fromfile(GOLDEN_STORE / f"tokens.{shard:06d}.bin", dtype="<i4")
        for row in pq.read_table(pq_file).to_pylist():
            mid = row["media_id"]
            assert hashlib.sha256(reader.raw(mid)).hexdigest() == mid
            tokens = reader.tokens(mid)
            assert len(tokens) == row["length_elems"]
            np.testing.assert_array_equal(
                tokens,
                arena[row["offset_elems"] : row["offset_elems"] + row["length_elems"]],
            )
