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
"""Reader for the content-addressed media store sealed by ``vision_tokenization``.

On-disk format (the inter-repo contract; spec
``docs/design-docs/apertus-multimodal-preference-data.md``): each media root
holds immutable sealed triples ::

    media.NNNNNN.parquet   media_id (sha256 hex), shard, offset_elems,
                           length_elems, raw_offset_bytes, raw_length_bytes,
                           raw_ext, resize_h, resize_w, kind, source
    tokens.NNNNNN.bin      concatenated encapsulated blocks, raw ``<i4``
    raw.NNNNNN.bin         original media file bytes, concatenated

Units contract: ``offset_elems``/``length_elems`` are TOKEN ELEMENTS; the
``raw_*`` columns are bytes.

This class is a deliberate copy of the producer's ``MediaStoreReader``
(last synced: VT c3eee5e; re-sync on schema_version bump — the golden-store
conformance test is the drift detector)
(``vision_tokenization/pipeline/output/media_store.py``): the contract between
the repos is the on-disk format — pinned by the golden-store fixture under
``tests/fixtures/golden_store/`` — not a shared Python package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TOKEN_DTYPE = np.dtype("<i4")


class MediaStoreReader:
    """Union index over one or more media roots. Init-only IO (spec contract).

    All filesystem work happens here: media parquets become an in-RAM id map,
    and each token bin becomes either a RAM arena (one sequential read, zero
    random IO afterwards) or a read-only memmap when it exceeds
    ``load_to_ram_threshold_bytes``. Steady-state ``tokens()`` is a dict lookup
    plus an arena slice. ``raw()`` opens the raw bin per call and is for
    audits/debugging only — the training consumer never reads it.

    Args:
        roots: Media root directories, each holding sealed triples.
        token_dtype: numpy dtype string from the dataset manifest; only
            ``"<i4"`` is supported (refuse anything else, per spec).
        load_to_ram_threshold_bytes: Token bins at or below this size are read
            fully into RAM; larger bins are memory-mapped.
    """

    def __init__(
        self,
        roots: list[Path | str],
        token_dtype: str = "<i4",
        load_to_ram_threshold_bytes: int = 16 << 30,
    ) -> None:
        if np.dtype(token_dtype) != TOKEN_DTYPE:
            raise ValueError(f"unsupported token_dtype {token_dtype!r}; expected <i4")
        self.index: dict[str, tuple[int, int, int]] = {}  # id -> (arena, off, len)
        self._arenas: list[np.ndarray] = []
        self._raw_index: dict[str, tuple[Path, int, int]] = {}
        for root in (Path(r) for r in roots):
            for pq_file in sorted(root.glob("media.*.parquet")):
                shard = int(pq_file.name.split(".")[1])
                tok_path = root / f"tokens.{shard:06d}.bin"
                raw_path = root / f"raw.{shard:06d}.bin"
                if tok_path.stat().st_size <= load_to_ram_threshold_bytes:
                    arena = np.fromfile(tok_path, dtype=TOKEN_DTYPE)
                else:
                    arena = np.memmap(tok_path, dtype=TOKEN_DTYPE, mode="r")
                a_idx = len(self._arenas)
                self._arenas.append(arena)
                for row in pq.read_table(pq_file).to_pylist():
                    mid = row["media_id"]
                    if mid in self.index:
                        raise ValueError(f"duplicate media_id across roots: {mid}")
                    self.index[mid] = (a_idx, row["offset_elems"], row["length_elems"])
                    self._raw_index[mid] = (
                        raw_path,
                        row["raw_offset_bytes"],
                        row["raw_length_bytes"],
                    )

    def tokens(self, media_id: str) -> np.ndarray:
        """Return the encapsulated token block for ``media_id`` (``<i4`` view)."""
        a_idx, off, length = self.index[media_id]
        return np.asarray(self._arenas[a_idx][off : off + length])

    def raw(self, media_id: str) -> bytes:
        """Return the original media file bytes for ``media_id`` (audit path)."""
        path, off, length = self._raw_index[media_id]
        with open(path, "rb") as f:
            f.seek(off)
            return f.read(length)
