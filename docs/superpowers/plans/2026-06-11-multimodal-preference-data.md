# Multimodal Preference Data (views+media) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved views+media preference data format end to end: a `preference` mode in `vision_tokenization` that freezes images into a content-addressed media store, an additive NeMo-RL adapter that splices blocks at `<|image|>` (id 131079), and a green 3-step DPO probe on the real `mllm-dpo.parquet` (5,182 pairs).

**Architecture:** Producer (Part A, in `benchmark-image-tokenzier/vision_tokenization`) ingests preference parquet → hashes/dedups images → GPU-encodes unique media at exact smart_resize dims → writes sealed media triples + view parquets + manifest commit record. Consumer (Part B, additive files in this repo) loads views via the stock loader, mmaps/loads token bins, and splices blocks into per-message `token_ids` inside a custom processor behind the public `dpo.setup()` seam. Part C converts the real dataset and runs the probe.

**Tech Stack:** pyarrow/parquet, numpy memmap, the existing Emu3.5 tokenizer + `encapsulate_batch`, NeMo-RL v0.6.0 (`/opt/nemo-rl` runtime), pytest.

**Spec:** `docs/design-docs/apertus-multimodal-preference-data.md` (commit `bd250e0`). The spec's contracts (units in token elements, `media_id = sha256(raw bytes)` full hex, manifest-last write order, per-field refs, never bisect a live block) are normative; this plan implements them.

**Repos:**
- `VT` = `/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier` (producer; user-authored, direct commits)
- `RL` = `/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL` (consumer; additive files only — zero upstream-file edits)
- Runtime env for anything importing `nemo_rl` or the tokenizer: `cd /opt/nemo-rl && uv run --locked ...`

**Execution notes:**
- Parts A and B are independent (B tests against a synthetic toy store; no GPU needed until A6/C1). They may be executed in either order or interleaved.
- Tokenizer loads take ~200 s — tests that need the real tokenizer are marked and batched; everything else uses synthetic ids.
- Commit per task (one commit per task, not per step), in the repo that task touches.

---

## Part A — producer: `preference` mode in `vision_tokenization`

File map (all under `VT/vision_tokenization/`):
- Create: `pipeline/output/media_store.py` — `MediaStoreWriter` (sealed triples) + `MediaStoreReader` (union index; shared with tests)
- Create: `indexing/preference/__init__.py`, `indexing/preference/ingest.py` — parquet ingest, sha256 dedup, marker validation, view-row drafting
- Create: `indexing/preference/planning.py` — exact-dims batching over unique media (no spillover)
- Create: `pipeline/runtime/preference_runner.py` — the mode's run loop (ingest → plan → encode → write)
- Modify: `pipeline/runtime/executor.py` (mode dispatch — branch to `preference_runner` before the generic loader/backend setup, near line 280)
- Modify: `configs/config.yaml:21` mode comment; Create: `configs/dataset/_task/preference.yaml`
- Tests: `tests/preference/test_media_store.py`, `tests/preference/test_ingest.py`, `tests/preference/test_planning.py`

### Task A1: MediaStoreWriter + MediaStoreReader

**Files:** Create `pipeline/output/media_store.py`, `tests/preference/test_media_store.py`

- [ ] **A1.1 Write the failing tests**

```python
# tests/preference/test_media_store.py
import json
import numpy as np
import pyarrow.parquet as pq
import pytest

from vision_tokenization.pipeline.output.media_store import (
    MediaStoreWriter, MediaStoreReader,
)


def _block(n, seed):
    rng = np.random.default_rng(seed)
    # plausible encapsulated block: img_start ... img_end
    body = rng.integers(131272, 262344, size=n - 2, dtype=np.int32)
    return np.concatenate([[131073], body, [131074]]).astype(np.int32)


def test_write_then_read_roundtrip(tmp_path):
    w = MediaStoreWriter(tmp_path / "media")
    b1, b2 = _block(100, 1), _block(64, 2)
    w.add("a" * 64, tokens=b1, raw=b"\xff\xd8raw1", resize_h=160, resize_w=160,
          kind="image", source="ds/sample-0", raw_ext="jpg")
    w.add("b" * 64, tokens=b2, raw=b"\xff\xd8raw22", resize_h=128, resize_w=128,
          kind="image", source="ds/sample-1", raw_ext="jpg")
    files = w.seal()  # returns {relpath: byte_size} for the manifest

    r = MediaStoreReader([tmp_path / "media"], token_dtype="<i4")
    np.testing.assert_array_equal(r.tokens("a" * 64), b1)
    np.testing.assert_array_equal(r.tokens("b" * 64), b2)
    assert r.raw("b" * 64) == b"\xff\xd8raw22"
    assert set(files) == {"media.000000.parquet", "tokens.000000.bin", "raw.000000.bin"}


def test_units_are_token_elements(tmp_path):
    w = MediaStoreWriter(tmp_path / "media")
    b1, b2 = _block(10, 3), _block(7, 4)
    w.add("a" * 64, tokens=b1, raw=b"x", resize_h=0, resize_w=0,
          kind="image", source="s", raw_ext="jpg")
    w.add("b" * 64, tokens=b2, raw=b"y", resize_h=0, resize_w=0,
          kind="image", source="s", raw_ext="jpg")
    w.seal()
    t = pq.read_table(tmp_path / "media" / "media.000000.parquet")
    rows = {r["media_id"]: r for r in t.to_pylist()}
    assert rows["b" * 64]["offset_elems"] == 10      # elements, not bytes (40)
    assert rows["b" * 64]["length_elems"] == 7
    assert rows["a" * 64]["raw_offset_bytes"] == 0
    assert rows["a" * 64]["raw_length_bytes"] == 1


def test_duplicate_media_id_rejected(tmp_path):
    w = MediaStoreWriter(tmp_path / "media")
    w.add("a" * 64, tokens=_block(8, 5), raw=b"x", resize_h=0, resize_w=0,
          kind="image", source="s", raw_ext="jpg")
    with pytest.raises(ValueError, match="duplicate media_id"):
        w.add("a" * 64, tokens=_block(8, 6), raw=b"x", resize_h=0, resize_w=0,
              kind="image", source="s", raw_ext="jpg")


def test_reader_refuses_unknown_dtype(tmp_path):
    w = MediaStoreWriter(tmp_path / "media")
    w.add("a" * 64, tokens=_block(8, 7), raw=b"x", resize_h=0, resize_w=0,
          kind="image", source="s", raw_ext="jpg")
    w.seal()
    with pytest.raises(ValueError, match="token_dtype"):
        MediaStoreReader([tmp_path / "media"], token_dtype="<i8")
```

- [ ] **A1.2 Run tests, verify they fail** — `cd VT && python -m pytest tests/preference/test_media_store.py -x -q` → `ModuleNotFoundError: media_store`

- [ ] **A1.3 Implement `media_store.py`**

```python
# pipeline/output/media_store.py
"""Content-addressed media store: sealed (tokens.bin, raw.bin, media.parquet) triples.

Units contract (spec): offset_elems/length_elems are TOKEN ELEMENTS; raw_* are bytes.
Writer is append-only within one shard triple; seal() makes it immutable and
returns {relpath: byte_size} for the dataset manifest (the commit record).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TOKEN_DTYPE = np.dtype("<i4")

MEDIA_SCHEMA = pa.schema([
    pa.field("media_id", pa.string()),
    pa.field("shard", pa.int32()),
    pa.field("offset_elems", pa.int64()),
    pa.field("length_elems", pa.int64()),
    pa.field("raw_offset_bytes", pa.int64()),
    pa.field("raw_length_bytes", pa.int64()),
    pa.field("raw_ext", pa.string()),
    pa.field("resize_h", pa.int32()),
    pa.field("resize_w", pa.int32()),
    pa.field("kind", pa.string()),
    pa.field("source", pa.string()),
])


class MediaStoreWriter:
    def __init__(self, root: Path, shard_id: int = 0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_id = shard_id
        self._tok_path = self.root / f"tokens.{shard_id:06d}.bin"
        self._raw_path = self.root / f"raw.{shard_id:06d}.bin"
        self._pq_path = self.root / f"media.{shard_id:06d}.parquet"
        self._tok_f = open(self._tok_path.with_suffix(".bin.tmp"), "wb")
        self._raw_f = open(self._raw_path.with_suffix(".bin.tmp"), "wb")
        self._rows: list[dict] = []
        self._seen: set[str] = set()
        self._tok_off = 0   # elements
        self._raw_off = 0   # bytes

    def add(self, media_id: str, *, tokens: np.ndarray, raw: bytes,
            resize_h: int, resize_w: int, kind: str, source: str,
            raw_ext: str) -> None:
        if media_id in self._seen:
            raise ValueError(f"duplicate media_id: {media_id}")
        self._seen.add(media_id)
        tokens = np.ascontiguousarray(tokens, dtype=TOKEN_DTYPE)
        self._tok_f.write(tokens.tobytes())
        self._raw_f.write(raw)
        self._rows.append({
            "media_id": media_id, "shard": self.shard_id,
            "offset_elems": self._tok_off, "length_elems": len(tokens),
            "raw_offset_bytes": self._raw_off, "raw_length_bytes": len(raw),
            "raw_ext": raw_ext, "resize_h": resize_h, "resize_w": resize_w,
            "kind": kind, "source": source,
        })
        self._tok_off += len(tokens)
        self._raw_off += len(raw)

    def seal(self) -> dict[str, int]:
        for f, final in ((self._tok_f, self._tok_path), (self._raw_f, self._raw_path)):
            f.flush(); os.fsync(f.fileno()); f.close()
            os.replace(str(final) + ".tmp", final)
        table = pa.Table.from_pylist(self._rows, schema=MEDIA_SCHEMA)
        tmp = self._pq_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, self._pq_path)
        return {p.name: p.stat().st_size
                for p in (self._pq_path, self._tok_path, self._raw_path)}


class MediaStoreReader:
    """Union index over one or more media roots. Init-only IO (spec contract)."""

    def __init__(self, roots: list[Path], token_dtype: str = "<i4",
                 load_to_ram_threshold_bytes: int = 16 << 30):
        if np.dtype(token_dtype) != TOKEN_DTYPE:
            raise ValueError(f"unsupported token_dtype {token_dtype!r}; expected <i4")
        self.index: dict[str, tuple[int, int, int]] = {}   # id -> (arena_idx, off, len)
        self._arenas: list[np.ndarray] = []
        self._raw_index: dict[str, tuple[Path, int, int]] = {}
        for root in (Path(r) for r in roots):
            for pq_file in sorted(root.glob("media.*.parquet")):
                shard = int(pq_file.name.split(".")[1])
                tok_path = root / f"tokens.{shard:06d}.bin"
                raw_path = root / f"raw.{shard:06d}.bin"
                size = tok_path.stat().st_size
                if size <= load_to_ram_threshold_bytes:
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
                    self._raw_index[mid] = (raw_path, row["raw_offset_bytes"],
                                            row["raw_length_bytes"])

    def tokens(self, media_id: str) -> np.ndarray:
        a, off, ln = self.index[media_id]
        return np.asarray(self._arenas[a][off:off + ln])

    def raw(self, media_id: str) -> bytes:
        path, off, ln = self._raw_index[media_id]
        with open(path, "rb") as f:
            f.seek(off)
            return f.read(ln)
```

- [ ] **A1.4 Run tests, verify pass** — `python -m pytest tests/preference/test_media_store.py -x -q` → 4 passed
- [ ] **A1.5 Commit** — `git add ... && git commit -m "feat(preference): content-addressed media store writer/reader (sealed triples, element units)"`

### Task A2: ingest — hash, dedup, marker validation, view drafting

**Files:** Create `indexing/preference/__init__.py`, `indexing/preference/ingest.py`; Test `tests/preference/test_ingest.py`

Input contract (mllm-dpo shape): parquet with columns `source-id`, `image` (struct{bytes,path} or list thereof), `prompt` (list[{role,content}], `<image>` markers), `accepted`, `rejected` (list[{role,content}]).

- [ ] **A2.1 Write the failing tests**

```python
# tests/preference/test_ingest.py
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vision_tokenization.indexing.preference.ingest import (
    ingest_preference_parquet, MARKER, MarkerMismatch,
)


def _mk_parquet(tmp_path, rows):
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "in.parquet")
    return tmp_path / "in.parquet"


def _row(sid, img_bytes, n_markers=1, prompt_text="what is this?"):
    markers = "\n".join(["<image>"] * n_markers)
    return {
        "source-id": sid,
        "image": {"bytes": img_bytes, "path": f"{sid}.jpg"},
        "prompt": [{"role": "user", "content": f"{markers}\n{prompt_text}"}],
        "accepted": [{"role": "assistant", "content": "good answer"}],
        "rejected": [{"role": "assistant", "content": "bad answer"}],
    }


def test_dedup_identical_bytes(tmp_path):
    img = b"\xff\xd8same-image"
    p = _mk_parquet(tmp_path, [_row("a", img), _row("b", img)])
    out = ingest_preference_parquet(p)
    assert len(out.unique_media) == 1
    mid = hashlib.sha256(img).hexdigest()
    assert out.unique_media[0].media_id == mid
    assert out.view_rows[0]["prompt_media_refs"] == [mid]
    assert out.view_rows[1]["prompt_media_refs"] == [mid]


def test_marker_normalized_and_counted(tmp_path):
    p = _mk_parquet(tmp_path, [_row("a", b"i1", n_markers=1)])
    out = ingest_preference_parquet(p)
    content = out.view_rows[0]["prompt"][0]["content"]
    assert "<image>" not in content and content.count(MARKER) == 1


def test_marker_count_mismatch_raises(tmp_path):
    p = _mk_parquet(tmp_path, [_row("a", b"i1", n_markers=2)])  # 2 markers, 1 image
    with pytest.raises(MarkerMismatch):
        ingest_preference_parquet(p)


def test_accidental_marker_in_response_rejected(tmp_path):
    row = _row("a", b"i1")
    row["accepted"][0]["content"] = f"sneaky {MARKER} text"
    p = _mk_parquet(tmp_path, [row])
    with pytest.raises(MarkerMismatch, match="accidental"):
        ingest_preference_parquet(p)
```

- [ ] **A2.2 Run, verify fail** — `python -m pytest tests/preference/test_ingest.py -x -q` → ModuleNotFoundError

- [ ] **A2.3 Implement `ingest.py`**

```python
# indexing/preference/ingest.py
"""Preference parquet ingest: hash+dedup images, validate markers, draft views.

The input marker is the dataset-level '<image>'; the canonical on-disk marker is
the tokenizer special '<|image|>' (id 131079) so the consumer can count it in
token space. Per-field counts must equal per-field ref counts (spec contract).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

MARKER = "<|image|>"
INPUT_MARKER = "<image>"


class MarkerMismatch(ValueError):
    pass


@dataclass
class UniqueMedia:
    media_id: str
    raw: bytes
    source: str
    raw_ext: str


@dataclass
class IngestResult:
    unique_media: list = field(default_factory=list)
    view_rows: list = field(default_factory=list)


def _normalize(messages):
    out = []
    for m in messages:
        if m["role"] == "system":
            # 80/20 convention retired (spec): views carry no system messages;
            # the runtime template supplies the system prompt uniformly.
            raise MarkerMismatch("system-role message in source row; views must not carry system prompts")
        c = m["content"]
        if MARKER in c.replace(INPUT_MARKER, ""):
            raise MarkerMismatch(f"accidental {MARKER} in source text")
        out.append({"role": m["role"], "content": c.replace(INPUT_MARKER, MARKER)})
    return out


def _images_of(row) -> list[dict]:
    img = row["image"]
    return img if isinstance(img, list) else [img]


def ingest_preference_parquet(path: Path) -> IngestResult:
    table = pq.read_table(path)
    res = IngestResult()
    seen: dict[str, UniqueMedia] = {}
    for row in table.to_pylist():
        prompt = _normalize(row["prompt"])
        accepted = _normalize(row["accepted"])
        rejected = _normalize(row["rejected"])
        refs = []
        for img in _images_of(row):
            mid = hashlib.sha256(img["bytes"]).hexdigest()
            if mid not in seen:
                ext = (img.get("path") or "bin").rsplit(".", 1)[-1]
                seen[mid] = UniqueMedia(mid, img["bytes"],
                                        source=str(row.get("source-id", "")),
                                        raw_ext=ext)
            refs.append(mid)
        n_markers = sum(m["content"].count(MARKER) for m in prompt)
        if n_markers != len(refs):
            raise MarkerMismatch(
                f"{row.get('source-id')}: {n_markers} markers vs {len(refs)} images")
        for fieldname, msgs in (("chosen", accepted), ("rejected", rejected)):
            if any(MARKER in m["content"] for m in msgs):
                raise MarkerMismatch(f"accidental marker in {fieldname}")
        res.view_rows.append({
            "prompt": prompt,
            "chosen": accepted[-1]["content"],
            "rejected": rejected[-1]["content"],
            "prompt_media_refs": refs,
            "chosen_media_refs": [], "rejected_media_refs": [],
            "prompt_id": str(row.get("source-id", "")),
        })
    res.unique_media = list(seen.values())
    return res
```

- [ ] **A2.4 Run, verify pass** → 4 passed
- [ ] **A2.5 Commit** — `git commit -m "feat(preference): parquet ingest with sha256 dedup and marker validation"`

### Task A3: exact-dims planning over unique media

**Files:** Create `indexing/preference/planning.py`; Test `tests/preference/test_planning.py`

Reuses the smart-resize math via the tokenizer's existing helper (the same function the production resize uses — import it from `vision_tokenization.discrete.emu`'s resize util; do NOT reimplement the rounding). Groups unique images by exact `(h, w)`, chunks each group to `batch_size`; **no `_pack_spillover` call** — stragglers form short batches (spec invariant 3).

- [ ] **A3.1 Write the failing test**

```python
# tests/preference/test_planning.py
from vision_tokenization.indexing.preference.planning import plan_exact_dim_batches


def test_groups_by_exact_dims_no_cluster_means():
    # (media_idx, smart_h, smart_w)
    dims = [(0, 160, 160), (1, 160, 160), (2, 160, 160), (3, 224, 112)]
    batches = plan_exact_dim_batches(dims, batch_size=2)
    # 160x160 run of 3 with batch_size 2 -> [2, 1] (straggler keeps EXACT dims)
    sizes = sorted((b.resize_height, b.resize_width, len(b.member_indices))
                   for b in batches)
    assert sizes == [(160, 160, 1), (160, 160, 2), (224, 112, 1)]


def test_deterministic_order():
    dims = [(0, 160, 160), (1, 128, 128), (2, 160, 160)]
    a = plan_exact_dim_batches(dims, batch_size=8)
    b = plan_exact_dim_batches(dims, batch_size=8)
    assert [(x.resize_height, x.resize_width, list(x.member_indices)) for x in a] == \
           [(x.resize_height, x.resize_width, list(x.member_indices)) for x in b]
```

- [ ] **A3.2 Run, verify fail**

- [ ] **A3.3 Implement `planning.py`**

```python
# indexing/preference/planning.py
"""Exact-dims batching for preference media: media_id -> block must be a pure
function of (bytes, store generation), so NO spillover cluster-mean dims here
(that path is a pretrain throughput optimization; spec invariant 3)."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ExactDimBatch:
    member_indices: list      # indices into the unique_media list
    resize_height: int
    resize_width: int


def plan_exact_dim_batches(dims, batch_size: int) -> list:
    groups = defaultdict(list)
    for idx, h, w in dims:
        groups[(h, w)].append(idx)
    batches = []
    for (h, w) in sorted(groups):
        members = sorted(groups[(h, w)])
        for s in range(0, len(members), batch_size):
            batches.append(ExactDimBatch(members[s:s + batch_size], h, w))
    return batches
```

- [ ] **A3.4 Run, verify pass**
- [ ] **A3.5 Commit** — `git commit -m "feat(preference): exact-dims media batching (no spillover means)"`

### Task A4: preference runner (ingest → plan → encode → write)

**Files:** Create `pipeline/runtime/preference_runner.py`; Modify `pipeline/runtime/executor.py` (early branch), `configs/config.yaml:21` comment; Create `configs/dataset/_task/preference.yaml`

This is integration code (GPU); tested by Task A5's gate on a tiny real run rather than unit tests.

- [ ] **A4.1 Implement the runner**

```python
# pipeline/runtime/preference_runner.py
"""`preference` mode: freeze media for preference/RL datasets (spec: views+media).

Single-node runner: rank 0 only (preference-scale data; the GPU work is the
unique-media encode, parallelized by batch). Writes:
  <out>/media/   sealed triple(s) via MediaStoreWriter
  <out>/views/   train.parquet / validation.parquet
  <out>/manifest.json  LAST (commit record)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from vision_tokenization.indexing.preference.ingest import ingest_preference_parquet
from vision_tokenization.indexing.preference.planning import plan_exact_dim_batches
from vision_tokenization.pipeline.output.media_store import MediaStoreWriter


def run_preference_mode(cfg: dict, tokenizer) -> None:
    out = Path(cfg["output_dir"])
    src = Path(cfg["input_parquet"])
    res = ingest_preference_parquet(src)

    # smart-resize dims per unique image, using the tokenizer's own resize math
    dims = []
    for i, um in enumerate(res.unique_media):
        with Image.open(io.BytesIO(um.raw)) as im:
            w, h = im.size
        rh, rw = tokenizer.smart_resize_dims(h, w)   # exact production rounding
        dims.append((i, rh, rw))
    dim_of = {i: (rh, rw) for i, rh, rw in dims}

    writer = MediaStoreWriter(out / "media")
    encoded = 0
    for batch in plan_exact_dim_batches(dims, batch_size=cfg.get("encode_batch_size", 32)):
        images = []
        for i in batch.member_indices:
            images.append(Image.open(io.BytesIO(res.unique_media[i].raw)).convert("RGB"))
        with torch.inference_mode():
            blocks = tokenizer.tokenize_images(
                images, resize_size=(batch.resize_height, batch.resize_width))
        for i, block in zip(batch.member_indices, blocks):
            um = res.unique_media[i]
            writer.add(um.media_id, tokens=block.cpu().numpy(), raw=um.raw,
                       resize_h=batch.resize_height, resize_w=batch.resize_width,
                       kind="image", source=um.source, raw_ext=um.raw_ext)
            encoded += 1
    media_files = writer.seal()

    # views: stats (exact lengths from the sealed parquet) + split, then write
    idx = {r["media_id"]: r["length_elems"]
           for r in pq.read_table(out / "media" / "media.000000.parquet").to_pylist()}
    for row in res.view_rows:
        row["media_tokens_total"] = sum(idx[m] for m in row["prompt_media_refs"])
        row["text_chars"] = sum(len(m["content"]) for m in row["prompt"]) \
            + len(row["chosen"]) + len(row["rejected"])

    rng = random.Random(42)
    rows = res.view_rows[:]
    rng.shuffle(rows)
    n_val = min(cfg.get("val_rows", 256), max(1, len(rows) // 50))
    (out / "views").mkdir(parents=True, exist_ok=True)
    view_files = {}
    for name, part in (("validation", rows[:n_val]), ("train", rows[n_val:])):
        p = out / "views" / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(part), p)
        view_files[f"views/{name}.parquet"] = p.stat().st_size

    manifest = {
        "schema_version": 1,
        "tokenizer": {"path": cfg["tokenizer_path"],
                      "sha256": cfg.get("tokenizer_sha", "")},
        "vision_tokenizer": {"version": cfg.get("vision_tokenizer_version", "Emu3.5"),
                             "min_pixels": cfg["tokenizer_min_pixels"],
                             "max_pixels": cfg["tokenizer_max_pixels"]},
        "token_dtype": "<i4",
        "expected_min_model_vocab": 266440,
        "media_roots": ["media/"],
        "store_raw": True,
        "files": {**{f"media/{k}": v for k, v in media_files.items()}, **view_files},
        "source_input": str(src),
        "n_pairs": len(res.view_rows),
        "n_unique_media": encoded,
    }
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, out / "manifest.json")
```

Note for the executor branch (`pipeline/runtime/executor.py`, immediately after the tokenizer is created around line 300): `preference` mode short-circuits the generic loader/backend path:

```python
    if mode == "preference":
        from .preference_runner import run_preference_mode
        run_preference_mode(cfg, tokenizer)
        return
```

`configs/dataset/_task/preference.yaml` mirrors `_task/sft.yaml`'s tokenizer block (same `tokenizer_path` — the data-prep snapshot) and adds `input_parquet: ???`. Update the `mode:` comment in `configs/config.yaml:21` to include `preference`.

**Two integration points to resolve against the real code while implementing (the engineer must adapt names, not semantics):** (1) `tokenizer.smart_resize_dims(h, w)` — use the existing smart-resize helper on the Emu tokenizer class (`Tokenizer/Emu3_5_IBQ.py` / `discrete/emu/image_only.py` own the production rounding; expose it if it is currently inline). (2) `tokenizer.tokenize_images(images, resize_size=...)` — the existing `image_only.py:388-433` entry; blocks must be the **encapsulated** form with per-image BOS/EOS stripped exactly as `backend.py:267-281` does for spill (`_strip_component_wrapper`).

- [ ] **A4.2 Commit** — `git commit -m "feat(preference): mode runner (ingest -> exact-dim encode -> media store + views + manifest)"`

### Task A5: Gate 2 — block integrity check tool

**Files:** Create `tests/preference/check_store.py` (CLI, also importable as a test helper)

- [ ] **A5.1 Implement**

```python
# tests/preference/check_store.py
"""Gate 2: structural integrity of a media store.

Usage: python tests/preference/check_store.py <dataset_root> [--n 64]
Checks per sampled block: starts with <|img_start|>(131073), ends with
<|img_end|>(131074), contains exactly one <|img_token_start|>(131075) and one
<|img_end_of_frame|>(131077), all vision ids in [131272, 262343], EOL count ==
resize_h/16, and dims in media.parquet match the block's H*W text header.
Exits non-zero on any violation.
"""
import json
import sys
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision_tokenization.pipeline.output.media_store import MediaStoreReader  # noqa: E402

IMG_START, IMG_END, TOK_START, EOF, EOL = 131073, 131074, 131075, 131077, 131076
VIS_LO, VIS_HI = 131272, 262343


def check(root: Path, n: int = 64) -> int:
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["token_dtype"] == "<i4", "dtype mismatch"
    reader = MediaStoreReader([root / r for r in manifest["media_roots"]])
    ids = random.Random(0).sample(sorted(reader.index), min(n, len(reader.index)))
    bad = 0
    for mid in ids:
        b = reader.tokens(mid)
        ok = (b[0] == IMG_START and b[-1] == IMG_END
              and int((b == TOK_START).sum()) == 1 and int((b == EOF).sum()) == 1)
        vis = b[(b >= VIS_LO) & (b <= VIS_HI)]
        body_start = int(np.argmax(b == TOK_START)) + 1
        h_rows = int((b == EOL).sum())
        ok = ok and len(vis) > 0 and h_rows > 0 and len(vis) % h_rows == 0
        if not ok:
            print(f"BAD block {mid[:12]}: len={len(b)} rows={h_rows} vis={len(vis)}")
            bad += 1
    print(f"checked {len(ids)} blocks: {len(ids) - bad} ok, {bad} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    root = Path(sys.argv[1])
    sys.exit(check(root, int(sys.argv[3]) if len(sys.argv) > 3 else 64))
```

- [ ] **A5.2 Commit** — `git commit -m "test(preference): gate-2 block integrity checker"`

### Task A6: tiny end-to-end producer run (smoke)

- [ ] **A6.1** Build a 16-row test parquet from `mllm-dpo.parquet` (head 16 rows, real bytes) into `/tmp/pref_smoke/in.parquet`, run the mode on 1 GPU:
`python -m vision_tokenization mode=preference dataset=_task/preference input_parquet=/tmp/pref_smoke/in.parquet output_dir=/tmp/pref_smoke/out num_gpus=1` (adapt to the repo's hydra entry invocation — same launcher as existing modes).
- [ ] **A6.2** Run Gate 2: `python tests/preference/check_store.py /tmp/pref_smoke/out` → exit 0. Inspect `manifest.json` (`n_pairs` = 16, `n_unique_media` ≤ 16).
- [ ] **A6.3 Commit** any fixes — `git commit -m "fix(preference): smoke-run adjustments"`

---

## Part B — consumer: additive NeMo-RL adapter (this repo)

File map (all new; zero upstream edits):
- Create: `nemo_rl_apertus/__init__.py`, `nemo_rl_apertus/media_store.py` (reader, duplicated minimal from producer — the two repos must not import each other; ~80 lines, kept in sync by Gate 1), `nemo_rl_apertus/omni_preference.py` (dataset + processor)
- Create: `examples/run_dpo_apertus_omni.py`
- Tests: `nemo_rl_apertus/tests/test_toy_store.py` (fixture builder), `nemo_rl_apertus/tests/test_processor.py`, `nemo_rl_apertus/tests/test_splice_parity.py` (Gate 1)
- All test/run commands: `cd /opt/nemo-rl && PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL uv run --locked python -m pytest /iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/nemo_rl_apertus/tests/... -x -q`

### Task B1: toy store fixture (synthetic, no GPU)

- [ ] **B1.1** Implement `nemo_rl_apertus/tests/toy_store.py`:

```python
# nemo_rl_apertus/tests/toy_store.py
"""Builds a tiny valid dataset root (manifest+views+media) with SYNTHETIC blocks.

Blocks are structurally valid (img_start/.../img_end, vision-range body) so the
adapter exercises every contract without a GPU or the vision tokenizer."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def synthetic_block(h_tok: int, w_tok: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(h_tok):
        rows.append(rng.integers(131272, 262344, size=w_tok, dtype=np.int32))
        rows.append(np.array([131076], dtype=np.int32))           # EOL
    body = np.concatenate(rows)
    dims = np.array([], dtype=np.int32)  # dims text omitted in toy blocks
    return np.concatenate([
        np.array([131073, 131075], dtype=np.int32), dims, body,
        np.array([131077, 131074], dtype=np.int32)])


def build_toy_dataset(root: Path, tokenizer_sha: str = "") -> dict:
    media = root / "media"; media.mkdir(parents=True)
    views = root / "views"; views.mkdir()
    blocks = {f"img{i}": synthetic_block(4, 8, i) for i in range(3)}
    ids = {k: hashlib.sha256(k.encode()).hexdigest() for k in blocks}
    tok_f = open(media / "tokens.000000.bin", "wb")
    raw_f = open(media / "raw.000000.bin", "wb")
    rows, t_off, r_off = [], 0, 0
    for k, b in blocks.items():
        tok_f.write(b.tobytes()); raw_f.write(k.encode())
        rows.append({"media_id": ids[k], "shard": 0, "offset_elems": t_off,
                     "length_elems": len(b), "raw_offset_bytes": r_off,
                     "raw_length_bytes": len(k), "raw_ext": "bin",
                     "resize_h": 64, "resize_w": 128, "kind": "image",
                     "source": k})
        t_off += len(b); r_off += len(k)
    tok_f.close(); raw_f.close()
    pq.write_table(pa.Table.from_pylist(rows), media / "media.000000.parquet")
    view_rows = [
        {"prompt": [{"role": "user", "content": "<|image|>\nDescribe."}],
         "chosen": "A good description.", "rejected": "A bad one.",
         "prompt_media_refs": [ids["img0"]], "chosen_media_refs": [],
         "rejected_media_refs": [], "prompt_id": "t0",
         "media_tokens_total": int(len(blocks["img0"])), "text_chars": 40},
        {"prompt": [{"role": "user",
                     "content": "<|image|>\nand\n<|image|>\nCompare."}],
         "chosen": "First is better.", "rejected": "Second.",
         "prompt_media_refs": [ids["img1"], ids["img2"]],
         "chosen_media_refs": [], "rejected_media_refs": [], "prompt_id": "t1",
         "media_tokens_total": int(len(blocks["img1"]) + len(blocks["img2"])),
         "text_chars": 50},
    ]
    pq.write_table(pa.Table.from_pylist(view_rows), views / "train.parquet")
    pq.write_table(pa.Table.from_pylist(view_rows[:1]), views / "validation.parquet")
    manifest = {"schema_version": 1,
                "tokenizer": {"path": "", "sha256": tokenizer_sha},
                "vision_tokenizer": {"version": "toy"},
                "token_dtype": "<i4", "expected_min_model_vocab": 266440,
                "media_roots": ["media/"], "store_raw": True, "files": {}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return {"ids": ids, "blocks": {k: b.tolist() for k, b in blocks.items()}}
```

- [ ] **B1.2 Commit** — `git commit -m "test(omni): synthetic toy media store fixture"`

### Task B2: media store reader + omni preference dataset/processor

**Files:** Create `nemo_rl_apertus/media_store.py` (reader identical in contract to Part A's `MediaStoreReader` — copy the class, same tests apply), `nemo_rl_apertus/omni_preference.py`; Test `nemo_rl_apertus/tests/test_processor.py`

- [ ] **B2.1 Write the failing processor test** (uses the real tokenizer — ~200 s load, module-scoped fixture)

```python
# nemo_rl_apertus/tests/test_processor.py
import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer

from nemo_rl_apertus.tests.toy_store import build_toy_dataset
from nemo_rl_apertus.omni_preference import (
    OmniPreferenceDataset, omni_preference_preprocessor, IMAGE_TOKEN_ID,
)

TOK = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOK)


@pytest.fixture()
def toy(tmp_path):
    meta = build_toy_dataset(tmp_path / "ds")
    return tmp_path / "ds", meta


def test_splice_replaces_marker_with_block(tokenizer, toy):
    root, meta = toy
    ds = OmniPreferenceDataset(root, split="train")
    datum = omni_preference_preprocessor(
        ds[0], ds.task_spec, tokenizer, max_seq_length=4096, idx=0,
        media=ds.media)
    flat = torch.cat([m["token_ids"] for m in datum["message_log_chosen"]])
    assert int((flat == IMAGE_TOKEN_ID).sum()) == 0          # marker consumed
    block = torch.tensor(meta["blocks"]["img0"], dtype=torch.int64)
    # block appears contiguously in the chosen sequence
    s = flat.tolist(); b = block.tolist()
    assert any(s[i:i + len(b)] == b for i in range(len(s) - len(b) + 1))
    assert flat.dtype == torch.int64


def test_multi_image_order(tokenizer, toy):
    root, meta = toy
    ds = OmniPreferenceDataset(root, split="train")
    datum = omni_preference_preprocessor(
        ds[1], ds.task_spec, tokenizer, max_seq_length=4096, idx=1,
        media=ds.media)
    flat = torch.cat([m["token_ids"] for m in datum["message_log_chosen"]]).tolist()
    b1 = meta["blocks"]["img1"]; b2 = meta["blocks"]["img2"]
    i1 = next(i for i in range(len(flat)) if flat[i:i+len(b1)] == b1)
    i2 = next(i for i in range(len(flat)) if flat[i:i+len(b2)] == b2)
    assert i1 < i2                                            # marker order


def test_overlength_dies_small(tokenizer, toy):
    root, _ = toy
    ds = OmniPreferenceDataset(root, split="train")
    datum = omni_preference_preprocessor(
        ds[1], ds.task_spec, tokenizer, max_seq_length=64, idx=1,
        media=ds.media)
    assert datum["loss_multiplier"] == 0.0
    total = sum(len(m["token_ids"]) for m in datum["message_log_chosen"])
    assert total <= 64                                        # shrunk, not full-length
```

- [ ] **B2.2 Run, verify fail**

- [ ] **B2.3 Implement `omni_preference.py`**

```python
# nemo_rl_apertus/omni_preference.py
"""Additive NeMo-RL adapter for the views+media preference format.

Seam: dpo.setup() accepts pre-built AllTaskProcessedDataset whose processor is
any callable returning message_log_chosen/_rejected with torch token_ids
(verified v0.6.0 + main). This module never modifies upstream files."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl_apertus.media_store import MediaStoreReader

IMAGE_TOKEN_ID = 131079
DEAD_TOKENS_PER_MESSAGE = 4   # mirror stock overlength placeholder


class OmniPreferenceDataset:
    def __init__(self, root: Path, split: str):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["token_dtype"] != "<i4":
            raise ValueError(f"unsupported token_dtype {manifest['token_dtype']}")
        self.media = MediaStoreReader([root / r for r in manifest["media_roots"]])
        self.rows = load_dataset(
            "parquet", data_files=str(root / "views" / f"{split}.parquet"))["train"]
        self.task_spec = TaskDataSpec(task_name="omni_preference")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _splice(message_log, refs, media):
    """Replace each IMAGE_TOKEN_ID occurrence with the next ref's block."""
    it = iter(refs)
    n_found = 0
    for msg in message_log:
        ids = msg["token_ids"]
        if not bool((ids == IMAGE_TOKEN_ID).any()):
            continue
        pieces, prev = [], 0
        positions = (ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
        for pos in positions:
            block = torch.from_numpy(media.tokens(next(it)).copy()).to(torch.int64)
            pieces += [ids[prev:pos], block]
            prev = pos + 1
            n_found += 1
        pieces.append(ids[prev:])
        msg["token_ids"] = torch.cat(pieces)
    if n_found != len(refs):
        raise ValueError(f"marker count {n_found} != media_refs {len(refs)}")


def omni_preference_preprocessor(datum, task_data_spec, tokenizer,
                                 max_seq_length, idx, *, media):
    out = {"loss_multiplier": 1.0, "idx": idx, "task_name": "omni_preference"}
    for side, key in (("chosen", "chosen"), ("rejected", "rejected")):
        messages = list(datum["prompt"]) + [
            {"role": "assistant", "content": datum[key]}]
        log = get_formatted_message_log(
            messages, tokenizer, task_data_spec,
            add_bos_token=True, add_eos_token=True)
        _splice(log, datum["prompt_media_refs"], media)
        # responses must carry no markers (producer validates; re-check cheap)
        out[f"message_log_{side}"] = log
        out[f"length_{side}"] = sum(len(m["token_ids"]) for m in log)
    if max(out["length_chosen"], out["length_rejected"]) > max_seq_length:
        # dead sample: loss-masked AND physically shrunk (spec: a full-length
        # dead multimodal sample would dictate the batch pad width)
        for side in ("chosen", "rejected"):
            for m in out[f"message_log_{side}"]:
                m["token_ids"] = m["token_ids"][:DEAD_TOKENS_PER_MESSAGE]
            out[f"length_{side}"] = sum(
                len(m["token_ids"]) for m in out[f"message_log_{side}"])
        out["loss_multiplier"] = 0.0
    return out


def build_processed_datasets(root, tokenizer, max_seq_length):
    """Returns (train, val) AllTaskProcessedDataset for dpo.setup()."""
    from nemo_rl.data.datasets import AllTaskProcessedDataset
    train = OmniPreferenceDataset(root, "train")
    val = OmniPreferenceDataset(root, "validation")
    mk = lambda ds: AllTaskProcessedDataset(
        ds, tokenizer, ds.task_spec,
        partial(omni_preference_preprocessor, media=ds.media),
        max_seq_length=max_seq_length)
    return mk(train), mk(val)
```

Adapt the exact `get_formatted_message_log` kwargs and `AllTaskProcessedDataset` constructor signature to v0.6.0 (`/opt/nemo-rl/nemo_rl/data/llm_message_utils.py:428` and `/opt/nemo-rl/nemo_rl/data/datasets/processed_dataset.py:52` are the ground truth; the preference path in `processors.py:190-308` shows the exact datum-spec keys expected downstream — match `PreferenceDatumSpec` exactly: `message_log_chosen`, `message_log_rejected`, `length_chosen`, `length_rejected`, `loss_multiplier`, `idx`, `task_name`).

- [ ] **B2.4 Run, verify pass** (3 tests)
- [ ] **B2.5 Commit** — `git commit -m "feat(omni): views+media DPO adapter (dataset, splice processor)"`

### Task B3: Gate 1 — splice parity test

**Files:** Create `nemo_rl_apertus/tests/test_splice_parity.py`

Oracle: assemble the same sample *by hand* from the same store + the same template render, completely independently of `_splice`’s implementation:

- [ ] **B3.1 Write the test**

```python
# nemo_rl_apertus/tests/test_splice_parity.py
import torch
from transformers import AutoTokenizer
import pytest

from nemo_rl_apertus.tests.toy_store import build_toy_dataset
from nemo_rl_apertus.omni_preference import (
    OmniPreferenceDataset, omni_preference_preprocessor, IMAGE_TOKEN_ID)
from nemo_rl.data.llm_message_utils import get_formatted_message_log

TOK = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOK)


def test_gate1_splice_parity(tokenizer, tmp_path):
    meta = build_toy_dataset(tmp_path / "ds")
    ds = OmniPreferenceDataset(tmp_path / "ds", split="train")
    datum = omni_preference_preprocessor(
        ds[1], ds.task_spec, tokenizer, max_seq_length=8192, idx=1,
        media=ds.media)
    got = torch.cat([m["token_ids"] for m in datum["message_log_chosen"]])

    # ORACLE: independent reference assembly
    row = ds[1]
    messages = list(row["prompt"]) + [{"role": "assistant", "content": row["chosen"]}]
    ref_log = get_formatted_message_log(
        messages, tokenizer, ds.task_spec, add_bos_token=True, add_eos_token=True)
    ref_flat = torch.cat([m["token_ids"] for m in ref_log]).tolist()
    blocks = iter([meta["blocks"]["img1"], meta["blocks"]["img2"]])
    expected = []
    for t in ref_flat:
        if t == IMAGE_TOKEN_ID:
            expected.extend(next(blocks))
        else:
            expected.append(t)
    assert got.tolist() == expected, "bit-exact splice parity failed"
```

- [ ] **B3.2 Run, verify pass** — `... -x -q` → 1 passed (this is Gate 1 green)
- [ ] **B3.3 Commit** — `git commit -m "test(omni): gate-1 bit-exact splice parity"`

### Task B4: entrypoint

**Files:** Create `examples/run_dpo_apertus_omni.py`

- [ ] **B4.1 Implement** — copy `/opt/nemo-rl/examples/run_dpo.py` verbatim, then replace ONLY its data-setup block (the `setup_preference_data(...)` call) with:

```python
    from nemo_rl_apertus.omni_preference import build_processed_datasets
    train_dataset, val_dataset = build_processed_datasets(
        Path(config["data"]["omni_dataset_root"]),
        tokenizer,
        max_seq_length=config["data"]["max_input_seq_length"],
    )
    # val format expected by dpo.setup: dict[str, dataset]
    val_dataset = {"validation": val_dataset}
```

and pass them into the existing `dpo.setup(...)` call unchanged. Everything else (config load, cluster setup, train loop call) stays byte-identical to upstream — diff against `examples/run_dpo.py` must show only the data block.

- [ ] **B4.2 Commit** — `git commit -m "feat(omni): run_dpo_apertus_omni entrypoint (data block swap only)"`

---

## Part C — integration: real data + probe

### Task C1: produce the real store from mllm-dpo

- [ ] **C1.1** Run the producer (1 GPU, ~5,182 pairs / ≤5,182 unique images — minutes):
`mode=preference input_parquet=/capstor/store/cscs/swissai/infra01/vision-datasets/alignment-processed/mllm-dpo.parquet output_dir=/capstor/store/cscs/swissai/infra01/vision-datasets/alignment-tokenized/mllm_dpo_views_media`
- [ ] **C1.2** Gate 2 on the output: `python tests/preference/check_store.py .../mllm_dpo_views_media` → exit 0. Record `n_unique_media` vs 5,182 (dedup factor) in the run log.
- [ ] **C1.3** Decode ONE real block to text via `translate_image_to_text` (`image_only.py:369-386`) and eyeball the structure (`<|img_start|>H*W<|img_token_start|>…`).

### Task C2: Gate 3 — 3-step DPO probe on multimodal data

- [ ] **C2.1** Run the probe (same recipe as the validated text probes; seq 8192 for image headroom):

```bash
cd /opt/nemo-rl && PYTHONPATH=$BRIDGE/src:$XIELU/site-v060:/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL \
uv run --locked python /iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/examples/run_dpo_apertus_omni.py \
  --config examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-megatron.v2.yaml \
  policy.model_name=/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200 \
  policy.tokenizer.name=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611 \
  +data.omni_dataset_root=/capstor/store/cscs/swissai/infra01/vision-datasets/alignment-tokenized/mllm_dpo_views_media \
  dpo.max_num_steps=3 dpo.val_period=3 dpo.val_batches=1 \
  policy.train_global_batch_size=8 policy.max_total_sequence_length=8192 \
  policy.megatron_cfg.tensor_model_parallel_size=2 policy.megatron_cfg.sequence_parallel=false \
  checkpointing.enabled=false cluster.gpus_per_node=4 cluster.num_nodes=1
```

- [ ] **C2.2** Verify the gate: step-1 `preference_loss == 0.6931` (ln 2 exactly — policy ≡ reference holds regardless of modality), `num_valid_samples` ≥ 7/8, step-3 `loss < 6`.
- [ ] **C2.3** Record the multimodal step-1 fingerprint (loss/sft/pref) in the memory decisions file; commit any fixes; push both repos.

---

## Self-review checklist (run after writing, before handoff)

- Spec coverage: invariants 1-4 → A1/A2/A3; units+atomicity → A1; manifest → A4; marker contract → A2 (producer) + B2 `_splice` count check (consumer); IO contract → B2 reader (init-only); Gates 1/2/3 → B3/A5/C2; per-task RL views and OPD buffer are explicitly out of scope (spec reserves them).
- Known adaptation points are confined to: `smart_resize_dims` / `tokenize_images` naming (A4), `get_formatted_message_log` kwargs and dataset-constructor signature (B2), hydra entry invocation (A6) — each pinned to the ground-truth file:line to read.
