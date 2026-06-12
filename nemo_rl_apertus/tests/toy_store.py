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
"""Builds a tiny valid dataset root (manifest + views + media) with SYNTHETIC blocks.

Blocks are structurally valid (``<|img_start|>``/.../``<|img_end|>`` with a
vision-range body) so the adapter exercises every contract without a GPU or the
vision tokenizer. The manifest carries ``files: {}`` — an empty files map skips
the existence+size checks in the toy path (real stores list every file).

View rows:
  0: single image in the prompt.
  1: two images in the prompt (marker order matters).
  2: one image in the prompt AND one in the chosen response (per-field refs:
     the rejected side must not consume chosen's block).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def synthetic_block(h_tok: int, w_tok: int, seed: int) -> np.ndarray:
    """Build a structurally valid encapsulated image block of h_tok x w_tok tokens."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(h_tok):
        rows.append(rng.integers(131272, 262344, size=w_tok, dtype=np.int32))
        rows.append(np.array([131076], dtype=np.int32))  # <|img_eol|>
    body = np.concatenate(rows)
    return np.concatenate(
        [
            np.array([131073, 131075], dtype=np.int32),  # img_start, img_token_start
            body,
            np.array([131077, 131074], dtype=np.int32),  # img_end_of_frame, img_end
        ]
    )


def build_toy_dataset(root: Path, tokenizer_sha: str = "") -> dict[str, Any]:
    """Write a complete toy dataset root; return {"ids": ..., "blocks": ...} metadata."""
    media = root / "media"
    media.mkdir(parents=True)
    views = root / "views"
    views.mkdir()

    blocks = {f"img{i}": synthetic_block(4, 8, i) for i in range(3)}
    ids = {k: hashlib.sha256(k.encode()).hexdigest() for k in blocks}

    rows, t_off, r_off = [], 0, 0
    with (
        open(media / "tokens.000000.bin", "wb") as tok_f,
        open(media / "raw.000000.bin", "wb") as raw_f,
    ):
        for k, b in blocks.items():
            tok_f.write(b.tobytes())
            raw_f.write(k.encode())
            rows.append(
                {
                    "media_id": ids[k],
                    "shard": 0,
                    "offset_elems": t_off,
                    "length_elems": len(b),
                    "raw_offset_bytes": r_off,
                    "raw_length_bytes": len(k),
                    "raw_ext": "bin",
                    "resize_h": 64,
                    "resize_w": 128,
                    "kind": "image",
                    "source": k,
                }
            )
            t_off += len(b)
            r_off += len(k)
    pq.write_table(pa.Table.from_pylist(rows), media / "media.000000.parquet")

    view_rows = [
        {
            "prompt": [{"role": "user", "content": "<|image|>\nDescribe."}],
            "chosen": "A good description.",
            "rejected": "A bad one.",
            "prompt_media_refs": [ids["img0"]],
            "chosen_media_refs": [],
            "rejected_media_refs": [],
            "prompt_id": "t0",
            "media_tokens_total": int(len(blocks["img0"])),
            "text_chars": 40,
        },
        {
            "prompt": [
                {"role": "user", "content": "<|image|>\nand\n<|image|>\nCompare."}
            ],
            "chosen": "First is better.",
            "rejected": "Second.",
            "prompt_media_refs": [ids["img1"], ids["img2"]],
            "chosen_media_refs": [],
            "rejected_media_refs": [],
            "prompt_id": "t1",
            "media_tokens_total": int(len(blocks["img1"]) + len(blocks["img2"])),
            "text_chars": 50,
        },
        {
            "prompt": [{"role": "user", "content": "<|image|>\nEdit this image."}],
            "chosen": "Here you go: <|image|>",
            "rejected": "I cannot do that.",
            "prompt_media_refs": [ids["img0"]],
            "chosen_media_refs": [ids["img1"]],
            "rejected_media_refs": [],
            "prompt_id": "t2",
            "media_tokens_total": int(len(blocks["img0"])),
            "text_chars": 60,
        },
    ]
    pq.write_table(pa.Table.from_pylist(view_rows), views / "train.parquet")
    pq.write_table(pa.Table.from_pylist(view_rows[:1]), views / "validation.parquet")

    manifest = {
        "schema_version": 1,
        "tokenizer": {"path": "", "sha256": tokenizer_sha},
        "vision_tokenizer": {"version": "toy"},
        "token_dtype": "<i4",
        "token_layout": {
            "image_marker": "<|image|>", "image_marker_id": 131079,
            "img_start": 131073, "img_end": 131074, "img_token_start": 131075,
            "eol": 131076, "eof": 131077, "vision_lo": 131272, "vision_hi": 262343,
        },
        "expected_min_model_vocab": 266440,
        "media_roots": ["media/"],
        "store_raw": True,
        "files": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return {"ids": ids, "blocks": {k: b.tolist() for k, b in blocks.items()}}
