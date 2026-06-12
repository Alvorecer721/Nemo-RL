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
"""Views+media preference dataset and splice processor for NeMo-RL DPO/RM.

Seam (verified v0.6.0): ``dpo.setup()`` accepts pre-built
``AllTaskProcessedDataset`` objects whose processor is any callable invoked as
``(entry, task_data_spec, tokenizer, max_seq_length, idx)`` and returning the
``PreferenceDatumSpec`` keys (``message_log_chosen``/``_rejected``,
``length_chosen``/``_rejected``, ``loss_multiplier``, ``idx``). Downstream
(``preference_collate_fn``) only needs per-message ``token_ids`` tensors and
never re-tokenizes, so this module can splice pretokenized image blocks into
the rendered text without touching any upstream file.

Data flow per sample: the stock ``get_formatted_message_log`` templates and
tokenizes the text live — under a per-pair ``enable_thinking`` decided from the
CHOSEN response's think markers (see ``has_think_markers``); each ``<|image|>``
marker (id from the manifest token_layout) is
then replaced by the next pretokenized block from that *field's* media refs
(``prompt_media_refs`` for prompt messages, ``{side}_media_refs`` for the
response message — the chosen sequence must never consume rejected's refs).
Blocks are sliced from the media store, cast int32 -> int64 (text token_ids
are int64; the batch validator rejects mixed dtypes) into owned tensors.

Note (spec): for preference data the stock pipeline performs zero validation
between processor output and model forward — the marker-count checks here and
Gate 1 (``tests/test_splice_parity.py``) are the only integrity checks.

Worker model: dataloader workers MUST fork (Linux default) so the reader's RAM
arena / memmaps are shared copy-on-write across workers. Under a spawn start
method every worker would re-open the store (and reload the tokenizer);
``OmniPreferenceDataset`` refuses to construct — use ``data.num_workers=0``
on platforms without fork.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from contextlib import contextmanager
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset

# Module-level on purpose: this pulls upstream's full datasets chain (including
# import-order sensitive native libs such as decord) so importing this adapter
# completes ALL its imports up front. Entrypoints that import the adapter first
# and load the tokenizer afterwards are then always safe; lazy imports landing
# after a live tokenizer have been observed to crash in this environment.
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import (
    LLMMessageLogType,
    PathLike,
    PreferenceDatumSpec,
    TaskDataSpec,
    TokenizerType,
)
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl_apertus.media_store import MediaStoreReader

# No hardcoded token ids: the marker string and its id come from the store
# manifest's token_layout, derived from the tokenizer at build time.

# Mirrors _THINKING_OPEN_MARKERS in the SFT tokenization pipeline
# (vision_tokenization/discrete/sft_segments.py in benchmark-image-tokenzier)
# EXACTLY: preference rendering must key `enable_thinking` on the same opener
# set the SFT corpus was rendered with, or DPO would condition CoT pairs
# differently from their SFT counterparts. Openers only — an unclosed or
# partial trace still marks the sample as CoT; close-side markers add nothing.
THINKING_OPEN_MARKERS: tuple[str, ...] = (
    "<think>",            # Qwen 3, DeepSeek R1, prompted-style outputs
    "<|channel>thought",  # Gemma 4 native (`<|channel>thought\n…<channel|>`)
    "<thought>",          # Gemini API streamed-text mode
)


def has_think_markers(text: str) -> bool:
    """True if ``text`` carries a CoT opening marker (``THINKING_OPEN_MARKERS``)."""
    return any(marker in text for marker in THINKING_OPEN_MARKERS)


@contextmanager
def _deliberation_bound(tokenizer: TokenizerType, enable_thinking: bool):
    """Bind ``enable_thinking`` onto every chat-template render in scope.

    Mechanism — the same one NeMo-RL uses for config-level
    ``chat_template_kwargs`` (``nemo_rl/algorithms/utils.py:341-348``, locked
    v0.6.0 runtime): rebind ``tokenizer.apply_chat_template`` with
    ``functools.partial``. The stock ``get_formatted_message_log`` builds its
    template kwargs internally and cannot forward extras, so the per-pair
    value must ride the tokenizer attribute. Partial stacking resolves in our
    favor: an outer partial's kwargs override an inner (config-level) one's,
    and no stock call site passes ``enable_thinking`` itself — the value bound
    here always wins. Because the binding is uniform across every incremental
    per-turn render AND the one-shot render, Gate-1 parity (concatenated
    per-message render == one-shot template render) is preserved per mode.

    Restored on exit. Per-sample mutation of the tokenizer is safe here:
    dataloader workers are processes (fork is enforced at dataset init) and
    rendering is synchronous within a worker.
    """
    prior = tokenizer.__dict__.get("apply_chat_template")
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, enable_thinking=enable_thinking
    )
    try:
        yield
    finally:
        if prior is None:
            del tokenizer.apply_chat_template  # back to the plain class method
        else:
            tokenizer.apply_chat_template = prior


@lru_cache(maxsize=8)
def _sha256_file(path: str) -> str:
    """Cached file hash — both splits validate the same tokenizer.json."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OmniPreferenceDataset:
    """One split of a views+media dataset root, with the manifest contract enforced.

    ``manifest.json`` is the producer's commit record; a rootdir without one is
    unpublished garbage (spec invariant 4). Init validates token dtype, the
    ``files`` map (existence + byte size), and — when ``tokenizer_path`` is
    given — that the runtime tokenizer's ``tokenizer.json`` sha256 matches the
    one the store was generated with.

    All store IO happens at init (views -> arrow-backed rows; media parquets ->
    in-RAM id map; token bins -> RAM arena or memmap). ``__getitem__`` is a
    plain row lookup; block slicing happens in the processor.

    Args:
        root: Dataset root containing ``manifest.json``, ``views/``, ``media/``.
        split: View split name (``"train"`` or ``"validation"``).
        tokenizer_path: Optional tokenizer snapshot directory; enables the
            manifest sha check (the entrypoint passes it; tests pass ``None``).
    """

    def __init__(
        self,
        root: PathLike,
        split: str,
        tokenizer_path: Optional[PathLike] = None,
    ) -> None:
        if multiprocessing.get_start_method() != "fork":
            raise RuntimeError(
                "OmniPreferenceDataset requires the 'fork' multiprocessing start "
                "method so dataloader workers share the media arena copy-on-write "
                "(under 'spawn' each worker reloads the store and the tokenizer). "
                "Use data.num_workers=0 on platforms without fork."
            )
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(
                f"{root} has no manifest.json — the manifest is the commit record; "
                "a store without one is unpublished garbage (spec invariant 4)"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest["token_dtype"] != "<i4":
            raise ValueError(f"unsupported token_dtype {manifest['token_dtype']!r}")
        for rel, size in manifest["files"].items():
            try:
                actual = (root / rel).stat().st_size
            except FileNotFoundError:
                raise ValueError(f"manifest lists missing file: {rel}") from None
            if actual != size:
                raise ValueError(
                    f"size mismatch for {rel}: manifest says {size}, on disk {actual}"
                )
        if tokenizer_path is not None:
            sha = _sha256_file(str(Path(tokenizer_path) / "tokenizer.json"))
            if sha != manifest["tokenizer"]["sha256"]:
                raise ValueError(
                    f"tokenizer sha256 mismatch: store was generated with "
                    f"{manifest['tokenizer']['sha256']}, runtime tokenizer at "
                    f"{tokenizer_path} hashes to {sha}"
                )
        layout = manifest.get("token_layout")
        if layout is None:
            raise ValueError(
                f"{root}: manifest has no token_layout — stores must record "
                "their token conventions; regenerate with a current producer"
            )
        self.image_marker: str = layout["image_marker"]
        self.image_marker_id: int = int(layout["image_marker_id"])
        self.manifest = manifest
        self.media = MediaStoreReader(
            [root / r for r in manifest["media_roots"]],
            token_dtype=manifest["token_dtype"],
        )
        self.rows = load_dataset(
            "parquet", data_files=str(root / "views" / f"{split}.parquet")
        )["train"]
        self.task_spec = TaskDataSpec(task_name="omni_preference")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _splice_field(
    messages: LLMMessageLogType,
    refs: list[str],
    media: MediaStoreReader,
    marker_id: int,
) -> None:
    """Replace each marker id in one view field's messages with its blocks.

    ``refs`` are the field's media_ids in marker order. Enforces the spec
    marker contract at field granularity: marker count == len(refs), hard
    error on mismatch. Mutates ``token_ids`` in place; blocks become owned
    int64 tensors (``torch.tensor`` copies out of the arena/memmap).
    """
    ref_iter = iter(refs)
    n_spliced = 0
    for msg in messages:
        ids = msg["token_ids"]
        positions = (ids == marker_id).nonzero(as_tuple=True)[0].tolist()
        if not positions:
            continue
        pieces, prev = [], 0
        for pos in positions:
            try:
                block_ids = media.tokens(next(ref_iter))
            except StopIteration:
                raise ValueError(
                    f"marker count exceeds media refs ({len(refs)}) in field"
                ) from None
            pieces += [ids[prev:pos], torch.tensor(block_ids, dtype=torch.int64)]
            prev = pos + 1
            n_spliced += 1
        pieces.append(ids[prev:])
        msg["token_ids"] = torch.cat(pieces)
    if n_spliced != len(refs):
        raise ValueError(f"marker count {n_spliced} != media refs {len(refs)}")


def omni_preference_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
    *,
    media: MediaStoreReader,
    image_marker_id: int,
) -> PreferenceDatumSpec:
    """Render both preference sides live, then splice media blocks per field.

    Both sides render under one per-pair ``enable_thinking`` keyed on the
    chosen response's think markers (see the CONTRACT comment in the body).

    Mirrors the stock ``preference_preprocessor`` output contract exactly
    (``nemo_rl/data/processors.py``), including the overlength policy: a pair
    exceeding ``max_seq_length`` is loss-masked AND physically shrunk to the
    stock <=4-tokens-per-message placeholder — a dead full-length multimodal
    sample would otherwise dictate the padded width of the whole batch. Live
    blocks are never bisected; placeholder truncation only happens on dead
    samples.
    """
    prompt_refs = list(datum_dict["prompt_media_refs"] or [])
    output: dict[str, Any] = {
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": "omni_preference",
    }
    # CONTRACT (per-pair deliberation): ``enable_thinking`` is decided ONCE per
    # pair, keyed on the CHOSEN response only, and conditions BOTH sides'
    # renders — prompt conditioning stays identical across the pair, so the
    # DPO logprob difference is attributable to the responses alone. A rejected
    # response carrying think markers under ``Deliberation: disabled`` is
    # intentional negative signal (thinking when not asked), not an error.
    enable_thinking = has_think_markers(datum_dict["chosen"])
    with _deliberation_bound(tokenizer, enable_thinking):
        for side in ("chosen", "rejected"):
            messages = list(datum_dict["prompt"]) + [
                {"role": "assistant", "content": datum_dict[side]}
            ]
            message_log = get_formatted_message_log(
                messages,
                tokenizer,
                task_data_spec,
                add_bos_token=True,
                add_eos_token=True,
            )
            # Per-field refs: prompt markers consume prompt_media_refs; the
            # response message consumes only this side's refs (empty today —
            # image-bearing completions work the day they exist). This also
            # re-checks, in token space, that responses carry no accidental
            # markers.
            side_refs = list(datum_dict.get(f"{side}_media_refs") or [])
            _splice_field(message_log[:-1], prompt_refs, media, image_marker_id)
            _splice_field(message_log[-1:], side_refs, media, image_marker_id)
            output[f"message_log_{side}"] = message_log
            output[f"length_{side}"] = sum(len(m["token_ids"]) for m in message_log)

    if max(output["length_chosen"], output["length_rejected"]) > max_seq_length:
        for side in ("chosen", "rejected"):
            message_log = output[f"message_log_{side}"]
            cap = min(4, max_seq_length // len(message_log))  # stock divisor
            for msg in message_log:
                msg["token_ids"] = msg["token_ids"][:cap]
            output[f"length_{side}"] = sum(len(m["token_ids"]) for m in message_log)
        output["loss_multiplier"] = 0.0

    return output


def build_processed_datasets(
    root: PathLike,
    tokenizer: TokenizerType,
    max_seq_length: int,
    tokenizer_path: Optional[PathLike] = None,
) -> tuple[AllTaskProcessedDataset, AllTaskProcessedDataset]:
    """Build (train, validation) ``AllTaskProcessedDataset`` for ``dpo.setup()``.

    Args:
        root: Views+media dataset root (contains ``manifest.json``).
        tokenizer: The runtime tokenizer (templates + tokenizes view text live).
        max_seq_length: ``data.max_input_seq_length``; pairs longer than this
            are loss-masked and shrunk by the processor.
        tokenizer_path: Optional tokenizer snapshot dir to enforce the manifest
            sha check (pass it in entrypoints; ``None`` skips the check).
    """

    def _make(split: str) -> AllTaskProcessedDataset:
        ds = OmniPreferenceDataset(root, split, tokenizer_path=tokenizer_path)
        return AllTaskProcessedDataset(
            ds,
            tokenizer,
            ds.task_spec,
            partial(
                omni_preference_preprocessor,
                media=ds.media,
                image_marker_id=ds.image_marker_id,
            ),
            max_seq_length=max_seq_length,
        )

    return _make("train"), _make("validation")
