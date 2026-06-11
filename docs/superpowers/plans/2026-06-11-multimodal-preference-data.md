# Multimodal Preference Data (views+media) Implementation Plan — FINAL (post 4-reviewer verification)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved views+media preference data format end to end: a `preference` mode in `vision_tokenization` that freezes images into a content-addressed media store, an additive NeMo-RL adapter that splices blocks at `<|image|>` (id 131079), and a green 3-step DPO probe on the real `mllm-dpo.parquet` (5,182 pairs).

**Architecture:** Producer (Part A, in `benchmark-image-tokenizer/vision_tokenization`) ingests preference parquet → hashes/dedups images → GPU-encodes unique media at exact smart_resize dims → writes sealed media triples + view parquets + manifest commit record. Consumer (Part B, additive files in this repo) loads views via the stock loader, validates the manifest, and splices blocks into per-message `token_ids` inside a custom processor behind the public `dpo.setup()` seam. Part C converts the real dataset and runs the probe.

**Verification status:** every integration point below was checked against real code by 4 reviewers (2026-06-11): signatures confirmed live on `/opt/nemo-rl` v0.6.0; producer seams corrected against `vision_tokenization` source. Algorithm sweep verdicts: offline DPO ✓; RM training ✓ (same data path, zero extra work); GRPO ✓ (same direct-`setup()` pattern; spliced ids survive `prompt_token_ids` end-to-end, no decode/re-encode points); OPD ✓ (additionally requires teacher vocab ⊇ media id range); image-*generating* rollouts are the named P4 boundary (env/judge decode paths).

**Spec:** `docs/design-docs/apertus-multimodal-preference-data.md`. Normative: units in token elements; `media_id = sha256(raw bytes)` full hex; manifest-last atomic write; per-field refs; never bisect a live block; views carry no system messages.

**Repos:**
- `VT` = `/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier` (producer; direct commits)
- `RL` = `/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL` (consumer; additive files only)
- Runtime env for anything importing `nemo_rl` or the tokenizer: `cd /opt/nemo-rl && uv run --locked ...`

**Execution notes:** Parts A and B independent (B uses a synthetic toy store; no GPU before A6/C1). Tokenizer loads ~200 s — real-tokenizer tests are module-scoped and batched. Commit per task. Scale note: this plan targets mllm-dpo scale (5k); the MMPR-scale extension (resume via seal-every-K-batches + plan fingerprint + streaming ingest; `store_raw` opt-out; `MADV_RANDOM`; `lfs setstripe -c 4`) is **deliberately deferred** — designed in the spec, not implemented here.

---

## Part A — producer: `preference` mode in `vision_tokenization`

File map (paths relative to `VT/vision_tokenization/`):
- Create: `pipeline/output/media_store.py`, `indexing/preference/{__init__,ingest,planning}.py`, `pipeline/runtime/preference_runner.py`
- Create: `tests/preference/__init__.py` + test files (NOTE: `pytest.ini` sets `testpaths = vision_tokenization`, so tests live at `vision_tokenization/tests/preference/` — NOT repo-root `tests/`; run from repo root so the suite auto-collects them)
- Modify: `discrete/emu/__init__.py` (factory: add `preference` branch), `tokenize.py:31` (`_VALID_MODES`) + error strings at `tokenize.py:70/74`, `pipeline/__init__.py` (mode branch — NOT executor.py), `configs/config.yaml:21` (mode comment)
- Create: `configs/dataset/_task/preference.yaml`, `configs/dataset/preference/mllm_dpo_smoke.yaml`, `configs/dataset/preference/mllm_dpo.yaml`

### Task A1: MediaStoreWriter + MediaStoreReader

Code as previously planned (verified sound), with one addition: an `_atomic_write_json` helper shared by the manifest writer.

**Files:** Create `pipeline/output/media_store.py`, `tests/preference/test_media_store.py` (tests verbatim from prior plan version — roundtrip, element-units, duplicate-id rejection, dtype refusal)

- [ ] **A1.1** Write the four failing tests (`vision_tokenization/tests/preference/test_media_store.py`; add `__init__.py` beside it)
- [ ] **A1.2** Run: `cd VT && python -m pytest vision_tokenization/tests/preference/test_media_store.py -x -q` → fails (no module)
- [ ] **A1.3** Implement `media_store.py` — `MediaStoreWriter` (`add(media_id, *, tokens, raw, resize_h, resize_w, kind, source, raw_ext)`, dup-id ValueError, `seal()` → fsync bins → atomic parquet → returns `{relpath: byte_size}`), `MediaStoreReader(roots, token_dtype="<i4", load_to_ram_threshold_bytes=16<<30)` (refuses non-`<i4`; RAM arena below threshold else memmap; union index; `tokens(id)`, `raw(id)`), plus:

```python
def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
```

- [ ] **A1.4** Tests pass → **A1.5** Commit `feat(preference): content-addressed media store (sealed triples, element units)`

### Task A2: ingest — hash, dedup, marker validation, view drafting

**Files:** Create `indexing/preference/__init__.py`, `indexing/preference/ingest.py`; Test `tests/preference/test_ingest.py`

Code as previously planned (tests + implementation verified against the real mllm-dpo schema), unchanged contracts: sha256 full-hex dedup, `<image>`→`<|image|>` normalization, per-row marker-count == image-count (`MarkerMismatch`), accidental-marker rejection in responses, **system-role rejection** (80/20 retired permanently).

- [ ] **A2.1** Write the 4 failing tests → **A2.2** verify fail → **A2.3** implement → **A2.4** pass → **A2.5** Commit
- [ ] **A2.6** Pre-C1 audit (container env — head-node pyarrow hits a row-group quirk on this file): `pq.read_table(mllm-dpo.parquet, columns=['prompt']).to_pylist()`, count `role=='system'` rows. Expectation: zero (prompts are user-only). If nonzero: decide strip-vs-drop explicitly and update the spec line before C1.

### Task A3: exact-dims planning over unique media

**Files:** Create `indexing/preference/planning.py`; Test `tests/preference/test_planning.py`

Code as previously planned (verified) — `plan_exact_dim_batches(dims, batch_size)` groups by exact `(h, w)`, chunks, stragglers keep exact dims (NO `_pack_spillover`). Dims source correction: the production rounding is the free function
`vision_tokenization/utils/image_geometry.py:118 smart_resize_dims(height, width, *, min_pixels, max_pixels, factor)` — the runner calls it directly (factor=16) so the band declared in config and the band recorded in the manifest are one source.

- [ ] **A3.1-A3.5** TDD as planned; commit `feat(preference): exact-dims media batching (no spillover means)`

### Task A4: factory branch, mode wiring, preference runner

**Files:** Modify `discrete/emu/__init__.py`, `tokenize.py`, `pipeline/__init__.py`, `configs/config.yaml`; Create `pipeline/runtime/preference_runner.py`, `configs/dataset/_task/preference.yaml`, `configs/dataset/preference/mllm_dpo_smoke.yaml`

- [ ] **A4.1** Factory: in `discrete/emu/__init__.py` dispatch (lines 25-40) add `elif mode == "preference": tokenizer_class = EMUImageOnlyTokenizer` (it swallows the mode kwarg via `**kwargs`, `image_only.py:34`); keep the ValueError listing updated. In `tokenize.py:31` add `"preference"` to `_VALID_MODES` (+ error strings at :70/:74, `configs/config.yaml:21` comment).

- [ ] **A4.2** Mode branch goes in **`pipeline/__init__.py` `run_distributed_pipeline`** — after output_dir namespacing (line 109) and `torch.cuda.set_device` (line 111), BEFORE the executor import (line 118). The executor path is unusable: its plan load (`executor.py:239`) requires a manifest that preference mode doesn't have.

```python
    if cfg["mode"] == "preference":
        if cfg["world_size"] != 1:
            raise RuntimeError(f"preference mode is single-rank; got world_size={cfg['world_size']}")
        from .runtime.preference_runner import run_preference_mode
        return run_preference_mode(cfg)
```

- [ ] **A4.3** Configs. `configs/dataset/_task/preference.yaml`:

```yaml
# Task fragment for preference-mode datasets (composed via defaults).
tokenizer_path: /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed
# Resize band ALIGNED WITH PRETRAIN/SFT (user decision 2026-06-11): post-training
# never silently changes the vision front-end. 1400^2 -> <=~7.7k tokens/image;
# multimodal DPO runs use max_total_sequence_length 16384. Recorded in manifest.json.
min_pixels: "128*128"
max_pixels: "1400*1400"
encode_batch_size: 32
val_rows: 256
```

`configs/dataset/preference/mllm_dpo_smoke.yaml` (and `mllm_dpo.yaml` analog with the real input + capstor output):

```yaml
defaults:
  - /dataset/_pipeline@_here_
  - /dataset/_task/preference@_here_
output_name: mllm_dpo_smoke
output_dir: ???
input_parquet: ???
```

(The `_pipeline` fragment satisfies `tokenize.py`'s unconditional `cfg.dataset.min_pixels/max_pixels` reads; the unused manifest/plan keys are never read once the branch lives in `pipeline/__init__.py`. Note the final dataset root is namespaced: `<output_dir>/preference/<output_name>/`.)

- [ ] **A4.4** Implement `pipeline/runtime/preference_runner.py`:

```python
# pipeline/runtime/preference_runner.py
"""`preference` mode: freeze media for preference/RL datasets (views+media spec).

Single-rank. Writes <root>/media/ (sealed triple via MediaStoreWriter),
<root>/views/{train,validation}.parquet, then manifest.json LAST (commit record),
where <root> = cfg["output_dir"] (already namespaced .../preference/<output_name>).
"""
from __future__ import annotations

import hashlib
import io
import logging
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from vision_tokenization.indexing.preference.ingest import ingest_preference_parquet
from vision_tokenization.indexing.preference.planning import plan_exact_dim_batches
from vision_tokenization.pipeline.output.media_store import (
    MediaStoreWriter, atomic_write_json,
)
from vision_tokenization.utils.image_geometry import smart_resize_dims

logger = logging.getLogger(__name__)
SPATIAL_FACTOR = 16


def run_preference_mode(cfg: dict) -> None:
    from vision_tokenization.discrete.emu import create_tokenizer

    out = Path(cfg["output_dir"])
    res = ingest_preference_parquet(Path(cfg["input_parquet"]))

    tokenizer = create_tokenizer(
        mode="preference",
        text_tokenizer_path=cfg["tokenizer_path"],
        device=f"cuda:{cfg['local_rank']}",
        min_pixels=cfg["tokenizer_min_pixels"],
        max_pixels=cfg["tokenizer_max_pixels"],
        max_encode_pixels=cfg.get("max_encode_pixels"),
        **(cfg.get("tokenizer_kwargs", {})),
    )

    # exact smart-resize dims per unique image; skip-and-count degenerate images
    dims, skipped = [], set()
    for i, um in enumerate(res.unique_media):
        with Image.open(io.BytesIO(um.raw)) as im:
            w, h = im.size
        if h < SPATIAL_FACTOR or w < SPATIAL_FACTOR:
            skipped.add(um.media_id)
            continue
        rh, rw = smart_resize_dims(
            h, w, min_pixels=cfg["tokenizer_min_pixels"],
            max_pixels=cfg["tokenizer_max_pixels"], factor=SPATIAL_FACTOR)
        dims.append((i, rh, rw))
    if skipped:
        logger.warning("skipping %d sub-%dpx images (and their pairs)",
                       len(skipped), SPATIAL_FACTOR)

    writer = MediaStoreWriter(out / "media")
    for batch in plan_exact_dim_batches(dims, batch_size=cfg["encode_batch_size"]):
        images = [Image.open(io.BytesIO(res.unique_media[i].raw)).convert("RGB")
                  for i in batch.member_indices]
        # [B, L] int64 CPU, rows INCLUDE outer BOS/EOS (encapsulate_batch);
        # tokenize_images is already @torch.inference_mode-decorated.
        batched = tokenizer.tokenize_images(
            images, (batch.resize_height, batch.resize_width))
        for i, row in zip(batch.member_indices, batched):
            um = res.unique_media[i]
            assert int(row[0]) == tokenizer.bos_id and int(row[-1]) == tokenizer.eos_id
            block = row[1:-1].numpy().astype(np.int32)   # <|img_start|>...<|img_end|>
            writer.add(um.media_id, tokens=block, raw=um.raw,
                       resize_h=batch.resize_height, resize_w=batch.resize_width,
                       kind="image", source=um.source, raw_ext=um.raw_ext)
    media_files = writer.seal()

    # views: drop rows referencing skipped media; exact media token stats
    length_of = {r["media_id"]: r["length_elems"] for r in
                 pq.read_table(out / "media" / "media.000000.parquet").to_pylist()}
    rows = []
    for row in res.view_rows:
        if any(m in skipped for m in row["prompt_media_refs"]):
            continue
        row["media_tokens_total"] = sum(length_of[m] for m in row["prompt_media_refs"])
        row["text_chars"] = (sum(len(m["content"]) for m in row["prompt"])
                             + len(row["chosen"]) + len(row["rejected"]))
        rows.append(row)

    rng = random.Random(42)
    rng.shuffle(rows)
    n_val = min(cfg["val_rows"], max(1, len(rows) // 50))
    (out / "views").mkdir(parents=True, exist_ok=True)
    view_files = {}
    for name, part in (("validation", rows[:n_val]), ("train", rows[n_val:])):
        p = out / "views" / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(part), p)
        view_files[f"views/{name}.parquet"] = p.stat().st_size

    tok_sha = hashlib.sha256(
        (Path(cfg["tokenizer_path"]) / "tokenizer.json").read_bytes()).hexdigest()
    atomic_write_json(out / "manifest.json", {
        "schema_version": 1,
        "tokenizer": {"path": cfg["tokenizer_path"], "sha256": tok_sha},
        "vision_tokenizer": {"version": "Emu3.5",
                             "min_pixels": cfg["tokenizer_min_pixels"],
                             "max_pixels": cfg["tokenizer_max_pixels"]},
        "token_dtype": "<i4",
        "expected_min_model_vocab": 266440,
        "media_roots": ["media/"],
        "store_raw": True,
        "files": {**{f"media/{k}": v for k, v in media_files.items()}, **view_files},
        "source_input": str(cfg["input_parquet"]),
        "n_pairs": len(rows),
        "n_unique_media": len(dims),
        "n_skipped_media": len(skipped),
    })
```

- [ ] **A4.5** Commit `feat(preference): mode wiring + runner (exact-dim encode -> media store + views + manifest)`

### Task A5: Gate 2 — block integrity checker (verifies the exact-dims contract)

**Files:** Create `tests/preference/check_store.py`

- [ ] **A5.1** Implement with argparse and the **dims cross-check** (this is what makes Gate 2 protect invariant 3): load `{media_id: (resize_h, resize_w, length_elems)}` from the media parquets; per sampled block assert: `b[0]==131073 and b[-1]==131074`; exactly one 131075 and one 131077; `h_rows := (b==131076).sum() == resize_h // 16`; `len(vis) == (resize_h//16) * (resize_w//16)` with `vis = b[(b>=131272)&(b<=262343)]`; and the body region (after the first 131075, before the trailing `[131077, 131074]`) contains ONLY vision-range ids and 131076 (the dims header before `img_token_start` is the only legal place for text ids). Path fix: `sys.path.insert(0, str(Path(__file__).resolve().parents[3]))` (file sits at `vision_tokenization/tests/preference/`).
- [ ] **A5.2** Commit `test(preference): gate-2 checker verifies dims contract`

### Task A6: end-to-end producer smoke (16 rows)

- [ ] **A6.1** Build `/tmp/pref_smoke/in.parquet` (head 16 rows of mllm-dpo, real bytes). Launch with the repo's real convention (`PYTHONPATH` export + `python -m vision_tokenization.tokenize`, cf. `scripts/slurm/tokenize/*/*.slurm`):

```bash
cd VT && PYTHONPATH=$PWD:$PYTHONPATH python -m vision_tokenization.tokenize \
  mode=preference dataset=preference/mllm_dpo_smoke \
  dataset.input_parquet=/tmp/pref_smoke/in.parquet \
  dataset.output_dir=/tmp/pref_smoke/out num_gpus=1
```

- [ ] **A6.2** Gate 2 on the **namespaced** root: `python vision_tokenization/tests/preference/check_store.py /tmp/pref_smoke/out/preference/mllm_dpo_smoke` → exit 0; manifest sane (`n_pairs<=16`, `n_unique_media<=16`, nonzero tokenizer sha).
- [ ] **A6.3** Commit fixes.

---

## Part B — consumer: additive NeMo-RL adapter

File map: `nemo_rl_apertus/{__init__,media_store,omni_preference}.py`, `examples/run_dpo_apertus_omni.py`, `nemo_rl_apertus/tests/{toy_store,test_processor,test_splice_parity}.py`, `nemo_rl_apertus/tests/fixtures/golden_store/` (committed, few KB).
All commands: `cd /opt/nemo-rl && PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL uv run --locked python -m pytest <abs path> -x -q`.
Verified live (v0.6.0): `get_formatted_message_log(messages, tokenizer, task_data_spec, add_bos_token=True, add_eos_token=True)` ✓; `TaskDataSpec(task_name=...)` ✓; `AllTaskProcessedDataset(dataset, tokenizer, task_spec, processor, max_seq_length=...)` with a single callable invoked `(entry, spec, tokenizer, max_seq_length, idx)` ✓; datum keys `message_log_chosen/_rejected, length_chosen/_rejected, loss_multiplier, idx` ✓ (`task_name` extra but harmless); `+data.omni_dataset_root` override survives config validation ✓; dataloader workers fork (py3.13 Linux) → arena/views are COW-shared ✓.

### Task B1: toy store fixture + golden-store conformance

- [ ] **B1.1** `nemo_rl_apertus/tests/toy_store.py` as previously planned (synthetic structurally-valid blocks; manifest with `files: {}` — empty `files` map skips size checks in the toy path).
- [ ] **B1.2** **Golden-store conformance**: once Task A1 lands, generate a tiny store with VT's real `MediaStoreWriter` (3 synthetic blocks), commit it under `nemo_rl_apertus/tests/fixtures/golden_store/`, and add `test_golden_store_roundtrip` opening it with the **copied** reader, asserting tokens/raw/units round-trip. This fixture is the cross-repo sync contract: regenerate it on any `schema_version` bump. (The duplicated ~80-line reader is deliberate — the inter-repo contract is the on-disk format, not a shared Python package; VT will be extracted to its own repo and the RL side runs a locked env.)
- [ ] **B1.3** Commit.

### Task B2: media store reader + dataset + splice processor

- [ ] **B2.1** `nemo_rl_apertus/media_store.py`: copy of A1's `MediaStoreReader` (same tests).
- [ ] **B2.2** Failing tests (as previously planned: marker consumed + block contiguous in chosen sequence + int64; multi-image order; overlength dies small) — module-scoped real-tokenizer fixture.
- [ ] **B2.3** Implement `nemo_rl_apertus/omni_preference.py` with these review-mandated deltas from the prior version:
  - `OmniPreferenceDataset.__init__` enforces **the full manifest contract**: dtype check; `for rel, size in manifest["files"].items(): assert (root/rel).exists() and st_size == size` (raise ValueError naming the file); tokenizer sha check — `hashlib.sha256(Path(tokenizer_path)/'tokenizer.json').read_bytes()).hexdigest() == manifest["tokenizer"]["sha256"]` when the caller passes `tokenizer_path` (the entrypoint does; tests pass `tokenizer_path=None` to skip); expose `self.manifest`.
  - Splice refs are **per-field**: `refs = list(datum["prompt_media_refs"]) + list(datum.get(f"{side}_media_refs") or [])` per side — today's empty side-lists are a no-op; image-bearing completions work the day they exist.
  - Overlength shrink uses the **stock divisor**: `cap = min(4, max_seq_length // len(log)); m["token_ids"] = m["token_ids"][:cap]` per message, then `loss_multiplier = 0.0`.
  - Docstring + init assert: `multiprocessing.get_start_method() == "fork"` (under spawn the arena/tokenizer reload per worker; set `data.num_workers=0` instead).
- [ ] **B2.4** Tests pass → **B2.5** Commit.

### Task B3: Gate 1 — splice parity with independent checks

- [ ] **B3.1** Test as previously planned (oracle: independent reference assembly — render via `get_formatted_message_log`, replace marker ids with blocks by hand, compare bit-exact), **plus two oracle-independent assertions in the same test** (kills the common-mode failure with the SUT, tokenizer already loaded):
  1. `ids = tokenizer.apply_chat_template(messages, tokenize=True); assert ids.count(131079) == 2` — re-verifies marker atomicity against the *current* snapshot on every run, bypassing `get_formatted_message_log` entirely;
  2. the concatenated per-message render equals the one-shot `apply_chat_template(messages, tokenize=False)` string (catches incremental-render drift).
- [ ] **B3.2** Pass → **B3.3** Commit `test(omni): gate-1 splice parity + independent template checks`.

### Task B4: entrypoint

- [ ] **B4.1** Copy `/opt/nemo-rl/examples/run_dpo.py` verbatim; replace ONLY the `setup_preference_data(...)` block with:

```python
    from transformers import AutoConfig
    from nemo_rl_apertus.omni_preference import build_processed_datasets

    train_dataset, val_dataset = build_processed_datasets(
        config["data"]["omni_dataset_root"],            # no Path() — run_dpo.py has no pathlib import
        tokenizer,
        max_seq_length=config["data"]["max_input_seq_length"],
        tokenizer_path=config["policy"]["tokenizer"]["name"],   # enables the sha check
    )
    vocab = AutoConfig.from_pretrained(config["policy"]["model_name"]).vocab_size
    need = train_dataset.dataset.manifest["expected_min_model_vocab"]
    assert vocab >= need, f"model vocab {vocab} < required {need}"
    val_dataset = {"default": val_dataset}   # stock key — keeps metric 'validation-default_loss' and checkpointing.metric_name working
```

Diff against upstream `run_dpo.py` must show only this block.
- [ ] **B4.2** Commit.

---

## Part C — integration: real data + probe

### Task C1: produce the real store

- [ ] **C1.1** `dataset=preference/mllm_dpo` (input = `.../alignment-processed/mllm-dpo.parquet`, output_dir = `.../alignment-tokenized/mllm_dpo_views_media`); **real root** = `.../mllm_dpo_views_media/preference/mllm_dpo` — use this path in all of Part C.
- [ ] **C1.2** Gate 2 → exit 0. Record dedup factor (`n_unique_media` vs 5,182) and `n_skipped_media`.
- [ ] **C1.3** Decode one real block via `translate_image_to_text` (`image_only.py:369-386`); eyeball `<|img_start|>H*W<|img_token_start|>…`.
- [ ] **C1.4** Length-budget report: pyarrow over `views/train.parquet` — distribution of `media_tokens_total + text_chars/3.5` vs 16384; expectation ≥99% under budget (band 1400² → ≤~7.7k tokens/image). Also report native-resolution histogram (how many images the cap actually touched).

### Task C2: Gates — content audit, then probe

- [ ] **C2.0 Content audit (the REAL splice gate — ln 2 cannot catch content bugs).** No GPU. Instantiate `OmniPreferenceDataset` on the C1 root, process ~32 datums through `omni_preference_preprocessor` with the real tokenizer, assert per datum: (a) zero residual id-131079 in both message logs; (b) for each ref **in order**, the slice between consecutive `<|img_start|>`/`<|img_end|>` boundaries equals `media.tokens(ref)` exactly (`np.array_equal`); (c) stripping all block spans, the remaining ids decode to text containing no vision-range ids and matching the view row's prompt/response text. Hard exit on any failure.
- [ ] **C2.1** 3-step probe (same shape as the validated text probes; note `+data.omni_dataset_root` and seq 16384):

```bash
cd /opt/nemo-rl && PYTHONPATH=$BRIDGE/src:$XIELU/site-v060:/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL \
uv run --locked python /iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/examples/run_dpo_apertus_omni.py \
  --config examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-megatron.v2.yaml \
  policy.model_name=/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200 \
  policy.tokenizer.name=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611 \
  +data.omni_dataset_root=/capstor/store/cscs/swissai/infra01/vision-datasets/alignment-tokenized/mllm_dpo_views_media/preference/mllm_dpo \
  dpo.max_num_steps=3 dpo.val_period=3 dpo.val_batches=1 \
  policy.train_global_batch_size=8 policy.max_total_sequence_length=16384 \
  policy.megatron_cfg.tensor_model_parallel_size=2 policy.megatron_cfg.sequence_parallel=false \
  checkpointing.enabled=false cluster.gpus_per_node=4 cluster.num_nodes=1
```

- [ ] **C2.2** Gate: step-1 `preference_loss == 0.6931` (policy ≡ reference, modality-independent); `num_valid_samples ≥ 7/8`; step-3 `loss < 6`. (Content correctness is C2.0's job, not this gate's.)
- [ ] **C2.3** Record the multimodal step-1 fingerprint in the memory decisions file; push both repos.

---

## Deferred (designed, deliberately not in this plan)
- MMPR-scale producer extension: resume (seal a triple every K batches + plan fingerprint + the existing `save_checkpoint` convention), streaming ingest (no `to_pylist()`/full-RAM raw retention), `store_raw: false` config path, `MADV_RANDOM` on memmap, `lfs setstripe -c 4` provisioning.
- RL-prompt views (GRPO feeding), OPD round buffers, audio blocks — spec-reserved.
- Image-generating rollouts: env/judge decode paths are the named boundary (`rollouts.py:97` batch_decode, `:502` env-obs tokenization without template); requires token_ids-fed envs + the session-scoped vision-tokenizer service.

## Self-review (post-correction)
- All 9 blockers from the 4-reviewer pass are incorporated (factory branch, smart_resize source, BOS/EOS strip, `pipeline/__init__.py` branch, CLI/hydra invocation, namespaced output root, Path import, manifest validation, val key "default").
- Spec contracts now each have an implementing step: units (A1), identity+dedup (A2), exact dims (A3 + Gate 2 dims check), manifest commit record + files/sha/vocab verification (A4 + B2 + B4), per-field refs (B2), no-system-messages (A2), length budget (A4 band + C1.4), content gate (C2.0).
- Known remaining adaptation risk: none flagged as guesswork; every API call cites verified source.
