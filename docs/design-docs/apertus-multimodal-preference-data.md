# Apertus Multimodal Preference Data: Pretokenization Format and Pipeline Design

**Status**: IMPLEMENTED & PROVEN · 2026-06-12 — producer = `posttraining` mode
(CLI name; internal modules keep `alignment` naming) unified into the one
executor path; consumer = `nemo_rl_apertus` adapter. End-to-end probe green:
step-1 preference_loss = ln 2 exact on spliced image-text sequences at seq
16384, 8/8 valid. Canonical store: vision-datasets/tokenized/preference/mllm_dpo
(task-keyed roots: tokenized/preference/, tokenized/rl/). All gates green:
Gate 1 parity, Gate 2 64/64, source audit 5,182/5,182, C2.0 64/64, suites
433+19, regression canary (resume/spill/merge) clean.
**Scope**: storage format + producer (`vision_tokenization` `posttraining` mode, new) + consumer (NeMo-RL adapter, new) for DPO/RL preference data, text-only today and multi-image later (audio reserved).

## Assumptions

- Apertus multimodality is discrete-token (Emu3.5 vision / wavtok audio): media are
  ordinary vocab ids (vision 131272–262343, audio 262344–266439), not pixel tensors.
- Runtime is NeMo-RL with the Megatron backend; text tokenization stays inside
  NeMo-RL (measured ~684 samples/s ⇒ ~12 min/epoch for 2×245k sequences — and online
  methods sample + template text live, so frozen text would be unusable there).
- The vision tokenizer is GPU inference and deterministic *for a fixed planned
  resolution* ⇒ media blocks are the textbook pretokenization target.
- Verified empirically (2026-06-11, v0.6.0 runtime): the literal `<|image|>` in
  message content survives `apply_chat_template` → encode as exactly one id
  131079; spliced block tensors pass `assert_no_double_bos` and the batch dtype
  validator; vLLM generation accepts raw `prompt_token_ids`.

## No backward compatibility

- The existing text-only `MaxMin_…-binpref` arrow dataset remains valid for the
  text-only era but is NOT this format; new multimodal sets use only the layout
  below. No shims between the two.
- The mode (CLI: `posttraining`, task-keyed) is implemented per this design. No attempt is
  made to read `sft`-mode `.bin/.idx` output for RL.
- Audio is **permanently out of scope** (user decision 2026-06-12): audio data
  prep lives in the separate audio pipeline (`benchmark-audio-tokenizer`). This
  format defines `kind="image"` only; producers MUST NOT write any other kind.
  Fail loud.

## Decision summary (the four invariants)

1. **Text live, media frozen.** Views store text as messages/strings; NeMo-RL
   templates + tokenizes at `__getitem__`. Media blocks are GPU-encoded once,
   upstream, and stored as raw token-id arrays.
2. **media_id = content identity only**: `sha256(source_file_bytes).hexdigest()`
   — full 64 hex chars, never truncated (dedup makes collisions data corruption,
   not inconvenience). Tokenizer/codebook version is **store-generation** state
   pinned in `manifest.json`, NOT part of the id — a codebook upgrade produces a
   new media store with identical media_ids and leaves `views/` untouched.
3. **Blocks are canonical per store generation**: dedup ⇒ one encode per
   `media_id`; deterministic plan ⇒ the same input set reproduces a
   bit-identical store. `posttraining` uses the **standard planner including
   spillover cluster-packing** (user decision 2026-06-12: less complexity,
   better GPU occupancy — the earlier exact-dims policy rested on a refuted
   spillover-waste measurement). Named costs: a changed input set may re-dim
   unchanged images (regeneration is all-or-nothing), and cross-store
   bit-identity of shared images is not guaranteed (the union reader already
   errors on duplicate media_ids).
4. **Fail loudly on provenance.** `manifest.json` is the commit record (below);
   the consumer refuses mismatched tokenizer sha / vision-tokenizer version /
   unknown dtype, and treats a manifest-less directory as unpublished garbage.

## Format

```
<dataset_name>/
  manifest.json            # COMMIT RECORD, written LAST (tmp + fsync + os.replace):
                           #   schema_version
                           #   tokenizer snapshot path + sha256 (string-compared at
                           #     init; deep re-hash only behind --verify-deep)
                           #   vision_tokenizer {version, resize band}
                           #   expected_min_model_vocab (≥ 266440; engine-side bound)
                           #   token_dtype: "<i4"   (numpy dtype string; consumer
                           #     refuses unknown dtypes)
                           #   media_roots: ["media/"]  (union-indexed; duplicate
                           #     media_id across roots is an error)
                           #   files: {relpath: byte_size}  (existence+size check)
                           #   token_layout: {image_marker, image_marker_id,
                           #     img_start, img_end, img_token_start, eol, eof,
                           #     vision_lo, vision_hi} — derived from the live
                           #     tokenizer at BUILD; consumers/gates read these,
                           #     never hardcode ids (user decision 2026-06-12);
                           #     cross-check convert_tokens_to_ids(marker) at init
  media/                   # log-structured: immutable sealed triples
    media.000000.parquet   #   media_id (sha256 hex), shard, offset_elems,
                           #   length_elems, resize_h, resize_w, kind ("image"), source,
                           #   raw_offset_bytes, raw_length_bytes, raw_ext
    tokens.000000.bin      #   raw <i4, concatenated encapsulated blocks
    raw.000000.bin         #   original image file bytes, as-is (jpeg/png/...),
                           #   concatenated; media_id = sha256 of exactly these
                           #   bytes, so the raw layer is a content-addressed
                           #   blob store with built-in verification
  views/
    train.parquet
    validation.parquet
```

**Units contract** (explicit because the spill precedent mixes units —
`spill.py`'s `token_offset` is in *bytes* while `token_length` is in
*elements*, compensated inside its own readers): the media store uses
**unit-suffixed names** so the schema cannot be confused with the spill
schema, and both are in **token elements**. The consumer slices
`mmap[offset_elems : offset_elems + length_elems]` on a memmap constructed
with the manifest's `token_dtype`.

**Write/append order** (producer): bins → fsync → media parquets → views →
manifest last via the existing `finalize_shard_writer` idiom (tmp + fsync +
`os.replace`). Append (online/agentic future) = seal a new
(`tokens.NNNNNN.bin`, `raw.NNNNNN.bin`, `media.NNNNNN.parquet`) triple, then
atomically replace the manifest. The consumer's index is the union over
`media_roots` of all media parquets.

**Raw payload layer**: the store keeps the original image bytes, not just a
`source` pointer — required to make the store self-contained for (a)
**re-encoding on codebook upgrade** (the store-generation story would
otherwise depend on source-dataset longevity), (b) **pixel-based consumers**:
on-policy-distillation teachers, judge/reward VLMs in online preference loops,
vLLM's image path if a rollout component is not token-native, (c) visual
debugging and Gate-2 spot-checks, (d) takedown compliance (delete one media_id
everywhere). `manifest.json` carries `store_raw: bool` (default true for
preference-scale stores; raw typically dominates store size — original
compressed images vs ~30 KB token blocks — and is still small at preference
scale). The DPO/training consumer never reads `raw.*.bin`.

**`source` column**: first-seen origin (dataset/sample key or path/URL) per
deduped media_id — required by Gate 2, audits, and takedown requests.

View row schema:

| column | type | semantics |
|---|---|---|
| `prompt` | list[{role, content}] | text; one literal `<|image|>` string per image |
| `chosen` / `rejected` | str | response text; markers permitted (image-gen RL later) |
| `prompt_media_refs`, `chosen_media_refs`, `rejected_media_refs` | list[str] | media_ids in marker order, **per field** (flat list was ambiguous: the chosen sequence must not consume rejected's refs) |
| `media_tokens_total`, `text_chars` | int | banding/filtering without payload reads |
| provenance | — | prompt_id, scores, models — passthrough |

Degenerate case: empty `*_media_refs` ⇒ plain text dataset; one schema across eras.

**Views are per-task schemas over one task-agnostic media layer.** The
downloaded corpus splits into preference data (MMPR-v1.2, Vero-600k, mllm-dpo)
and RL prompt sets with verifiable answers (ViRL39K, DeepVision-103K,
Innovator-VL-RL, HLE++ — completions come from rollouts, so there is no
chosen/rejected). The second view type:

| column | type | semantics |
|---|---|---|
| `prompt` | list[{role, content}] | text with `<|image|>` markers |
| `prompt_media_refs` | list[str] | as in preference views |
| `answer` / verifier metadata | task-specific | ground truth for the reward env |
| `task_name` | str | routes to the NeMo-RL environment |

Media stores, producer GPU work, and the consumer splice are identical for
both; DPO splices into chosen/rejected sequences, GRPO splices into rollout
prompts (`prompt_token_ids`). One substrate, two view schemas.

**Marker contract** (per field, both sides):
- Producer: count of literal `<|image|>` in raw text ≡ `len(field_media_refs)`;
  reject or escape source text containing accidental `<|image|>` substrings.
  This is a **new** validator over raw text — the existing marker validation
  lives inside the sft render path this mode drops, and cannot be reused.
- Consumer: count of **id 131079 occurrences in the tokenized field** ≡
  `len(field_media_refs)`, hard error on mismatch. This transitively validates
  that the tokenizer kept the marker atomic. (Note: for preference data the
  stock NeMo-RL pipeline performs **zero** validation between processor output
  and model forward — the double-BOS gate only fires for `message_log` specs —
  so the adapter's checks and Gate 1 are the *only* integrity checks. They are
  not redundant.)

## Component 1 — producer: `preference` mode in `vision_tokenization` (NEW)

Reuse map (corrected post-review):

| stage | status |
|---|---|
| GPU encode + `encapsulate_image` / `encapsulate_batch` (pure functions, verified text-free) | reuse |
| `_SUCCESS` / shard recovery / `finalize_shard_writer` atomic-publish idiom | reuse |
| chat-template render of text (incl. its marker validator) | **dropped** — text stays raw |
| content-hash dedup | **new, at scan/manifest stage** — the plan layer never reads source bytes (header-only scan), so hashing means a full-byte read pass at scan time; cheap fallback: dedup by manifest physical coordinates (same stored row ⇒ same image) with byte-hash as the canonical id |
| unique-media plan | **new shape**: the plan iterates the *deduped media inventory* (one planned unit per media_id; batching via the standard shared planner incl. spillover), not preference documents — downstream rebuild assumes every planned component is encoded, so dedup cannot be an encode-time skip |
| media store + view writer | **new** (the spill writer's schema/keying is hard-coded to `(document_id, component_index)`; what is reused is the idiom, not the class) |
| raw-text marker validator (count vs refs, accidental-marker rejection) | new |

**Sequence-budget defense in depth** (one 1400²-cap image ≈ 7.7k tokens —
alone over a 4096 cap; multi-image docs run ~23k median): (1) the **resize
band is a store-generation parameter** — preference stores may use a smaller
cap than pretrain (1024² → ~4.1k, 768² → ~2.3k tokens/image) when judging
signal does not need full resolution; (2) block lengths are exact at
view-write time, so **length banding is a metadata-only view filter**
(`media_tokens_total` + `text_chars`), routing bands to runs with matching
`max_total_sequence_length` (8192–16384 for multimodal; 4096 is a text-era
setting) and making runtime overlength deaths approximately never instead of
the primary mechanism; (3) the adapter's loss-mask + placeholder-shrink guard
remains as the last-resort net only.

**No system messages in views** (user decision 2026-06-11): the 80/20
system-prompt strip convention is **retired permanently — all future data
prep, SFT included**; existing tokenized SFT data keeps it as a historical
artifact only. Views carry user/assistant turns only; the runtime chat
template supplies the system prompt uniformly — system-prompt changes never
require data regeneration. The producer rejects rows containing system-role
messages (fail loud).

Input: preference rows {prompt messages with `<|image|>` markers, chosen,
rejected, per-field image refs}. Canonical first input:
`vision-datasets/alignment-processed/mllm-dpo.parquet` (5,182 pairs;
`source-id`, `image{bytes,path}`, `prompt` messages with `<image>` marker,
`accepted`/`rejected` message lists) — already this shape modulo marker
spelling and response unwrapping. Its predecessor cut
(`alignment-tokenized/mllm_dpo_naive_layout`: fully-fused accepted/rejected
`.bin/.idx`) is the frozen-text layout this design supersedes. Output: the
layout above. Length bands (e.g. `under_8k`) apply at the **view** level
(metadata-only).

**Scale envelope**: designed for 5k (today's mllm-dpo) → 1M+ pairs. Views stay
text-only (~1–2 GB at 1M pairs). Tokens layer at 1M unique images ≈ 10–30 GB —
**above the RAM-arena threshold, so mmap + striping is the expected steady
state at scale** (the arena is the small-set fast path). Raw layer dominates
(~80–200 GB at 1M; cf. mllm-dpo's 427 MB for 5,182 embedded images) — fine on
capstor, `store_raw` opt-out exists, and the split is what keeps raw bytes out
of the training path entirely (the embedded-bytes parquet approach would not
survive this scale).

## Component 2 — consumer: NeMo-RL adapter (NEW, additive files only)

Seam (verified v0.6.0 + main, empirically probed): `dpo.setup()` accepts
pre-built `AllTaskProcessedDataset`s with an arbitrary processor; downstream
requires only per-message `token_ids` tensors; collation never re-tokenizes.
Zero upstream-file edits.

- `omni_preference_dataset.py`: loads view parquet via the stock loader; builds
  the union `media_id → (root, shard, offset, length)` map from all media
  parquets; **size-gated payload policy**: total media bytes ≤ threshold
  (default 16 GiB) ⇒ read bins fully into RAM at init (one sequential read per
  node, zero random IO after); else `np.memmap` + `MADV_RANDOM` (and bins
  should be written with stripe_count ≥ 4 — Lustre page cache is per-node, and
  block reads are 4–70 KB random); validates manifest at init (string-compare
  shas, dtype check, file existence+size).
- **Training-time IO contract**: all store files are consulted **at init only**
  (views → arrow table; media parquets → in-RAM id map; tokens bins → RAM
  arena or warmed mmap); `raw.*.bin` is never opened by the training consumer.
  Steady-state `__getitem__` = dict lookup + arena slice + splice — zero
  filesystem operations per sample. Implementations MUST NOT do per-sample
  parquet lookups or lazy per-block file opens.
- `omni_preference_preprocessor`: runs stock `get_formatted_message_log`, then
  per field replaces each id-131079 occurrence with the next block from that
  field's refs — slice → **cast int32 → int64 at read** (text token_ids are
  int64; the batch validator rejects mixed dtypes) → owned tensor (`torch.cat`
  copies; never keep a raw memmap-backed tensor). Per-field count validation.
  **Overlength**: set `loss_multiplier = 0` AND shrink to the stock ≤4-tokens-
  per-message placeholder — a dead full-length multimodal sample would
  otherwise dictate the padded width of the whole interleaved batch (OOM risk);
  bisecting ids in the placeholder is irrelevant since the sample is dead.
  Live blocks are never bisected.
- `run_dpo_apertus_omni.py`: thin entrypoint building dataset + processor and
  calling the standard `dpo.setup`/train.
- GRPO later: assembled prompts feed generation as `prompt_token_ids` (verified
  path); the adapter's datum spec must simply **never emit `vllm_content`**
  (its presence flips vLLM to the pixel/multi-modal path). Media ids exceed the
  tokenizer-config vocab of some tools; the engine bound is the *model* vocab —
  hence `expected_min_model_vocab` in the manifest. If an environment produces
  new images mid-rollout, a vision-tokenizer service seals session-scoped
  (`bin`, `parquet`) pairs behind the same union-index interface.

## Validation gates

1. **Splice parity**: adapter-assembled token stream ≡ reference assembly built
   producer-side from the **same media store** + the same template render.
   (Redefined post-review: the old "bit-equality vs sft-mode rendering" is not
   well-defined — sft-mode plans over full doc-grouped, non-deduped sets where
   spillover packing can assign cluster-mean dims, so the same image can encode
   at different dims in the two modes. The oracle shares blocks and tests the
   *splice*, which is what the adapter owns.)
2. **Block integrity**: decoded blocks have intact
   `<|img_start|>H*W…<|img_end_of_frame|><|img_end|>` structure; dims match
   `media.parquet` `resize_h/w`; ids in [131073, 262343]; spot-check vs source
   via the `source` column.
3. **End-to-end fingerprint**: 3-step DPO probe on the first multimodal set;
   step-1 `preference_loss = ln 2` exactly (policy ≡ reference invariant holds
   regardless of modality).

## Phasing

- **P1**: `preference` mode in `vision_tokenization` (+ Gate 2).
- **P2**: NeMo-RL adapter + Gate 1 (parallelizable with P1 — fixtures need
  only a hand-built toy store).
- **P3**: convert `mllm-dpo.parquet` (data exists today) + Gate 3 probe.
- **P4**: GRPO/generation integration (separate design: vLLM scoping, refit
  export parity, no-`vllm_content` datum spec).

Text-only DPO requires none of the above and is already running.

## Appendix A — parked optimizations (priced, deliberately not in scope)

| idea | size | reference |
|---|---|---|
| Prefix-shared DPO forward (one sequence, block-sparse mask via FlexAttention) | 1.1–1.5× DPO throughput; 1.3–1.6× with packing | arXiv:2410.20305, github.com/frankxwang/dpo-prefix-sharing |
| Dynamic tree attention for RL training (prefix tree over rollout groups, DFS fwd/bwd) | up to 8.31× RL training throughput | AReaL-DTA, arXiv:2602.00482 (ICML 2026) |
| Tokenizer artifact rebake (fold 135k visual/audio ids into base vocab; keep ~200 structural specials added) | load ~199 s → seconds, per process | this repo's measurements, 2026-06-11 |
| Content-addressed message-level token cache (unify text chunks with media blocks) | erases the residual 12 min/epoch; memoizes online prompt re-templating | — |

The chosen format already encodes the shared-prefix structure these engines
consume; adopting any of them later requires no data migration.

## Appendix B — review provenance

Adversarial review 2026-06-11 (3 verifying critics over producer code, NeMo-RL
code, and ops surfaces). Empirically verified: marker→131079 atomicity through
the stock chain; double-BOS/dtype validator behaviour on spliced blocks; vLLM
`prompt_token_ids` path. Blockers resolved in v2: media identity vs spillover
cluster dims (→ invariants 2–3, Gate 1 redefinition); units/dtype pinning;
manifest-as-commit-record atomicity. Marker-adjacent whitespace tokenizes
differently than markerless text (added-token boundary) — irrelevant to
correctness since producer and consumer split identically, but token-count
heuristics and cache keys must treat marker-adjacent text as distinct.
