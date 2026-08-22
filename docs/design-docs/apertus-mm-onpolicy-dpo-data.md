# On-Policy Multimodal DPO Data Generation for Apertus 1.5 8B (Omni)

Status: draft for review (v4, prompt source = Innovator-VL-RL-172K) · Date: 2026-06-18 · Target branch: fresh branch off `main`

## 1. Goal

Sample on-policy rollouts from Apertus 1.5 8B (omni) on **verifiable multimodal reasoning prompts**
(Innovator-VL-RL-172K, expanding to ViRL39K / DeepVision / Vero-STEM), score each rollout for
**correctness + termination**,
and emit **correctness-matched** preference pairs that the DPO trainer consumes — so multimodal DPO
also reduces "non-stop thinking" (runaway, never-emits-EOS generation). Deliverable is the
**data-generation half**; training reuses stock DPO.

## 2. Non-goals

- Not `swiss-ai/model-launch` (a serving-job launcher; NeMo-RL samples natively).
- Not a pixel-in VLM path. Apertus 1.5 is an **Emu3-style omni causal LM**: image/audio are discrete
  vocab tokens. After the producer pass they are **inline** `<|visual token N|>` blocks — so there is
  **no MediaStore, no `<|image|>` marker, no splice** in the generation/training path; the pipeline is
  **token-id-in / token-id-out + stock DPO over the omni vocab**.
- Not re-tokenizing prompts from text downstream (inline image tokens cannot survive a text round-trip).
- Not online DPO; not a new preference loss (the data design carries the length signal).

## 3. Context (verified)

- Checkpoint `ap1p5-8b-sft-256k-…` = `ApertusForCausalLM`. Tokenizer
  `apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611`: text `vocab_size=131072`,
  full `len=266440`. Model embedding covers the full range.
- **Specials:** `<|user_end|>`=66, `<|assistant_start|>`=67, `<|assistant_end|>`=68, `</s>`=2;
  image block `<|img_start|>`=131073 … `<|img_end|>`=131074; `<|image|>` marker=131079.
  EOS stop set = `[2, 68, 72]`.
- **Producer output format (same as the existing SFT corpus):** Megatron `MMIDIDX` IndexedDataset,
  int32, one doc/seq, **image tokens inline**; per-seq
  `[<|img_start|> …visual… <|img_end|>] question <|user_end|>(66) <|assistant_start|>(67) …`.
  Sequence cap 16,384 (`under_16k`); prompts are short→moderate, leaving ample room for a thinking trace.
- **`enable_thinking` = a developer-preamble flag** (token 18730 ` enabled` / 21332 ` disabled`); the
  generation prompt ends at `<|assistant_start|>`(67). Tokenizing the RL prompts **fresh** lets us set
  `Deliberation: enabled` at producer time (no post-hoc token flip needed).
- **Prompt source — Innovator-VL-RL-172K** (`…/raw/rl/hf___InnovatorLab___Innovator-VL-RL-172K`):
  ~172K instances (`id, images, problem, answer, problem_type, answer_type, source, prompt_type`). GT is
  the `answer` field (e.g. `['D']`); `answer_type` selects the verifier — **multiple-choice ~40%,
  math-expressions ~42%, boolean/number ~3%** (cleanly verifiable), plus svg-code/bbox/any/ocrtext
  (excluded, or GIoU for bbox). `problem_type`: stem ~57%, spatial, ocr, coding, grounding. **The whole
  dataset is pre-curated to MEDIUM difficulty** (Pass@N-vs-Pass@1 discrepancy + reward filtering,
  arXiv:2601.19325) — solvable-but-not-reliably, exactly the regime that yields multiple correct
  rollouts of differing length for pairing, so the sparse-pair risk is mitigated at source. Images are
  bytes+path. Companions for later: ViRL39K (boxed + per-item pass-rate), DeepVision-103K
  (math+visual-logic, `equivalent_answers`), Vero-600k STEM (`ground_truth` + numeric `tolerance`).

## 4. Assumptions

1. The **producer pass** (`vision_tokenization`) can tokenize Innovator `(image, problem)` into the
   omni `MMIDIDX` format with **`Deliberation: enabled`** and **carry `answer` (GT) + `answer_type`
   (verifier selector)** in a sidecar keyed by `id`. This is a defined prerequisite stage, not an open
   question.
2. The served checkpoint's embedding covers `len(tokenizer)=266440` (it is the SFT ckpt).
3. Boxed answers give a clean correctness oracle (math-verify for numeric, exact-match for the letter).
4. No backward-compat constraints: net-new code in a new module dir + one additive `finish_reason`
   field; nothing existing changes behavior.

## 5. Architecture & contracts

- **Source of truth:** producer-tokenized ViRL39K prompts (read-only) + a GT/verifier/difficulty
  sidecar keyed by sample id.
- **Harvest contract:** we use only `(prompt token-ids, ground-truth, verifier_type)`. The
  ground-truth `answer`/`answer_type` lives in the producer **sidecar / view payload, NEVER in the
  prompt token stream** — otherwise the model reads the answer at rollout time and every rollout is
  trivially correct. The RL dataset is a bank of verifiable reasoning prompts; we generate the
  responses ourselves.
- **Prompt/answer split contract:** producer prompts end at `<|assistant_start|>`(67); if a trailing
  answer is present, split at the last 67/68 span and keep the prefix. Count 67s to skip multi-turn.
- **Token-id passthrough contract:** generation prompts and preference message-logs carry **token-ids
  directly** (inline image tokens included). `format_prompt_for_vllm_generation` already emits
  `{"prompt_token_ids": …}`; `preference_collate_fn` consumes per-message `token_ids` and never
  re-tokenizes. The whole pipeline is token-id-in / token-id-out — standard DPO over the omni vocab.
- **`enable_thinking` contract:** thinking is required (non-stop thinking only exists in thinking mode).
  Set `Deliberation: enabled` **at producer time** for the RL path (preferred); the in-place preamble
  token flip is only a fallback for reusing already-tokenized (deliberation-off) data.
- **Isolation contract:** new logic in a dedicated module dir; the single shared change is the additive
  `finish_reason` field (§9.4).

## 6. Prompt source: Innovator-VL-RL-172K (then expand)

1. **Producer pass (prerequisite) — the `rl_prompt` task in `vision_tokenization`** (branch
   `yxu/multirank-alignment`). The alignment mode is already `task`-parametrized and reserves
   `rl_prompt` (`_task/alignment.yaml`), so this is a planned ~85%-reuse extension of the DPO path, not
   a new mode. It emits: omni `MMIDIDX` **prompt-only** docs (image inline + problem, ending at
   `<|assistant_start|>(67)`, **no answer in tokens**) + a **sidecar keyed by `prompt_id`** carrying
   `answer` (raw `list[str]`) + `answer_type` (raw string passthrough). New code is small: a
   `task: rl_prompt` branch in `indexing/alignment/ingest.py` + `answer`/`answer_type` on
   `payload.py:VIEW_SCHEMA`, a prompt-only binidx (drop the chosen/rejected concat at
   `dpo_pairs.py:140`), the sidecar emitter, and two config files. Two producer-side **musts**:
   (a) **force `enable_thinking=True`** for `rl_prompt` — deliberation is auto-derived from assistant
   CoT markers (`_has_thinking_content`), which a prompt-only doc lacks, so it would otherwise emit
   `Deliberation: disabled`; (b) producer container **pyarrow ≥14** (the `Repetition level histogram`
   error is an old-pyarrow bug; the scanner reads bytes-image parquet natively at ≥14). Emit the
   sidecar in the same row loop as the `.bin` docs so order ≡ `.idx` doc order; hard-fail on unmatched
   `id`; use `len(IndexedDataset)` for the doc count (MMIDIDX header off-by-one).
2. **Answer-type filter:** keep verifiable types — `multiple-choice` (exact-match), `math-expressions`
   (math-verify), `boolean` (yes/no), `number` (numeric). Exclude `svg-code`/`any`/`ocrtext`; defer
   `bbox` (needs GIoU). **No per-item difficulty filter needed** — Innovator is pre-curated to medium
   difficulty, which mitigates the sparse-pair risk at source.
3. **Verifier dispatch by `answer_type`:** exact-match / math-verify / yes-no / numeric
   (`exact_answer_alphanumeric_reward`, `dapo_math_verifier`, `answer_parsing`).
4. **Expansion (later phases):** ViRL39K (boxed + per-item difficulty), DeepVision-103K (math +
   visual-logic), Vero-600k STEM, the remaining Innovator answer-types — same producer pass + verifier
   dispatch.

## 7. Pipeline (reuse vs build)

| Stage | Mechanism | Status |
|---|---|---|
| 0. Producer pass | `vision_tokenization` → omni `MMIDIDX` prompts (deliberation on) + GT sidecar | EXTERNAL (your tool) |
| 1. Read + answer-type filter | Megatron `IndexedDataset`; load `answer`/`answer_type` sidecar; keep verifiable types | **BUILD** (reader reuse + thin filter) |
| 2. K-sample rollouts | `format_prompt_for_vllm_generation` (`prompt_token_ids`) → `generate()`; `repeat_interleave(K)`; `stop_token_ids=[2,68,72]` | REUSE + 1 config line + `finish_reason` fix |
| 3. Score | boxed math-verify / exact-match; termination from `finish_reason` | REUSE + dispatcher |
| 4. Pairs | build `PreferenceDatumSpec` directly: prompt token-ids + `tokenize(response)`; no MediaStore | **BUILD** ~60 lines |
| 5. Train | **stock** `run_dpo.py` + a thin preference dataset yielding the specs | REUSE + thin dataset |
| 6. Eval | Stages 1–3 on held-out → gen-len, EOS-rate, pass-rate | REUSE + thin call-site |

## 8. Labeling contract

Per prompt with K rollouts (thinking on):
1. Score each: `is_correct` (boxed verify) and `terminated` (clean EOS via `finish_reason`, not
   length-cap).
2. **Pair only among correct rollouts.** `chosen` = shortest cleanly-terminated; `rejected` = longer /
   runaway / truncated. Drop prompts with <2 correct or no length spread.
3. **Never label by length alone** — correctness gates, length/termination is the tiebreak. The chosen
   (shorter) is the higher-likelihood sequence, so DPO's gradient pushes down hardest on the long
   rejected one.

## 9. New modules (smallest-first)

1. `…/rl_prompt_reader.py` — wrap Megatron `IndexedDataset`; join the `answer`/`answer_type` sidecar;
   keep verifiable types; yield `(prompt_token_ids, ground_truth, verifier_type)`. (~70 lines)
2. `…/omni_scorer.py` — `score_rollout(text, gt, verifier_type) → (is_correct, terminated)` over
   existing reward funcs + `finish_reason`. (~40 lines)
3. `…/rollout_preference.py` — pick chosen/rejected per §8; assemble `PreferenceDatumSpec` (prompt
   token-ids + `tokenize(chosen/rejected)` + `<|assistant_end|>`/eos), loss-mask response only; write
   records + a thin loader. (~60 lines)
4. **Additive interface change** — `finish_reason: NotRequired[list[str]]` on `GenerationOutputSpec`
   (`nemo_rl/models/generation/interfaces.py`), populated in `vllm_worker.py:~799` /
   `vllm_worker_async.py:~920` (currently collapsed to `is_truncated`). **Required** for termination
   labeling. (~6 lines / 3 files)
5. Rollout driver call-site: `reader → repeat_interleave(K) → generate() → score → build pairs`.
   (`online_dpo.py` on `yxu/v0.6.0-online-dpo` is the structural precedent.)
6. Config: Apertus recipe sets `generation.vllm_cfg.stop_token_ids: [2, 68, 72]`;
   `max_total_sequence_length = 16384`; generous `max_new_tokens`.

## 10. Phased plan (dependency-ordered)

- **Phase 0a — producer pass** on a small Innovator slice (verifiable `answer_type`s) → omni `MMIDIDX`
  prompts + `answer`/`answer_type` sidecar.
- **Phase 0b — gating smoke test (half-day).** Build §9.4 + §9.6; on a few producer-tokenized prompts,
  confirm (a) thinking trace + boxed answer is produced, (b) **clean EOS** (`finish_reason != "length"`)
  on at least some rollouts, (c) inline visual-token ids load into vLLM without OOB. **If (b) fails,
  stop.**
- **Phase 1 — read/filter + gen.** §9.1 reader + §9.5 driver over the Innovator verifiable subset.
  **Measure Apertus-8B pass-rate and pair yield.**
- **Phase 2 — score + pairs.** §9.2 + §9.3; round-trip test (records load + collate into a valid DPO batch).
- **Phase 3 — train + eval.** Stock DPO; eval gen-len / EOS-rate / pass-rate vs the pre-DPO checkpoint.
- **Phase 4 — expand** to ViRL39K / DeepVision / Vero-STEM once the mechanism is proven.

## 11. Branch & integration

Fresh branch off `main`. **No `omni_preference`/`media_store` port** (this data is inline, markerless).
Dependencies: Megatron `IndexedDataset` reader (submodule at `3rdparty/Megatron-LM`), stock
`run_dpo.py` + `dpo.py`, the existing vLLM generation path, the new modules, and the external producer
pass. Self-contained, mergeable PR.

## 12. Risks & open questions

**Risks**
1. **(Core premise) Clean EOS under forced thinking** — never exercised; retired by Phase 0b. If the
   model rarely stops, there is no chosen/rejected length signal.
2. **(Producer dependency)** the RL→omni tokenization (deliberation on + `answer`/`answer_type`
   carriage) is the external `vision_tokenization` tool; confirm it can carry these fields keyed by `id`.
3. **(Difficulty for Apertus-8B)** Innovator's medium-difficulty curation targets other models;
   re-measure Apertus-8B pass-rate in Phase 1 — if too hard (few correct rollouts), raise K or add
   easier sources.
4. **(Pair yield)** still needs ≥2 correct, different-length rollouts/prompt; the medium-difficulty
   curation + K are the levers; measure early.
5. **(Verifier strictness)** boxed-format / answer-tag mismatch silently scores 0 → false "all wrong";
   enforce the boxed convention in the generation prompt.
6. **(Toolchain)** bytes-image RL parquets (Innovator/DeepVision/Vero) need `duckdb` / HF `datasets`
   (this env's pyarrow errors); confirmed `duckdb` reads Innovator.

**Open questions**
1. Producer config for Innovator: confirm it emits omni `MMIDIDX` prompts (deliberation on) + an
   `answer`/`answer_type` sidecar joined by sample `id`.
2. `K` and `max_new_tokens` — set after the Phase 1 yield/pass-rate measurement.

## 13. Success criteria

- **Phase 0b gate:** a non-trivial fraction of rollouts stop on a clean EOS with thinking on.
- **Phase 1 gate:** pair yield high enough for a usable dataset at the chosen K and `pr` band.
- **Final:** on a held-out Innovator split, the DPO'd checkpoint shows **lower mean generation length and
  higher clean-EOS rate at equal-or-better pass-rate** than the pre-DPO checkpoint.
