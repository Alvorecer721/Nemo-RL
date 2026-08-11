# Design Doc: Online Omni GRPO+ALP over the `rl_prompt` store (consumer)

Status: draft for review · Date: 2026-06-18 · Branch: fresh off `main`
Producer: **IMPLEMENTED** on `benchmark-image-tokenzier` branch `yxu/rl-prompt` (`rl_prompt` task; `tokenize_rl_prompt_text`, `_parse_rl_prompt_row_with_refs`, `_build_rl_prompt_binidx`, `test_rl_prompt.py`). First dataset = **DeepVision-103K** (`configs/dataset/alignment/deepvision_rl.yaml`), output `tokenized/alignment/deepvision_rl/`.
Producer doc: `benchmark-image-tokenzier/docs/design-docs/rl-prompt-task.md`. Supersedes the consumer half of `docs/design-docs/apertus-mm-onpolicy-dpo-data.md`.

**Real producer contract (verified against code):** per split, `{split}.bin`/`.idx` (prompt-only omni token-id docs, end at id 67, `enable_thinking=True`, no answer tokens) + `index_{split}.parquet` (the sidecar, in `.bin` doc order): `prompt_id, prompt_len, image_offsets, image_lengths, image_tok, answer (str), answer_variants (list[str]), enable_thinking (bool)`. Grading metadata is **`answer` + `answer_variants`**, NOT `answer_type`.

## 1. Summary & how it fits together

Apertus 1.5 8B omni is a token-id-in/out causal LM, so online RL on it is **text GRPO over the omni
vocab** — not the pixel-VLM path. The `rl_prompt` producer emits already-tokenized, generation-ready
prompt docs (Megatron `MMIDIDX`, omni-vocab token-ids with inline image tokens, ending at
`<|assistant_start|>` id 67) + a sidecar `{answer, answer_type}` per doc. A new dataset reads that; a
new processor wraps each doc into a `DatumSpec` whose `message_log[0]["token_ids"]` **is** the
pre-tokenized prompt (no `apply_chat_template`, no re-tokenize) and whose
`extra_env_info = {"ground_truth": answer, "answer_variants": [...]}`. From there the **existing GRPO
loop is untouched**: `batched_message_log_to_flat_message()` concatenates token_ids → `input_ids` →
`VllmGeneration.generate()` via the `prompt_token_ids` path (image tokens ride as opaque ints) →
`rollouts.py` slices `extra_env_info` per sample → a new single-turn verifier env grades by
`answer_type` → ALP shaping (`grpo.py:1629-1641`) subtracts `alp_coef·pass_rate·resp_len/ℓ_max` →
advantage → policy update. **Net new: one dataset reader, one processor, one verifier env, one recipe.**

## 2. Stages (reuse vs build)

| Stage | Mechanism | Status |
|---|---|---|
| Dataset reader | `{split}.bin`/`.idx` + `index_{split}.parquet` (producer's index = the sidecar) → rows `{prompt_token_ids, ground_truth, answer_variants, task_name}` | **NEW** (no MMIDIDX reader exists in `nemo_rl/`) |
| Processor | row → `DatumSpec`: `message_log=[{role:user, content:"", token_ids}]`, `extra_env_info={ground_truth, answer_variants}`, `loss_multiplier` like `math_data_processor:362` | **NEW** + register in `PROCESSOR_REGISTRY` |
| Generation | `format_prompt_for_vllm_generation` → `{prompt_token_ids}` (`vllm/utils.py:46-54`); omni image tokens opaque | **REUSE** |
| Reward/metadata flow | `rl_collate_fn` → `rollouts.py:283-287` → `env.step(messages, extra_env_info)` | **REUSE** |
| Verifier env | `@ray.remote` env+worker, dispatch by `answer_type` over `rewards.py` | **NEW** (~150 LOC; `MathEnvironment` is the template) |
| ALP + stop tokens | `reward_shaping.{enabled,alp_coef,max_response_length}`; `generation.stop_token_ids=[2,68,72]` | **REUSE** (config only) |
| Recipe | inherit `…-gym-reasoning-alp.yaml`; override data/env/policy/generation/ALP | **CONFIG** |

## 3. The verifier environment (folds in the Innovator data reality)

`nemo_rl/environments/single_turn_verifier_environment.py` — `@ray.remote` env + worker, registered as
`single_turn_verifier` in `ENV_REGISTRY` (`utils.py:31`). Structure clones `MathEnvironment.step`
(`math_environment.py:418-473`): extract assistant content, pull `ground_truths` **and** `answer_types`
from `metadata`, `chunk_list_to_workers`, fan out `worker.verify`, return `EnvironmentReturn` with
`terminateds = ones` (single turn). Implement `global_post_process_and_metrics` (accuracy, pass@k,
properly-ended fraction) cloning `math_environment.py:368-406`.

**Grade against the gold set `{ground_truth} ∪ answer_variants`** (the producer carries both; **no
`answer_type` switch**). Per rollout: extract the final answer (boxed / `Answer:` / last line —
DeepVision's system turn enforces `\boxed{}`, so boxed extraction is the primary path), then accept iff
it matches **any** member of the gold set under a layered comparator: symbolic-equal
(`math_expression_reward` / `math_verify`) OR numeric-equal (epsilon) OR normalized exact-string
(lowercase/strip). Return reward in `[0,1]` (1.0 on match else 0.0) so ALP's `pass_rate` stays a true
difficulty signal; empty/garbled extraction → 0.0, never raise. This is `MathEnvironment`'s
boxed-extract + `math_verify` extended to (a) accept variants and (b) fall back to normalized-string for
non-symbolic golds (e.g. `Solid`, `a cat`). The earlier noisy-`answer_type` problem is moot — the
producer pre-curates the acceptable forms into `answer_variants`.

**Answer extraction is required on the output side**: `ground_truth` is the bare answer, but the rollout
is free-form (thinking + answer). The worker must **extract the final answer** (boxed / `Answer:` /
`<answer>…</answer>` / last line) before comparing — reuse `nemo_rl/evals/answer_parsing` + the existing
`rewards.py` extractors. The recipe's prompt/template must instruct a parseable final-answer format.

Reuse verbatim: `math_expression_reward`, `exact_answer_alphanumeric_reward` (`rewards.py:34-85`);
new ~10-LOC numeric and boolean comparators.

## 4. Dataset coverage

DeepVision-103K is uniformly verifiable (every row has `answer` + `answer_variants`), so **no
`answer_type` filter is needed** — the gold-set grader (§3) handles all rows. Log per-row reward + mean
pass-rate to watch coverage. (Future: adding Innovator-VL-RL-172K would need its noisy `answer_type`
handled — drop `svg-code`/`bbox`, and since Innovator lacks curated variants its `answer` is the sole
gold, so the normalized/symbolic comparator carries it.)

## 5. ALP + generation (config, no code)

- ALP is env-agnostic and fully built (`reward_functions.py:55,80-111`; call site `grpo.py:1629-1641`).
  Enable: `grpo.reward_shaping.enabled=true`, `alp_coef=1.0`, `max_response_length ≤ max_new_tokens`.
  **Mutually exclusive** with `stop_properly_penalty_coef`/DAPO overlong (leave null). ALP requires
  exactly one assistant message — the single-turn verifier satisfies this.
- `generation.stop_token_ids=[2,68,72]` (omni EOS set; these are special-token ids, no collision with
  the high image-token range). `max_total_sequence_length=16384`, `max_new_tokens` generous,
  `vllm_cfg.max_model_len=16384`, `enable_thinking=true`.
- EOS-rate metric: start with the `1 - truncated.mean()` proxy (zero new plumbing); only add a real
  `finish_reason` field if ALP tuning needs the precise rate.

## 6. New modules + recipe

1. `nemo_rl/data/datasets/response_datasets/rl_prompt_dataset.py` — `RawDataset` reading `{split}.bin`/`.idx`
   (**use `len(IndexedDataset(prefix))` for the doc count — MMIDIDX header off-by-one**) joined to
   `index_{split}.parquet` by doc order (the producer guarantees alignment); yields
   `{prompt_token_ids, ground_truth, answer_variants, task_name}`.
2. `nemo_rl/data/processors.py` — add `mmididx_grpo_data_processor` + register in `PROCESSOR_REGISTRY`.
3. `nemo_rl/environments/single_turn_verifier_environment.py` — the verifier env (§3).
4. `examples/configs/recipes/llm/probe-grpo-apertus-omni-reasoning-alp.yaml` — inherits the
   reasoning-ALP base; binds the new dataset + `single_turn_verifier` env + omni ckpt/tokenizer +
   `stop_token_ids` + ALP.

## 7. Critical path & first step

Critical path: **reader → processor → verifier env → recipe**. The reader+processor are the only place
a wrong contract *silently* corrupts a run (a mis-shaped `token_ids` tensor or a sidecar mis-join makes
garbage prompts that still "train").

**First step (a unit test, not a 4-node job):** load N rows; assert `message_log[0]["token_ids"]` is a
1-D `long` tensor matching the `MMIDIDX` doc (length + head/tail ints); assert
`len(IndexedDataset(prefix)) == len(sidecar)`; round-trip one prompt through
`batched_message_log_to_flat_message()` and confirm `input_ids` reconstructs the doc with no
re-tokenization. This closes the highest-risk contract before any GRPO launch.

## 8. Risks / open questions

- **Pre-tokenized entry (highest risk)** — processor must NOT call `apply_chat_template`/`tokenizer(...)`.
  Resolved cross-repo: the producer bakes the generation prompt (ends at id 67), so the doc is
  generation-ready; confirm against one decoded doc in the first-step test.
- **No `answer_type`** — grading is gold-set match (`{ground_truth} ∪ answer_variants`); non-symbolic
  golds rely on the normalized-string fallback, so variants must stay curated (DeepVision's are).
- **Answer extraction** — the verifier extracts the final answer from free-form rollouts; DeepVision's
  system turn already enforces `\boxed{}`, so boxed extraction is the primary path.
- **`answer`/`answer_variants` present in the eval split too**, not just train (env reads them both paths).
- **Env reward must be `[0,1]`** for ALP's `pass_rate` to be a true difficulty signal.
- **`max_response_length ≥ max_new_tokens`** or the ALP penalty asymptotes.
- **Scaling** — stay at/under the validated 8-node ceiling (omni sequences raise memory pressure);
  colocated + `sleep_level=2` offloads weights but KV cache stays on GPU at `max_model_len=16384`.

## 9. Test plan

1. **Data contract (first step, §7)** — token_ids shape/identity, sidecar join, no re-tokenize.
2. **Verifier dispatch** — unit-test each `answer_type` branch incl. the robust short-answer grader
   (`"Solid"`, `"15"`, `"4 - 3 = 1"` all grade correctly vs the right gold) and unknown → 0.0; reward in
   `[0,1]`.
3. **Answer extraction** — boxed / `Answer:` / last-line extraction from a free-form thinking rollout.
4. **ALP smoke** — tiny GRPO run on the clean subset; confirm `pass_rate`/length-penalty logged and
   shaped reward feeds advantage; mean gen-len trends down at equal/again-better accuracy.
5. **EOS proxy** — `1 - truncated.mean()` logged as the termination signal.

Verified anchors: `reward_functions.py:58-111`, `grpo.py:1625-1641`, `processors.py:312-379,740-780`,
`math_environment.py:344-473`, `environments/utils.py:31-55,106-136`, `rollouts.py:260-287`,
`data/interfaces.py:35-43`, `vllm/utils.py:46-54`, `vllm_worker.py:613,799`.
