# Apertus 1.5 8B DPO Pilot — Data Pipeline & Artifacts

Handoff doc for downstream agents that want to **analyze the data** (failure modes, content quality, prompt sourcing) or **rebuild the pipeline** under different assumptions.

> **⚠ Canonical Apertus tokenizer — always use this path:**
> ```
> /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed_tools_fixed
> ```
>
> This is the **`tools_fixed`** snapshot (Jun 11 2026): same special-token inventory as `…token_fixed.snapshot-20260611` but with the chat-template tool-call rendering patched (see `chat_template.jinja.bak_20260609` in the dir for the prior version). Every Apertus stage going forward — rollouts, verifier, format conversion, DPO training, future GRPO — should point `policy.tokenizer.name` (and any standalone `AutoTokenizer.from_pretrained` call) at this directory. Both the model checkpoint and the `<|inner_*|>` / `<|tools_*|>` / `<|assistant_end|>` token IDs are identical between the two snapshots, so existing rollouts/JSONL artifacts remain valid; only newly-rendered chat templates differ.
>
> **Files in this repo still pointing at the older `…snapshot-20260611` path** (need a one-line swap when convenient):
> - `examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml`
> - `examples/configs/recipes/llm/probe-grpo-apertus1p5-8b-1n4g-megatron.yaml`
> - `examples/configs/recipes/llm/grpo-format-apertus1p5-8b-1n4g-megatron.yaml`
> - `nemo_rl_apertus/rollouts_blend_offline.py`
> - `nemo_rl_apertus/rollouts_blend_passk_offline.py`
> - `nemo_rl_apertus/rollouts_ifbench_offline.py`
> - `nemo_rl_apertus/rollouts_ifbench_passk_offline.py`
> - `nemo_rl_apertus/vibe_test_offline.py`
>
> The DPO pilot recipe `examples/configs/recipes/llm/dpo-apertus1p5-8b-pilot-megatron.yaml` **inherits** from `dpo-apertus1p5-8b-maxmin-megatron.yaml`, so updating just the latter cascades to the pilot.

## What was built

A 39,815-pair preference dataset for DPO training of Apertus 1.5 8B SFT (`/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200`). Goal: fix two named failure modes seen at the SFT checkpoint — **unclosed thinking** (chat-template violation: model emits `<|inner_prefix|>` but stamps `<|assistant_end|>` from inside the thinking block) and **doom-loops / degenerate generation** (low TTR, pattern lock, runaway to max-tokens).

## Pipeline overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                         Apertus 1.5 8B SFT                            │
│  policy + reference checkpoint for the DPO run that consumes this data│
└───────────────────────────────────────────────────────────────────────┘
                    ▲                              ▲
                    │ chosen (when on-policy)      │ rejected (always on-policy)
                    │                              │
┌─────────┐  pass@8 │   ┌─────────────┐  pass@2    │
│ 10K     │────────►│   │ 3,296       │───────────►│   (39,815 pairs)
│ prompts │ 80K out │   │ all-fail    │  6.5K out  │  /logs/dpo_pairs_pilot_clean.jsonl
│ blend   │         │   │ subset      │            │
└─────────┘         │   └─────────────┘            │
                    │           ▲                  │
                    ▼           │                  │
            verify (rule-based) │                  ▼
            └─► 25.8% verifiable, 10.7% correct   chosen (when teacher fallback)
                                  │                ▲
                                  └────────────────┴── Qwen3.5-9B teacher
```

### Stage 1 — Sample 10K prompts from the blend

**Script:** `nemo_rl_apertus/rollouts_blend_offline.py` (originally; the 10K-prompt set used here is the same one that was checked into `logs/rollouts_blend_apertus_think_8k.jsonl` as the single-shot blend output).

**Source:** [`nvidia/Nemotron-RL-Ultra-Training-Blends`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends), six subsets:

| Subset | Count in 10K sample |
|---|---:|
| mopd | 1667 |
| rlvr1 | 1666 |
| rlvr2 | 1666 |
| ifbench | 1667 |
| rlhf | 1667 |
| reasoning | 1667 |

`swe` was excluded — its `responses_create_params.input` is empty (SWE-bench uses agentic multi-turn, not a single user prompt).

**Sampling**: per-subset stratified, deterministic shuffle (seed 42).

### Stage 2 — Apertus pass@8

**Script:** `nemo_rl_apertus/rollouts_blend_passk_offline.py` (now with `tools=` threaded through `apply_chat_template`)
**Slurm:** `infra/slurm/cscs/rollouts_blend_passk_apertus_10k.slurm`
**Run:** Slurm job 2556084, 4 nodes × TP=4, wall 1:21:43
**Sampling:** `temperature=1.0, top_p=1.0, max_tokens=8192, enable_thinking=True`, vLLM `n=8` per prompt
**Output:** `logs/rollouts_blend_passk_apertus_think_8k_10k.jsonl` (1.84 GB, **80,000 rows**)

Per-row schema:
```jsonc
{
  "prompt_idx": int,        // global, [0, 10000)
  "sample_idx": int,        // [0, 8)
  "config": str,            // mopd / rlvr1 / rlvr2 / ifbench / rlhf / reasoning
  "prompt": str,            // raw user prompt
  "generation_text": str,           // vLLM-decoded (skip_special_tokens=True)
  "generation_text_full": str,      // skip_special_tokens=False (keeps <|inner_*|>, <|tools_*|>, etc.)
  "n_output_tokens": int,
  "finish_reason": "stop" | "length",
  "thinking_emitted": bool,         // <|inner_prefix|> token in IDs
  "thinking_closed": bool,          // <|inner_suffix|> token in IDs
  "ttr": float,                     // word-level type-token ratio
  "top_5gram_count": int,           // top word 5-gram repeat count
  "is_correct": bool                // closed AND stop AND ttr>=0.20 AND top_5gram<10 (FORMAT correctness)
}
```

**Top-line stats (from the morning report):**
- thinking_emitted: 86.2%
- **thinking_closed: 62.1%** (37.9% of rollouts have the unclosed-thinking structural failure)
- `is_correct` (format): 47.6%
- Mean output tokens: 2,798
- `finish_reason=length` (hit 8K cap): 16.9%

### Stage 3 — Hard-verifier sweep

**Script:** `nemo_rl_apertus/verify_passk_offline.py` (Python verifiers ported from NeMo-Gym; deps preinstalled at `/iopsstor/scratch/cscs/nathanrchn/.cache/python_verify_deps`)
**Slurm:** `infra/slurm/cscs/verify_passk_apertus.slurm`
**Run:** Slurm job 2557568, 1 node debug, wall 25:30
**Output:** `logs/rollouts_blend_passk_apertus_think_8k_10k_verified.jsonl` (1.86 GB, same 80,000 rows + new fields)

Added fields:
```jsonc
{
  "agent_ref": str | null,             // e.g. "instruction_following_simple_agent"
  "content_verifiable": bool,          // we have a rule-based verifier for this agent
  "content_correct": bool | null,      // true/false if verifiable, null otherwise
  "verifier_reason": str | null        // diagnostic when we couldn't verify (skip reason)
}
```

**Verifiers ported:** `instruction_following_simple_agent` (IFEval rules via `verifiable_instructions` Google package), `mcqa_simple_agent`, `structured_outputs_simple_agent` / `_v3`, `single_step_tool_use_with_argument_comparison_agent`, `toolcall_schema_*`, `code_gen_simple_agent`.

**Skipped (LLM-judge / agentic):** equivalence_llm_judge, genrm, math_with_judge, swe_pivot, nvarc_*, reasoning_gym, jailbreak_*, abstention, multichallenge, inverse_if, math_formal_lean.

**Top-line stats:**
- 20,624 / 80,000 rows verifiable (25.8%)
- 2,215 / 20,624 content-correct (10.7%)
- Per-agent correctness (top): instruction_following 19.2%, mcqa 14.4%, structured_outputs 11.7% / v3 5.2%, code_gen 3.5%, single_step_tool_use 2.0% (was 0% before the `tools=` fix landed)

### Stage 4 — Extract "all-fail" subset

**Script:** `nemo_rl_apertus/extract_all_fail_prompts.py`
**Mode used:** `content_or_format` — a prompt is all-fail iff none of its 8 samples passes (content_correct when verifiable, else `is_correct` format flag).
**Output:** `logs/all_fail_prompts.jsonl` (one row per all-fail prompt; 3,296 prompts = 33% of the 10K)

Schema: `{prompt_idx, config, prompt, agent_ref}`.

Per-agent breakdown of all-fail prompts (top, see file for tail):
```
single_step_tool_use_with_argument_comparison_agent           634
swe_pivot_single_step_tool_use_with_argument_comparison_agent 491
instruction_following_simple_agent                            476
code_gen_simple_agent                                         413
toolcall_schema_single_step_tool_use_with_argument_comparison 186
structured_outputs_v3_simple_agent                            144
mcqa_simple_agent                                             133
math_with_judge_simple_agent                                  129
nvarc_transductive_simple_agent                               117
nvarc_inductive_simple_agent                                  115
...
```

### Stage 5 — Qwen3.5-9B teacher pass@2

**Script:** `nemo_rl_apertus/rollouts_teacher_qwen_offline.py`
**Slurm:** `infra/slurm/cscs/rollouts_teacher_qwen.slurm`
**Run:** Slurm job 2557647, 4 nodes × TP=4, wall 17:37
**Sampling:** `temperature=1.0, top_p=1.0, max_tokens=8192, enable_thinking=True`, vLLM `n=2` per prompt, `tools=` threaded
**Output:** `logs/rollouts_teacher_qwen_think_8k.jsonl` (210 MB, **6,592 rows = 3,296 prompts × 2 samples**)

Per-row schema:
```jsonc
{
  "prompt_idx": int,        // matches Apertus prompt_idx
  "sample_idx": int,        // [0, 2)
  "config": str,
  "prompt": str,
  "qwen_body": str,         // raw Qwen output (with <think>...</think>, <tool_call>... </tool_call>, <|im_end|>)
  "apertus_body": str,      // qwen_body re-emitted in Apertus format (see Stage 6)
  "n_output_tokens": int,
  "finish_reason": "stop" | "length",
  "model": "Qwen/Qwen3.5-9B"
}
```

**Stats:** mean output tokens 4,622; **45.0% hit 8K cap** (nearly 3× the Apertus rate — Qwen's thinking blocks are much longer).

### Stage 6 — Format conversion (Qwen → Apertus)

**Module:** `nemo_rl_apertus/convert_qwen_to_apertus.py` (smoke-tested, 6/6 cases pass)
**Critical for DPO**: chosen text must use Apertus's exact special tokens, otherwise DPO logprobs are computed over a non-Apertus structure.

Substitutions:
- `<think>X</think>` → `<|inner_prefix|>X<|inner_suffix|>`
- `<tool_call>{"name": N, "arguments": A}</tool_call>` → `<|tools_prefix|>[{"N": A}]<|tools_suffix|>` (Apertus inverted JSON; multiple Hermes calls collapsed into a single Apertus block)
- `<|im_end|>` → (stripped; chat template re-adds `<|assistant_end|>`)

### Stage 7 — Build DPO pair dataset

**Script:** `nemo_rl_apertus/build_dpo_dataset.py`
**Args used:**
```bash
python3 nemo_rl_apertus/build_dpo_dataset.py \
  --apertus-jsonl logs/rollouts_blend_passk_apertus_think_8k_10k_verified.jsonl \
  --qwen-jsonl    logs/rollouts_teacher_qwen_think_8k.jsonl \
  --out           logs/dpo_pairs_pilot.jsonl \
  --chosen-pred   content_or_format \
  --pair-mode     all
```

**Pairing strategy** (per prompt):
1. If Apertus has any sample matching `is_chosen` predicate AND any matching `is_rejected`: emit Apertus-on-policy pairs.
2. Else if Qwen has any sample, pair Qwen-as-chosen (converted to Apertus format) with the worst Apertus as rejected.
3. Else skip.

**Pair quality filters:**
- Chosen prefers longer sample (`n_output_tokens` desc) to avoid teaching the model to be terse.
- Rejected prefers shorter sample (asc).
- **Length-matched**: pair dropped if `(max-min)/max > 0.5` (i.e. neither side may be more than 1.5× the other). This catches the Qwen-much-longer-than-Apertus length bias.
- Max 64 pairs per prompt.

**Output (raw):** `logs/dpo_pairs_pilot.jsonl` (39,815 rows)

**Output (cleaned for DPO):** `logs/dpo_pairs_pilot_clean.jsonl` (same rows, trailing `<|assistant_end|>` stripped from chosen+rejected — the Apertus chat template re-adds it during DPO training; without stripping you'd get a double terminator)

Per-row schema:
```jsonc
{
  "prompt_idx": int,
  "config": str,
  "prompt": str,                 // raw user prompt — DPO renders with the Apertus chat template
  "agent_ref": str | null,
  "chosen": str,                 // Apertus assistant body (no trailing <|assistant_end|>)
  "rejected": str,               // same
  "chosen_source": "apertus" | "qwen",
  "chosen_n_tokens": int,
  "rejected_n_tokens": int
}
```

**Pair counts:**
| Bucket | Count |
|---|---:|
| Apertus on-policy pairs (both from Apertus rollouts) | 17,517 (44%) |
| Teacher-fallback pairs (chosen=Qwen-converted, rejected=Apertus) | 22,298 (56%) |
| **Total** | **39,815** |

**Per-subset pair count:** mopd 8,625 / rlvr2 7,765 / rlvr1 7,649 / ifbench 6,445 / rlhf 5,476 / reasoning 3,855.

**Coverage:** 6,151 / 10,000 prompts (61.5%) produced at least one pair. Why some prompts didn't:
- 292 had no Apertus chosen AND no Qwen sample (edge case — these are prompts that were marked all-fail but somehow didn't make it into the Qwen JSONL; probably index-skew during the all-fail extraction)
- 1,466 had a chosen but no rejected
- 2,091 dropped by the length-match filter

## File index (everything in `/iopsstor/scratch/cscs/nathanrchn/Nemo-RL/`)

### Data — `logs/`

| Path | Size | Rows | What's in it |
|---|---:|---:|---|
| `rollouts_blend_apertus_think_8k.jsonl` | 234 MB | 10,000 | Initial blend, **1 sample per prompt** (precursor to pass@8) |
| `rollouts_blend_passk_apertus_think_8k_10k.jsonl` | 1.84 GB | 80,000 | **Apertus pass@8 raw**, all 80K samples + format metrics |
| `rollouts_blend_passk_apertus_think_8k_10k_verified.jsonl` | 1.86 GB | 80,000 | Same + `agent_ref`, `content_verifiable`, `content_correct`, `verifier_reason` |
| `all_fail_prompts.jsonl` | 1.0 MB | 3,296 | Subset of prompts where Apertus has no acceptable sample in 8 tries |
| `rollouts_teacher_qwen_think_8k.jsonl` | 210 MB | 6,592 | **Qwen3.5-9B teacher rollouts** (pass@2 on all-fail), raw Qwen + Apertus-converted |
| `dpo_pairs_pilot.jsonl` | (large) | 39,815 | **Preference pairs**, raw (trailing `<|assistant_end|>` included) |
| `dpo_pairs_pilot_clean.jsonl` | (large) | 39,815 | **DPO training input** (terminator stripped) |

### Old artifacts (pre-fix, for comparison)

| Path | Notes |
|---|---|
| `rollouts_blend_passk_apertus_think_8k.jsonl` | Earlier 1,280-prompt pass@8 sample |
| `rollouts_blend_passk_apertus_think_8k_verified.jsonl` | Verified version of the 1,280-prompt sample |
| `rollouts_toolfix_apertus_think_8k.jsonl` | 103 tool-use prompts re-rolled with `tools=` threaded (showed emission rate 0.1% → 28.8%) |

### Scripts — `nemo_rl_apertus/`

| Script | Purpose |
|---|---|
| `rollouts_blend_offline.py` | Single-shot blend rollouts (10K × 1) |
| `rollouts_blend_passk_offline.py` | Pass@k with `tools=` threading (used for the 10K pass@8) |
| `rollouts_teacher_qwen_offline.py` | Qwen3.5-9B teacher rollouts + Apertus-format conversion inline |
| `convert_qwen_to_apertus.py` | Stand-alone Qwen→Apertus body converter (importable + CLI smoke test) |
| `verify_passk_offline.py` | Hard-verifier sweep (IFEval, mcqa, structured_outputs, tool_use, code_gen) |
| `extract_all_fail_prompts.py` | Extract subset of all-fail prompts for teacher pass |
| `build_dpo_dataset.py` | Pair builder with length-matching + Apertus-on-policy fallback to Qwen-teacher |

### Slurm wrappers — `infra/slurm/cscs/`

| Wrapper | Used for |
|---|---|
| `rollouts_blend_passk_apertus_10k.slurm` | 4-node Apertus pass@8 at 10K |
| `rollouts_teacher_qwen.slurm` | 4-node Qwen teacher on all-fail subset |
| `verify_passk_apertus.slurm` | 1-node verifier sweep |
| `probe_dpo_apertus_pilot.slurm` | 1-node DPO pilot training |

### Recipes — `examples/configs/recipes/llm/`

| Recipe | Purpose |
|---|---|
| `dpo-apertus1p5-8b-pilot-megatron.yaml` | DPO pilot recipe (inherits from `dpo-apertus1p5-8b-maxmin-megatron.yaml`, overrides data paths + hyperparams) |

## Known failure modes & open questions (for the next agent)

1. **`tools=` was missing in the original pass@8 (1,280-prompt sample).** That older file shows 0% tool-call emission. The 10K pass@8 has the fix and shows 28.8% tool-call emission. **Use the 10K file** when looking at tool-use behaviour; the 1,280 sample only reflects "tools disabled" rendering.
2. **Multichallenge / inverse_if are technically rule-graded but their rubrics are natural-language ("Does the response start with..." etc.) — they require an LLM judge.** The verifier sweep skips them. ~100 prompts in the 10K pass@8 sit in this bucket and got format-only signal.
3. **Code_gen verifier runs untrusted code in a subprocess with a 10-s timeout.** ~50K subprocess invocations during the 80K-row verify run. No sandboxing beyond timeout; safe-ish for stdin/stdout tasks but don't repurpose for adversarial code.
4. **Length-mismatch filter dropped 2,091 pairs.** Most are Qwen-much-longer-than-Apertus. Re-run with `--length-ratio 0.7` if you want them back at the cost of some length-bias risk.
5. **Tool-use content-correctness is still low even after the `tools=` fix** (single_step 2.0%, toolcall_schema 4.2%) — Apertus genuinely struggles with picking the right tool + args, not just format. Teacher (Qwen) fallback covers the gap for 634 + 186 = 820 tool-use prompts in the all-fail set.
6. **Qwen3.5-9B is multimodal (image-text-to-text).** Loaded text-only via vLLM (no images). Worked fine but if you swap to another teacher, double-check the chat-template/tool-call convention; my converter only handles Hermes-style `<tool_call>` (Qwen3) and `<think>`. Llama-style or GPT-OSS would need new branches.
7. **DPO training itself**: see slurm job 2557935 (latest) for the actual training run. The recipe at `examples/configs/recipes/llm/dpo-apertus1p5-8b-pilot-megatron.yaml` uses TP=4, seq=8192, activation_checkpointing, β=0.1, 311 steps over the cleaned pair file. Earlier attempts OOMed at seq=8192 with TP=2 — the vocab-parallel logits exp() needed 8.7 GB per GPU.

## How to reproduce / extend

**To rebuild the full pipeline from scratch:**
```bash
# 1. Apertus pass@8 (3h on 4 nodes)
sbatch infra/slurm/cscs/rollouts_blend_passk_apertus_10k.slurm

# 2. Verify (25 min on 1 debug node)
sbatch infra/slurm/cscs/verify_passk_apertus.slurm

# 3. Extract all-fail (instant)
python3 nemo_rl_apertus/extract_all_fail_prompts.py \
  --passk-jsonl logs/rollouts_blend_passk_apertus_think_8k_10k_verified.jsonl \
  --out logs/all_fail_prompts.jsonl --mode content_or_format

# 4. Qwen teacher (20 min on 4 nodes)
PROMPTS_JSONL=logs/all_fail_prompts.jsonl sbatch infra/slurm/cscs/rollouts_teacher_qwen.slurm

# 5. Build pairs (instant)
python3 nemo_rl_apertus/build_dpo_dataset.py \
  --apertus-jsonl logs/rollouts_blend_passk_apertus_think_8k_10k_verified.jsonl \
  --qwen-jsonl logs/rollouts_teacher_qwen_think_8k.jsonl \
  --out logs/dpo_pairs_pilot.jsonl --chosen-pred content_or_format --pair-mode all

# 6. Clean (strip terminator) — the snippet lives in the conversation transcript
#    but you can also just regenerate via:
python3 -c "
import json
with open('logs/dpo_pairs_pilot.jsonl') as fi, open('logs/dpo_pairs_pilot_clean.jsonl','w') as fo:
    for line in fi:
        r = json.loads(line)
        for k in ('chosen','rejected'):
            if r[k].endswith('<|assistant_end|>'): r[k] = r[k][:-len('<|assistant_end|>')].rstrip()
        fo.write(json.dumps(r, ensure_ascii=False)+'\n')
"

# 7. DPO training (~2h)
sbatch infra/slurm/cscs/probe_dpo_apertus_pilot.slurm
```

**Submission incantation when submitting from a compute node** (the dev session does this):
```
env -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts \
    LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:$LD_LIBRARY_PATH \
    sbatch <slurm-script>
```

**For different sample sizes / chosen predicates:**
- `extract_all_fail_prompts.py --mode format` only fails format (closes the loop on doom-loops, ignores content) — gives bigger all-fail set, more teacher pairs.
- `build_dpo_dataset.py --chosen-pred format` builds pairs from format signal alone — gives ~9K Apertus pairs without Qwen fallback (much smaller dataset but pure on-policy).
- `build_dpo_dataset.py --pair-mode one` picks 1 pair per prompt — ~6K pairs, lighter weight DPO.

## Pointers for an analysis agent

If you want to study **failure modes**, work from `rollouts_blend_passk_apertus_think_8k_10k_verified.jsonl`. Each row has both raw + with-specials text plus the structural flags and the rule-based verdict. Heavy hitters worth slicing on:

- `thinking_emitted AND NOT thinking_closed AND finish_reason=="stop"` → the **chat-template violation** case (the "natural-stop unclosed thinking" — model treats the thinking block as the final answer). Roughly 10% of all rollouts.
- `thinking_emitted AND NOT thinking_closed AND finish_reason=="length"` → the **runaway-thinking** case (true doom-loop in the inner block until 8K cap). Roughly 17%.
- `top_5gram_count >= 10` → **pattern-lock** (model gets stuck on a sentence template like "The universe is not just expanding; it's also...").
- `ttr < 0.20 AND n_output_tokens > 50` → **degenerate vocabulary** (severe doom-loop).
- `content_verifiable AND NOT content_correct` → **wrong answer** when verifier exists (decompose by `agent_ref` for failure-type analysis).
- `agent_ref starting with single_step_tool_use` AND `<|tools_prefix|>` not in `generation_text_full` → **tool-use refusal** (model declines to attempt a tool call even when tools=N>0).

For **pair-quality analysis**, work from `dpo_pairs_pilot_clean.jsonl`. `chosen_source` partitions on-policy vs teacher-fallback; `chosen_n_tokens` vs `rejected_n_tokens` shows length skew per-pair.

For **prompt-source provenance**, join back to the source rows via `(config, prompt)` against the cached HF files at:
```
/iopsstor/scratch/cscs/nathanrchn/.cache/huggingface/hub/datasets--nvidia--Nemotron-RL-Ultra-Training-Blends/snapshots/<sha>/<config>.jsonl
```
The verifier code already does this join; reuse `nemo_rl_apertus/verify_passk_offline.py::build_prompt_index`.

## Open questions worth answering

- Does the Apertus-on-policy half of the pairs perform differently from the Qwen-fallback half during DPO? (Likely yes — teacher pairs are stylistically further from base, may drive more aggressive updates.) Suggestion: split the dataset and ablate.
- Length-collapse risk: even with `length_ratio=0.5`, mean chosen (4,482) is close to mean rejected (4,580) at the macro level, but per-pair distribution may have a long tail. Plot the per-pair length-ratio distribution.
- mopd contributes 22% of pairs — that's the hardest subset for Apertus and also the most likely to teach the model unhelpful styles via teacher pairs. Could be worth a per-subset DPO ablation.
