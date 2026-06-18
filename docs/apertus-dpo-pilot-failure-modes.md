# Apertus 1.5 8B SFT — Failure Modes Report

Distilled from `docs/apertus-dpo-pilot-data.md`. All percentages are over the
80,000-row pass@8 sweep at `logs/rollouts_blend_passk_apertus_think_8k_10k_verified.jsonl`
(10K prompts × 8 samples), unless noted. The two named failure modes the DPO
pilot is designed to fix are **unclosed thinking** and **doom-loops**; this
report decomposes those plus the content-correctness failures the verifier
sweep exposed.

## Headline numbers

| Signal | Rate | Interpretation |
|---|---:|---|
| `thinking_emitted` | 86.2% | Model enters a think block on most prompts. |
| `thinking_closed` | 62.1% | **37.9% emit `<\|inner_prefix\|>` but never close it** — chat-template violation. |
| `is_correct` (format gate) | 47.6% | Closed AND `stop` AND TTR≥0.20 AND top-5gram<10. ~52% of rollouts fail at least one structural check. |
| `finish_reason == "length"` | 16.9% | Hit the 8K-token cap. |
| Mean output tokens | 2,798 | — |
| `content_verifiable` | 25.8% | 20,624 / 80,000 rows had a rule-based grader. |
| `content_correct` (of verifiable) | **10.7%** | Pass-rate is brutal on the slice we can grade. |
| All-fail prompts (none of 8 acceptable) | **33.0%** | 3,296 / 10,000 prompts. |

## 1. Structural / chat-template failures

The Apertus chat template surrounds the reasoning block with
`<|inner_prefix|>…<|inner_suffix|>` and ends the assistant turn with
`<|assistant_end|>`. The SFT checkpoint regularly stamps `<|assistant_end|>`
from *inside* the inner block, so the consumer sees an assistant turn whose
body is just the thinking text.

### 1a. Natural-stop unclosed thinking (~10% of all rollouts)

- Slice: `thinking_emitted AND NOT thinking_closed AND finish_reason == "stop"`.
- Symptom: model emits `<|inner_prefix|>`, reasons, decides it's done, and
  emits `<|assistant_end|>` without ever emitting `<|inner_suffix|>`.
- Why it matters: downstream chat-template parsers either show the raw think
  trace to the user or drop the whole turn. The "answer" never gets emitted.

### 1b. Runaway-thinking to the cap (~17% of all rollouts)

- Slice: `thinking_emitted AND NOT thinking_closed AND finish_reason == "length"`.
- Symptom: model enters the inner block and keeps going until the 8K-token
  cap (or whatever max was set). Token budget is consumed entirely by
  unbounded reasoning.
- Overlaps heavily with the degeneration modes below (pattern lock /
  low-TTR), which is what keeps the inner block alive.

Together 1a + 1b account for the full 37.9% unclosed-thinking rate. The
DPO pair set targets both: rejected samples are drawn from these slices,
chosen are length-matched samples that close cleanly.

## 2. Degeneration / doom-loops

These are *content-level* pathologies, mostly inside the never-closed think
block but also visible in answers.

### 2a. Pattern lock

- Slice: `top_5gram_count >= 10`.
- Symptom: a 5-word window repeats ≥10× in a single rollout — the model
  locks onto a sentence template (e.g. "The universe is not just expanding;
  it's also …") and emits it on a loop with minor edits.

### 2b. Degenerate vocabulary (severe doom-loop)

- Slice: `ttr < 0.20 AND n_output_tokens > 50`.
- Symptom: word-level type/token ratio collapses — large output, tiny
  vocabulary. Usually accompanied by 1b (runs to the length cap).

### 2c. Length-cap hit without one of the above

- Slice: `finish_reason == "length"` without 2a/2b.
- Less pathological than 2a/2b but still wastes the budget; often the model
  is "thinking productively" but never converges to an answer.

The `is_correct` format gate combines all of these: a rollout has to close
its think block, stop on EOS, keep TTR ≥ 0.20, and keep its top 5-gram count
< 10 to pass. The 52.4% fail rate is dominated by 1a/1b (structural) with a
long tail of 2a/2b.

## 3. Content-correctness failures (where a verifier exists)

Of the 25.8% of rollouts a rule-based verifier could grade, **89.3%** were
graded wrong. Per-agent breakdown of *content-correct* rate among graded
rollouts (higher = better; lower numbers = harder for the SFT model):

| Agent (verifier family) | Content-correct rate |
|---|---:|
| `instruction_following_simple_agent` | 19.2% |
| `mcqa_simple_agent` | 14.4% |
| `structured_outputs_simple_agent` | 11.7% |
| `structured_outputs_v3_simple_agent` | 5.2% |
| `code_gen_simple_agent` | 3.5% |
| `single_step_tool_use_with_argument_comparison_agent` | 2.0% |
| `toolcall_schema_single_step_tool_use_*` | 4.2% |

Caveats:
- LLM-judge / agentic verifiers (equivalence, math_with_judge, swe_pivot,
  reasoning_gym, jailbreak_*, abstention, multichallenge, inverse_if,
  math_formal_lean, nvarc_*) are skipped — those rollouts only have the
  *format* signal, so their content failure rate is unknown.
- `multichallenge` / `inverse_if` are rule-graded *in name* but the rules are
  natural-language ("Does the response start with…") and require an LLM
  judge. ~100 prompts in the 10K sit in this gap and got format-only signal.

## 4. Tool-use failures

Tool use is the worst-graded skill on the verifiable slice. Three distinct
sub-modes:

### 4a. Tool-call refusal

- Slice: `agent_ref` starts with `single_step_tool_use` AND
  `<|tools_prefix|>` is absent from `generation_text_full`.
- Symptom: even when the prompt offers tools (`tools=` is non-empty), the
  model emits a plain-text answer and never opens a tool-call block.
- The original 1,280-prompt pass@8 had **0%** tool-call emission because
  `tools=` wasn't threaded through `apply_chat_template`. The 10K pass@8 has
  the fix and emits a tool call **28.8%** of the time — but that still
  leaves ~71% of single-step tool prompts where the model just refuses to
  call a tool.

### 4b. Wrong tool / wrong arguments

- Slice: `single_step_tool_use_*` with `content_correct == False`.
- Even when the model emits `<|tools_prefix|>`, it picks the wrong function
  or the wrong arguments. This is the residual 98% of single_step failures
  after format is in place.

### 4c. Schema violations on the JSON envelope

- Slice: `toolcall_schema_*` with `content_correct == False`.
- Apertus's tool-call format is *inverted* JSON
  (`[{"<tool_name>": <args>}]`) rather than Hermes-style
  `{"name": ..., "arguments": ...}`. Schema-strict prompts catch the model
  emitting the wrong shape even when the underlying intent is right.

Tool-use prompts dominate the all-fail set: 634 (single_step) + 491
(swe_pivot single_step) + 186 (toolcall_schema) = **1,311 / 3,296 ≈ 40%** of
all-fail prompts are tool-use.

## 5. Domain / prompt-source breakdown

Of the 3,296 all-fail prompts (no acceptable sample in 8 tries), the top
contributors are:

| Agent ref | All-fail prompts |
|---|---:|
| `single_step_tool_use_with_argument_comparison_agent` | 634 |
| `swe_pivot_single_step_tool_use_with_argument_comparison_agent` | 491 |
| `instruction_following_simple_agent` | 476 |
| `code_gen_simple_agent` | 413 |
| `toolcall_schema_single_step_tool_use_with_argument_comparison_agent` | 186 |
| `structured_outputs_v3_simple_agent` | 144 |
| `mcqa_simple_agent` | 133 |
| `math_with_judge_simple_agent` | 129 |
| `nvarc_transductive_simple_agent` | 117 |
| `nvarc_inductive_simple_agent` | 115 |

The all-fail set was extracted under `content_or_format` — a prompt is
all-fail iff none of its 8 samples passes the content verifier (when
present) or the format gate (otherwise). So `math_with_judge_simple_agent`
and `nvarc_*` are present here only on the format signal (no rule grader);
their true content-correctness on the SFT model is unknown.

## 6. Coverage and known biases

- **Subset weighting in the DPO pair set is skewed by failure rate.** mopd
  contributes 8,625 / 39,815 pairs (22%) because it's the subset where
  Apertus struggles most; reasoning contributes only 3,855 (10%) because
  Apertus already has acceptable samples on most reasoning prompts.
- **Length-mismatch filter** dropped 2,091 / 6,151 candidate-pair-producing
  prompts. Vast majority were Qwen-much-longer-than-Apertus — i.e. the
  teacher solves the prompt by thinking 3× as long. That's a real
  capability gap, not just style.
- **Qwen teacher hit the 8K cap 45.0% of the time** (vs 16.9% for Apertus).
  Teacher rollouts that were truncated and still graded as best-available
  are baked into the chosen side of teacher-fallback pairs. Worth
  re-examining whether `chosen` ever ends mid-think due to teacher
  truncation.
- **Tool-use content metric is still ≤ 4% even with `tools=` threaded.** The
  fix moved emission from 0% → 28.8% but did not move correctness much — the
  model genuinely can't pick the right tool + args, not just the right
  format. Teacher fallback is doing the heavy lifting for the 820 tool-use
  prompts in all-fail.

## 7. Suggested next slices

Working from `logs/rollouts_blend_passk_apertus_think_8k_10k_verified.jsonl`:

- Cross-tabulate `agent_ref` × `(thinking_closed, finish_reason)` to see
  whether unclosed-thinking concentrates in any particular subset (intuition:
  reasoning + mopd should dominate, but quantify).
- Among `top_5gram_count >= 10` rollouts, extract the modal 5-gram per
  prompt — gives a catalog of "stuck phrases" that DPO needs to dispreference.
- Plot `chosen_n_tokens` vs `rejected_n_tokens` per pair in
  `dpo_pairs_pilot_clean.jsonl`, split by `chosen_source`. Confirms whether
  the length-match filter actually held in the teacher-fallback half.
- For `single_step_tool_use` prompts that did emit `<|tools_prefix|>` but
  failed content: extract the tool name the model picked vs. the expected
  one. Is there a systematic bias toward a small set of tools?
