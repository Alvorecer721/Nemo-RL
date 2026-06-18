# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apertus 1.5 SFT vibe test: detect thinking-when-disabled, doom-loop, Chinese drift.

Runs a diverse prompt set through vLLM offline against the Apertus SFT checkpoint
referenced in the GRPO probe recipe, with the chat template rendered using
`enable_thinking=False` by default (the failure mode is the model emitting
`<|inner_prefix|>` regardless). Token IDs (not text) drive the detectors so we
catch special-token leakage even when the tokenizer hides them in decode.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"  # pragma: allowlist secret
DEFAULT_TOKENIZER = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611"

INNER_PREFIX = "<|inner_prefix|>"
INNER_SUFFIX = "<|inner_suffix|>"
ASSISTANT_END = "<|assistant_end|>"

# CJK Unified Ideographs (Chinese characters live here; Japanese kanji overlaps,
# Korean hangul doesn't). Hiragana/Katakana would be U+3040-U+30FF.
CJK_RE = re.compile(r"[一-鿿㐀-䶿]")

# Prompt language tag: "en"/"fr"/"de"/"es" → no CJK expected in output.
# "zh" → CJK expected. "multi" → mixed, exempt from CJK-drift check.
PROMPTS: list[tuple[str, str]] = [
    # English factual
    ("en", "What is the capital of Switzerland?"),
    ("en", "Explain photosynthesis in one paragraph."),
    ("en", "Why is the sky blue?"),
    ("en", "How many planets are in our solar system?"),
    ("en", "Name three famous Swiss mountains."),
    # Doom-loop bait: long open-ended
    ("en", "List 30 interesting facts about the universe."),
    ("en", "Write a 200-word story about a wizard discovering a magic library."),
    ("en", "Continue this sentence in 5 paragraphs: The cat sat on the mat and"),
    # Code generation
    ("en", "Write a Python function to compute the nth Fibonacci number."),
    ("en", "Implement bubble sort in JavaScript with comments."),
    ("en", "Write a Rust function that reverses a string."),
    # Reasoning / math: often triggers stealth-thinking
    ("en", "What is 17 times 24? Just give me the number."),
    (
        "en",
        "Solve: a train leaves Zurich at 9am going 100 km/h; another leaves Geneva at 10am going 120 km/h. They head toward each other on a 250 km route. When do they meet?",
    ),
    ("en", "Prove that the square root of 2 is irrational."),
    ("en", "If a shirt costs 25 CHF after a 20% discount, what was the original price?"),
    # Non-Chinese multilingual: stay in source language, no CJK drift
    ("fr", "Bonjour ! Réponds-moi en français : quelle est la capitale de la France ?"),
    ("fr", "Explique-moi ce qu'est la photosynthèse en une phrase."),
    ("de", "Wie geht es dir? Antworte auf Deutsch."),
    ("de", "Was ist die Hauptstadt der Schweiz?"),
    ("es", "Hola, ¿cuál es la capital de España?"),
    ("it", "Ciao, qual è la capitale d'Italia?"),
    # Chinese in: Chinese out is fine
    ("zh", "你好，请用中文回答：瑞士的首都是什么？"),
    ("zh", "用中文写一首关于秋天的俳句。"),
    # Instruction following
    ("en", "Explain quantum entanglement to a 10-year-old in 3 sentences."),
    ("en", "Pretend you are a pirate. In two sentences, describe your typical morning."),
    ("en", "List exactly five animals that live in the Arctic, numbered 1-5."),
    ("en", "Write a haiku about autumn."),
    # Repetition bait
    ("en", "Repeat the word 'banana' exactly five times, then stop."),
    ("en", "Count from 1 to 10. Stop after 10."),
    # Edge cases / short
    ("en", "Hi"),
    ("en", "What comes after Saturday?"),
    ("en", "Define entropy in one sentence."),
    # Long-form summary
    ("en", "Write a 100-word summary of the French Revolution."),
    ("en", "Compare Python and Rust in 3 bullet points."),
]


def count_cjk(text: str) -> int:
    return len(CJK_RE.findall(text))


def top_ngram(token_ids: list[int], n: int) -> tuple[int, tuple[int, ...]]:
    if len(token_ids) < n:
        return 0, ()
    counter = Counter(tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1))
    ngram, count = counter.most_common(1)[0]
    return count, ngram


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Render chat template with enable_thinking=True (default False).",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size."
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rep-threshold", type=int, default=6, help="Flag if top 5-gram occurs >= this many times.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    inner_prefix_id = tokenizer.convert_tokens_to_ids(INNER_PREFIX)
    inner_suffix_id = tokenizer.convert_tokens_to_ids(INNER_SUFFIX)
    assistant_end_id = tokenizer.convert_tokens_to_ids(ASSISTANT_END)
    eos_id = tokenizer.eos_token_id

    print(f"Token IDs: inner_prefix={inner_prefix_id} inner_suffix={inner_suffix_id} "
          f"assistant_end={assistant_end_id} eos={eos_id}", flush=True)

    rendered_prompts: list[str] = []
    for _, prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        rendered_prompts.append(text)
    print(f"Rendered {len(rendered_prompts)} prompts. enable_thinking={args.enable_thinking}", flush=True)
    print(f"First rendered prompt (sanity):\n{rendered_prompts[0][:400]}\n...", flush=True)

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=False,
    )

    stop_token_ids = [tid for tid in (assistant_end_id, eos_id) if tid is not None]
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids,
        seed=42,
    )

    outputs = llm.generate(rendered_prompts, sampling)

    counters = {
        "thinking_emitted_when_disabled": 0,
        "thinking_never_closed": 0,
        "chinese_drift": 0,
        "doom_loop_5gram": 0,
        "hit_max_tokens": 0,
    }
    rows: list[dict] = []

    print()
    print("=" * 80)
    print(f"PER-PROMPT RESULTS  (enable_thinking={args.enable_thinking})")
    print("=" * 80)

    for (lang, prompt), result in zip(PROMPTS, outputs):
        out = result.outputs[0]
        token_ids = list(out.token_ids)
        text_clean = out.text  # detokenized; specials may or may not be filtered
        text_full = tokenizer.decode(token_ids, skip_special_tokens=False)
        finish = out.finish_reason

        flags: list[str] = []
        thinking_emitted = inner_prefix_id in token_ids
        thinking_closed = inner_suffix_id in token_ids
        cjk_in_text = count_cjk(text_clean)
        rep_count, rep_ngram = top_ngram(token_ids, n=5)
        hit_max = finish == "length"

        if thinking_emitted and not args.enable_thinking:
            flags.append("THINKING_EMITTED_WHEN_DISABLED")
            counters["thinking_emitted_when_disabled"] += 1
        if thinking_emitted and not thinking_closed:
            flags.append("THINKING_NEVER_CLOSED")
            counters["thinking_never_closed"] += 1
        if lang != "zh" and cjk_in_text > 0:
            flags.append(f"CHINESE_DRIFT({cjk_in_text})")
            counters["chinese_drift"] += 1
        if rep_count >= args.rep_threshold:
            flags.append(f"DOOM_LOOP(5g×{rep_count})")
            counters["doom_loop_5gram"] += 1
        if hit_max:
            flags.append("HIT_MAX_TOKENS")
            counters["hit_max_tokens"] += 1

        rows.append(
            {
                "lang": lang,
                "prompt": prompt,
                "text_clean": text_clean,
                "text_full": text_full,
                "n_tokens": len(token_ids),
                "finish_reason": finish,
                "thinking_emitted": thinking_emitted,
                "thinking_closed": thinking_closed,
                "cjk_chars_in_text": cjk_in_text,
                "top_5gram_count": rep_count,
                "top_5gram_ids": list(rep_ngram),
                "flags": flags,
            }
        )

        status = "PASS" if not flags else "FAIL: " + " ".join(flags)
        print()
        print(f"[{status}]  lang={lang}  n_tok={len(token_ids)}  finish={finish}")
        print(f"  PROMPT: {prompt[:140]}")
        snippet = text_clean.replace("\n", " ")[:200]
        print(f"  OUTPUT: {snippet}{'…' if len(text_clean) > 200 else ''}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(PROMPTS)
    print()
    print("=" * 80)
    print(f"SUMMARY (n={n}, enable_thinking={args.enable_thinking}, temp={args.temperature})")
    print("=" * 80)
    for k, v in counters.items():
        print(f"  {k:36s} {v:3d} / {n}")
    print(f"  output_jsonl                         {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
