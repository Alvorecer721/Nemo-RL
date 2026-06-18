# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pass@k offline vLLM rollouts on the blend prompts.

Picks ~``--n-prompts`` rows stratified-by-config from the prior single-shot
blend output (``rollouts_blend_apertus_think_8k.jsonl``), then generates
``--n-per-prompt`` completions for each via vLLM's native ``SamplingParams.n``.
Writes one row per generation with the TTR / pattern-lock / unclosed-thinking
metrics inline so the pass@k analysis can group by ``prompt_idx`` without
re-tokenizing.

"Correct" rollout (the thing the format reward wants):
  - ``finish_reason == "stop"`` (no max-token runaway)
  - ``thinking_closed`` true (no chat-template violation)
  - ``ttr >= 0.20`` (no doom-loop)
  - ``top_5gram_count < 10`` (no pattern-lock)

Per-prompt mix of pass/fail across the 8 samples gives the gradient signal
GRPO leave-one-out needs — all-pass and all-fail prompts contribute nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"  # pragma: allowlist secret
DEFAULT_TOKENIZER = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611"
DEFAULT_SOURCE_CACHE = "/iopsstor/scratch/cscs/nathanrchn/.cache/huggingface/hub/datasets--nvidia--Nemotron-RL-Ultra-Training-Blends/snapshots"

INNER_PREFIX = "<|inner_prefix|>"
INNER_SUFFIX = "<|inner_suffix|>"
ASSISTANT_END = "<|assistant_end|>"


def _stringify_none(obj):
    """Recursively replace ``None`` with ``""``.

    The Apertus chat template's ``render_tools`` macro concatenates field values
    directly (e.g. ``"// " + tool.description``), so a ``None`` description /
    param description / etc. raises ``TypeError`` mid-render.
    """
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {k: _stringify_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_none(x) for x in obj]
    return obj


def build_tool_index(cache_root, configs):
    """Map (config, first_user_prompt) -> normalized tools list (or None)."""
    out = {}
    for cfg in configs:
        paths = glob.glob(f"{cache_root}/*/{cfg}.jsonl")
        if not paths:
            print(f"[tool-index] {cfg}: no cached file", flush=True)
            continue
        n = 0
        with open(paths[0]) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                params = rec.get("responses_create_params") or {}
                inputs = params.get("input") or []
                user_msgs = [m for m in inputs if m.get("role") == "user"]
                if not user_msgs:
                    continue
                txt = user_msgs[0].get("content")
                if not txt:
                    continue
                tools = params.get("tools") or None
                if tools:
                    out[(cfg, txt)] = _stringify_none(tools)
                else:
                    out[(cfg, txt)] = None
                n += 1
        print(f"[tool-index] {cfg}: {n} entries", flush=True)
    return out

WORD_RE = re.compile(r"\w+|[一-鿿]", re.UNICODE)


def stratified_sample(records, n_total, seed):
    """Deterministic config-stratified sample: per-config quota proportional to source share."""
    by_cfg = defaultdict(list)
    for rec in records:
        by_cfg[rec["config"]].append(rec)
    cfgs = sorted(by_cfg.keys())
    total = sum(len(by_cfg[c]) for c in cfgs)
    rng = random.Random(seed)
    quotas = {}
    assigned = 0
    for c in cfgs[:-1]:
        q = round(n_total * len(by_cfg[c]) / total)
        quotas[c] = q
        assigned += q
    quotas[cfgs[-1]] = n_total - assigned
    out = []
    for c in cfgs:
        pool = by_cfg[c]
        rng.shuffle(pool)
        take = pool[: quotas[c]]
        out.extend(take)
    rng.shuffle(out)
    return out


def text_metrics(text):
    """Word-level TTR + top 5-gram repeat count (mirrors analyze script semantics)."""
    words = WORD_RE.findall(text)
    n = len(words)
    if n == 0:
        return 1.0, 0
    ttr = len(set(words)) / n
    if n < 5:
        return ttr, 0
    grams = Counter(tuple(words[i : i + 5]) for i in range(n - 4))
    return ttr, max(grams.values())


def is_correct(thinking_closed, finish_reason, ttr, top5):
    return (
        thinking_closed
        and finish_reason == "stop"
        and ttr >= 0.20
        and top5 < 10
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=Path("logs/rollouts_blend_apertus_think_8k.jsonl"),
        help="Prior single-shot blend output; we sample its prompts.",
    )
    parser.add_argument("--n-prompts", type=int, default=1280, help="Stratified prompt count (total across all ranks).")
    parser.add_argument("--n-per-prompt", type=int, default=8, help="Samples per prompt — the k in pass@k.")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--gen-seed", type=int, default=42, help="Added to rank for per-rank vLLM RNG.")
    parser.add_argument(
        "--source-cache",
        default=DEFAULT_SOURCE_CACHE,
        help="HF cache root containing the Nemotron-RL-Ultra-Training-Blends snapshots; used to fetch tool specs per prompt.",
    )
    parser.add_argument(
        "--configs",
        default="ifbench,reasoning,rlhf,mopd,rlvr1,rlvr2",
        help="Which blend subsets to index for tool specs.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"rank {args.rank} not in [0, {args.world_size})")
    if args.n_prompts % args.world_size != 0:
        raise ValueError(
            f"n-prompts {args.n_prompts} must be divisible by world-size {args.world_size}"
        )

    per_rank = args.n_prompts // args.world_size
    start = args.rank * per_rank
    stop = start + per_rank

    # Load + stratified sample on every rank (deterministic given the same seed).
    src_records = []
    with open(args.source_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("prompt") and rec.get("config"):
                src_records.append({"config": rec["config"], "prompt": rec["prompt"]})
    print(f"[rank {args.rank}] loaded {len(src_records)} source records", flush=True)

    sampled = stratified_sample(src_records, args.n_prompts, args.sample_seed)
    my_slice = sampled[start:stop]
    cfg_dist = Counter(r["config"] for r in my_slice)
    print(
        f"[rank {args.rank}/{args.world_size}] owns prompts [{start}, {stop}); "
        f"config distribution: {dict(cfg_dist)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    inner_prefix_id = tokenizer.convert_tokens_to_ids(INNER_PREFIX)
    inner_suffix_id = tokenizer.convert_tokens_to_ids(INNER_SUFFIX)
    assistant_end_id = tokenizer.convert_tokens_to_ids(ASSISTANT_END)
    eos_id = tokenizer.eos_token_id
    print(
        f"[rank {args.rank}] token IDs: inner_prefix={inner_prefix_id} "
        f"inner_suffix={inner_suffix_id} assistant_end={assistant_end_id} eos={eos_id}",
        flush=True,
    )

    # Build tool index so tool-use prompts actually see their tools — otherwise
    # the Apertus chat template renders ``Tool Capabilities: disabled`` and the
    # model never emits ``<|tools_prefix|>`` blocks.
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    tool_index = build_tool_index(args.source_cache, configs)
    print(f"[rank {args.rank}] tool index size: {len(tool_index)}", flush=True)

    rendered = []
    n_with_tools = 0
    for rec in my_slice:
        messages = [{"role": "user", "content": rec["prompt"]}]
        tools = tool_index.get((rec["config"], rec["prompt"]))
        if tools:
            n_with_tools += 1
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                tools=tools,
            )
        )
    print(f"[rank {args.rank}] rendered {len(rendered)} prompts; {n_with_tools} carry tools", flush=True)

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
        n=args.n_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids,
        seed=args.gen_seed + args.rank,
    )
    print(
        f"[rank {args.rank}] sampling: n={args.n_per_prompt} temp={args.temperature} "
        f"top_p={args.top_p} max_tokens={args.max_tokens} stop_token_ids={stop_token_ids}",
        flush=True,
    )

    outputs = llm.generate(rendered, sampling)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with args.out.open("w") as f:
        for local_idx, (rec, result) in enumerate(zip(my_slice, outputs)):
            global_prompt_idx = start + local_idx
            for sample_idx, out in enumerate(result.outputs):
                token_ids = list(out.token_ids)
                text_clean = out.text
                text_full = tokenizer.decode(token_ids, skip_special_tokens=False)
                ttr, top5 = text_metrics(text_clean)
                thinking_emitted = inner_prefix_id in token_ids
                thinking_closed = inner_suffix_id in token_ids
                row = {
                    "prompt_idx": global_prompt_idx,
                    "sample_idx": sample_idx,
                    "config": rec["config"],
                    "prompt": rec["prompt"],
                    "generation_text": text_clean,
                    "generation_text_full": text_full,
                    "n_output_tokens": len(token_ids),
                    "finish_reason": out.finish_reason,
                    "thinking_emitted": thinking_emitted,
                    "thinking_closed": thinking_closed,
                    "ttr": ttr,
                    "top_5gram_count": top5,
                    "is_correct": is_correct(thinking_closed, out.finish_reason, ttr, top5),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"[rank {args.rank}] wrote {n_rows} rows ({len(my_slice)} prompts × {args.n_per_prompt}) → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
