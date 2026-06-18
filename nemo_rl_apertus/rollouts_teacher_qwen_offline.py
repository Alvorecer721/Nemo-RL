# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate teacher rollouts on a list of prompts using Qwen3.5-9B.

Reads a JSONL of ``{prompt_idx, config, prompt, agent_ref(opt), expected_action(opt)}``
records — typically the all-fail subset of the Apertus pass@8 sweep — and
generates ``--n-per-prompt`` completions per row with the configured Qwen
chat template (thinking enabled, tools threaded when present).

Each output row carries both the raw Qwen text *and* the Apertus-format
conversion (from ``convert_qwen_body_to_apertus``) so the DPO pair builder
downstream can drop in the converted chosen string verbatim.

The Qwen3.5 family is published as a multimodal model
(image-text-to-text). Text-only inference is supported; we don't pass image
inputs so vLLM treats it as a plain LLM.
"""

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from nemo_rl_apertus.convert_qwen_to_apertus import convert_qwen_body_to_apertus

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_SOURCE_CACHE = "/iopsstor/scratch/cscs/nathanrchn/.cache/huggingface/hub/datasets--nvidia--Nemotron-RL-Ultra-Training-Blends/snapshots"


def _stringify_none(obj):
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {k: _stringify_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_none(x) for x in obj]
    return obj


def build_tool_index(cache_root, configs):
    """Map (config, first_user_prompt) -> Qwen-compatible tools list (or None).

    Qwen3.x's chat template uses the OpenAI-shaped tool spec — same as what the
    source rows already store under ``responses_create_params.tools`` — so no
    schema conversion is needed; we just normalize ``None`` fields away.
    """
    out = {}
    for cfg in configs:
        paths = glob.glob(f"{cache_root}/*/{cfg}.jsonl")
        if not paths:
            continue
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
                out[(cfg, txt)] = _stringify_none(tools) if tools else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None, help="Defaults to --model.")
    parser.add_argument(
        "--prompts-jsonl",
        type=Path,
        required=True,
        help="JSONL with {prompt_idx, config, prompt} rows — typically the all-fail subset.",
    )
    parser.add_argument("--n-per-prompt", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-cache", default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--configs", default="ifbench,reasoning,rlhf,mopd,rlvr1,rlvr2")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="Pass enable_thinking=True to Qwen's chat template (default).",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.model

    # 1) Load prompts and rank-shard them.
    all_prompts = []
    with open(args.prompts_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("prompt") and r.get("config"):
                all_prompts.append(r)
    print(f"[rank {args.rank}/{args.world_size}] loaded {len(all_prompts)} prompts from {args.prompts_jsonl}", flush=True)

    # Even-as-possible round-robin sharding (no divisibility requirement).
    my_prompts = [p for i, p in enumerate(all_prompts) if i % args.world_size == args.rank]
    print(f"[rank {args.rank}] owns {len(my_prompts)} prompts", flush=True)

    # 2) Build tool index (Qwen consumes the same OpenAI-shaped tools list).
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    tool_index = build_tool_index(args.source_cache, configs)
    print(f"[rank {args.rank}] tool index size: {len(tool_index)}", flush=True)

    # 3) Tokenizer + render.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    rendered = []
    n_with_tools = 0
    for r in my_prompts:
        messages = [{"role": "user", "content": r["prompt"]}]
        tools = tool_index.get((r["config"], r["prompt"]))
        if tools:
            n_with_tools += 1
        # Qwen3.x chat template accepts enable_thinking. Pass tools=None when
        # absent — the template handles both cases.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
            tools=tools,
        )
        rendered.append(text)
    print(
        f"[rank {args.rank}] rendered {len(rendered)} prompts; {n_with_tools} carry tools; "
        f"enable_thinking={args.enable_thinking}",
        flush=True,
    )
    if rendered:
        print(f"[rank {args.rank}] first rendered (sanity):\n{rendered[0][:500]}\n...", flush=True)

    # 4) vLLM.
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        n=args.n_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed + args.rank,
    )
    print(
        f"[rank {args.rank}] sampling: n={args.n_per_prompt} temp={args.temperature} "
        f"top_p={args.top_p} max_tokens={args.max_tokens}",
        flush=True,
    )

    outputs = llm.generate(rendered, sampling)

    # 5) Write per-sample rows. Each row carries the raw Qwen text and the
    # Apertus-format conversion so downstream pair-building is mechanical.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with args.out.open("w") as f:
        for r_in, result in zip(my_prompts, outputs):
            for sample_idx, out in enumerate(result.outputs):
                qwen_body = out.text
                apertus_body = convert_qwen_body_to_apertus(qwen_body)
                row = {
                    "prompt_idx": r_in.get("prompt_idx"),
                    "sample_idx": sample_idx,
                    "config": r_in["config"],
                    "prompt": r_in["prompt"],
                    "qwen_body": qwen_body,
                    "apertus_body": apertus_body,
                    "n_output_tokens": len(out.token_ids),
                    "finish_reason": out.finish_reason,
                    "model": args.model,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
    print(f"[rank {args.rank}] wrote {n_rows} rows → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
