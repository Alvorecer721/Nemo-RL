# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-rollout the tool-use prompts from the pass@8 sample with ``tools=`` actually
threaded through the chat template.

Background: the original ``rollouts_blend_passk_offline.py`` didn't pass
``tools=`` to ``apply_chat_template``, so the Apertus template rendered
``Tool Capabilities: disabled`` for every prompt — including the tool-use ones.
The model accordingly produced **zero** ``<|tools_prefix|>`` blocks on those
prompts (0/824). This script re-runs only the affected prompts with their
``responses_create_params.tools`` correctly attached.

Output schema is identical to ``rollouts_blend_passk_offline.py`` so the
existing verifier + downstream tools work unchanged.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"  # pragma: allowlist secret
DEFAULT_TOKENIZER = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611"
DEFAULT_SOURCE_CACHE = "/iopsstor/scratch/cscs/nathanrchn/.cache/huggingface/hub/datasets--nvidia--Nemotron-RL-Ultra-Training-Blends/snapshots"

DEFAULT_AGENTS = ",".join([
    "single_step_tool_use_with_argument_comparison_agent",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
])

INNER_PREFIX = "<|inner_prefix|>"
INNER_SUFFIX = "<|inner_suffix|>"
ASSISTANT_END = "<|assistant_end|>"


def _stringify_none(obj):
    """Recursively replace ``None`` with ``""`` in tool-spec dicts.

    The Apertus chat template's ``render_tools`` macro concatenates field values
    directly (e.g. ``"// " + tool.description``), so a ``None`` description /
    parameter description / etc. raises ``TypeError`` mid-render. Empty string
    is the right substitute: the macro still emits valid syntax just with a
    blank description.
    """
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {k: _stringify_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_none(x) for x in obj]
    return obj


def normalize_tools(tools):
    """Apply the None→\"\" replacement to a list of tool specs (or return None if empty)."""
    if not tools:
        return None
    return [_stringify_none(t) for t in tools]


def build_source_index(cache_root: str, configs):
    """Index source rows by ``(config, first_user_prompt_text)``."""
    out = {}
    for cfg in configs:
        paths = glob.glob(f"{cache_root}/*/{cfg}.jsonl")
        if not paths:
            print(f"[index] {cfg}: no cached file (skipping)", flush=True)
            continue
        with open(paths[0]) as f:
            n = 0
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
                if txt:
                    out[(cfg, txt)] = rec
                    n += 1
        print(f"[index] {cfg}: indexed {n} prompts", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--source-cache", default=DEFAULT_SOURCE_CACHE)
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=Path("logs/rollouts_blend_passk_apertus_think_8k.jsonl"),
        help="Original pass@8 JSONL — we read its (prompt_idx, config, prompt) tuples to know which prompts to re-roll.",
    )
    parser.add_argument(
        "--agents",
        default=DEFAULT_AGENTS,
        help="Comma-separated list of agent_ref.name values to re-roll.",
    )
    parser.add_argument(
        "--configs",
        default="ifbench,reasoning,rlhf,mopd,rlvr1,rlvr2",
    )
    parser.add_argument("--n-per-prompt", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    target_agents = {a.strip() for a in args.agents.split(",") if a.strip()}
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    # 1) Index source rows so we can join agent_ref + tools by prompt text.
    src_index = build_source_index(args.source_cache, configs)

    # 2) Walk the original pass@8 JSONL, collect unique (prompt_idx, config, prompt)
    #    rows whose source has one of the target agents.
    print(f"Collecting target prompts from {args.source_jsonl} ...", flush=True)
    seen = set()
    targets = []  # list of (prompt_idx, config, prompt, tools)
    with open(args.source_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r["prompt_idx"]
            if pid in seen:
                continue
            src = src_index.get((r["config"], r["prompt"]))
            if src is None:
                continue
            agent = (src.get("agent_ref") or {}).get("name")
            if agent not in target_agents:
                continue
            tools = (src.get("responses_create_params") or {}).get("tools") or None
            targets.append((pid, r["config"], r["prompt"], normalize_tools(tools), agent))
            seen.add(pid)

    n_per_agent = Counter(t[4] for t in targets)
    print(f"Target prompts: {len(targets)}")
    for agent, n in n_per_agent.items():
        print(f"  {agent}: {n}")

    if not targets:
        print("No target prompts found; nothing to do.")
        return 0

    # 3) Tokenizer + IDs.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    inner_prefix_id = tokenizer.convert_tokens_to_ids(INNER_PREFIX)
    inner_suffix_id = tokenizer.convert_tokens_to_ids(INNER_SUFFIX)
    assistant_end_id = tokenizer.convert_tokens_to_ids(ASSISTANT_END)
    eos_id = tokenizer.eos_token_id
    print(
        f"Token IDs: inner_prefix={inner_prefix_id} inner_suffix={inner_suffix_id} "
        f"assistant_end={assistant_end_id} eos={eos_id}",
        flush=True,
    )

    # 4) Render chat template with tools threaded through.
    rendered = []
    n_with_tools = 0
    for (_, _, prompt, tools, _) in targets:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            tools=tools,
        )
        rendered.append(text)
        if tools:
            n_with_tools += 1
    print(
        f"Rendered {len(rendered)} prompts; {n_with_tools} carry tools (the rest had none in source).",
        flush=True,
    )
    print(f"Sanity — first rendered prompt fragment:\n{rendered[0][:500]}\n...", flush=True)

    # 5) vLLM generate, n samples per prompt.
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
        seed=args.seed,
    )
    print(f"Sampling: n={args.n_per_prompt} temp={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens}", flush=True)

    outputs = llm.generate(rendered, sampling)

    # 6) Write per-sample rows in the same schema as the original pass@8 file.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_tool_blocks = 0
    with args.out.open("w") as f:
        for (pid, cfg, prompt, _, agent), result in zip(targets, outputs):
            for sample_idx, out in enumerate(result.outputs):
                token_ids = list(out.token_ids)
                text_clean = out.text
                text_full = tokenizer.decode(token_ids, skip_special_tokens=False)
                if "<|tools_prefix|>" in text_full:
                    n_tool_blocks += 1
                row = {
                    "prompt_idx": pid,
                    "sample_idx": sample_idx,
                    "config": cfg,
                    "prompt": prompt,
                    "generation_text": text_clean,
                    "generation_text_full": text_full,
                    "n_output_tokens": len(token_ids),
                    "finish_reason": out.finish_reason,
                    "thinking_emitted": inner_prefix_id in token_ids,
                    "thinking_closed": inner_suffix_id in token_ids,
                    "agent_ref": agent,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"Wrote {n_rows} rows → {args.out}", flush=True)
    print(f"Generated tool-call blocks: {n_tool_blocks} / {n_rows} ({n_tool_blocks/max(1,n_rows)*100:.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
