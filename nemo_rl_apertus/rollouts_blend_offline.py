# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline vLLM rollouts of Apertus 1.5 SFT against a blend of NVIDIA's
Nemotron-RL-Ultra-Training-Blends subsets (ifbench, reasoning, rlhf, swe, mopd,
rlvr1, rlvr2 — all 7 .jsonl files at the repo root).

Why ``hf_hub_download`` (downloads the full file, caches in HF_HOME) over
HTTP streaming: the RL-VR files are 5 GB each and a long ``requests`` stream
from four concurrent ranks fails partway with a ``ChunkedEncodingError``
(observed once: ``IncompleteRead(448103759 bytes read, 115062097 more
expected)``). ``hf_hub_download`` has resumable chunked downloads + cross-rank
file locking, so all four ranks share one cached blob. First job pays ~16 GB
of disk + a few minutes of download; later jobs cache-hit instantly.

After each subset's prompts are read, the unified pool is deterministically
shuffled (seed=42) so each rank gets a *mix* across subsets, not a contiguous
single-source block.

Same Apertus chat-template + token-ID detector shape as
``rollouts_ifbench_offline.py``; the only delta is the prompt source.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"  # pragma: allowlist secret
DEFAULT_TOKENIZER = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611"

DEFAULT_REPO = "nvidia/Nemotron-RL-Ultra-Training-Blends"
DEFAULT_CONFIGS = "ifbench,reasoning,rlhf,mopd,rlvr1,rlvr2"
# swe.jsonl is excluded by default: its `responses_create_params.input` is
# empty (the prompt lives at `responses_create_params.metadata.problem_statement`
# for SWE-bench-style agentic tasks), so the generic user-content parser yields
# zero prompts. Pass --configs ...,swe explicitly to include it once a custom
# schema branch is added.

INNER_PREFIX = "<|inner_prefix|>"
INNER_SUFFIX = "<|inner_suffix|>"
ASSISTANT_END = "<|assistant_end|>"


def load_subset_prompts(repo: str, config: str, n: int) -> list[tuple[str, str]]:
    """Pull ``<config>.jsonl`` via ``hf_hub_download`` (cache-aware) and return up to
    ``n`` ``(config, prompt)`` pairs.

    The local cached file is shared across ranks via HF Hub's file-locking, so a
    multi-rank job won't re-download the same blob four times.
    """
    src = hf_hub_download(repo_id=repo, filename=f"{config}.jsonl", repo_type="dataset")
    out: list[tuple[str, str]] = []
    with open(src) as f:
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
            content = user_msgs[0].get("content")
            if not content:
                continue
            out.append((config, content))
            if len(out) >= n:
                break
    return out


def load_blend(repo: str, configs: list[str], n_total: int, seed: int) -> list[tuple[str, str]]:
    """Build the unified blend prompt pool: equal-ish split across configs, then shuffle."""
    n = len(configs)
    base = n_total // n
    rem = n_total % n
    quotas = [base + (1 if i < rem else 0) for i in range(n)]

    pool: list[tuple[str, str]] = []
    for cfg, want in zip(configs, quotas):
        got = load_subset_prompts(repo, cfg, want)
        print(f"[load_blend] {cfg}: requested {want}, got {len(got)}", flush=True)
        pool.extend(got)
        if len(got) < want:
            raise RuntimeError(
                f"{cfg} only yielded {len(got)} usable prompts, needed {want}."
            )

    # Deterministic shuffle so each rank's contiguous slice ends up subset-mixed.
    random.Random(seed).shuffle(pool)
    if len(pool) != n_total:
        raise RuntimeError(
            f"blend pool has {len(pool)} prompts, expected {n_total}."
        )
    return pool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--configs",
        default=DEFAULT_CONFIGS,
        help="Comma-separated list of subset names (each <name>.jsonl must exist at the repo root).",
    )
    parser.add_argument("--n-prompts", type=int, default=10000, help="Total prompts (all ranks combined).")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-shuffle-seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True, help="Per-rank JSONL output path.")
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

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    print(
        f"[rank {args.rank}/{args.world_size}] configs={configs} owns prompt indices [{start}, {stop})",
        flush=True,
    )

    all_pairs = load_blend(args.repo, configs, args.n_prompts, args.blend_shuffle_seed)
    my_pairs = all_pairs[start:stop]
    from collections import Counter

    print(
        f"[rank {args.rank}] my shard config distribution: {Counter(c for c, _ in my_pairs)}",
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

    rendered: list[str] = []
    for _, prompt in my_pairs:
        messages = [{"role": "user", "content": prompt}]
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )
    print(
        f"[rank {args.rank}] first rendered prompt (sanity):\n{rendered[0][:400]}\n...",
        flush=True,
    )

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
        seed=args.seed + args.rank,
    )

    outputs = llm.generate(rendered, sampling)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for local_idx, ((cfg, prompt), result) in enumerate(zip(my_pairs, outputs)):
            out = result.outputs[0]
            token_ids = list(out.token_ids)
            text_clean = out.text
            text_full = tokenizer.decode(token_ids, skip_special_tokens=False)
            row = {
                "prompt_idx": start + local_idx,
                "config": cfg,
                "prompt": prompt,
                "generation_text": text_clean,
                "generation_text_full": text_full,
                "n_prompt_tokens": len(result.prompt_token_ids),
                "n_output_tokens": len(token_ids),
                "finish_reason": out.finish_reason,
                "thinking_emitted": inner_prefix_id in token_ids,
                "thinking_closed": inner_suffix_id in token_ids,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[rank {args.rank}] wrote {len(my_pairs)} rows → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
