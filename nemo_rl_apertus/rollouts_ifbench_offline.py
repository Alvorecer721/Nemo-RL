# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline vLLM rollouts of Apertus 1.5 SFT against ifbench prompts.

Loads the ifbench split of nvidia/Nemotron-RL-Ultra-Training-Blends, slices it
by ``--rank`` / ``--world-size`` (so each node owns a contiguous shard), and
generates one completion per prompt with ``enable_thinking=True`` on the chat
template. Token IDs (not text) drive the thinking-emitted / thinking-closed
flags so we catch leakage even when the tokenizer hides specials in decode.

Why ``hf_hub_download`` + manual JSONL parse: the ifbench blend's ``id`` column
mixes numeric and string types and the Arrow JSON parser used by
``datasets.load_dataset`` refuses the schema mid-batch. See
``nemo_rl_apertus/data/download_ifbench_prompts.py`` for the same workaround.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"  # pragma: allowlist secret
DEFAULT_TOKENIZER = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611"

DEFAULT_REPO = "nvidia/Nemotron-RL-Ultra-Training-Blends"
DEFAULT_CONFIG = "ifbench"

INNER_PREFIX = "<|inner_prefix|>"
INNER_SUFFIX = "<|inner_suffix|>"
ASSISTANT_END = "<|assistant_end|>"


def load_prompts(repo: str, config: str, n: int) -> list[str]:
    """Pull ``ifbench.jsonl`` from the Hub and return the first ``n`` user prompts.

    The order is the source file order: we want a deterministic, reproducible
    slicing across ranks (rank 0 gets prompts 0..2499, rank 1 gets 2500..4999,
    etc.). No shuffling — every rank just reads the same file and slices.
    """
    src = hf_hub_download(repo_id=repo, filename=f"{config}.jsonl", repo_type="dataset")

    prompts: list[str] = []
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
            prompts.append(content)
            if len(prompts) >= n:
                break

    if len(prompts) < n:
        raise RuntimeError(
            f"ifbench JSONL yielded only {len(prompts)} usable prompts, requested {n}."
        )
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--n-prompts", type=int, default=10000, help="Total prompts (all ranks combined).")
    parser.add_argument("--rank", type=int, required=True, help="This rank's index in [0, world-size).")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=12288,
        help="Headroom over max-tokens for prompt + generation.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    print(
        f"[rank {args.rank}/{args.world_size}] owns prompt indices [{start}, {stop}) "
        f"(per_rank={per_rank}, total={args.n_prompts})",
        flush=True,
    )

    all_prompts = load_prompts(args.repo, args.config, args.n_prompts)
    my_prompts = all_prompts[start:stop]
    print(f"[rank {args.rank}] loaded {len(my_prompts)} prompts.", flush=True)

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
    for prompt in my_prompts:
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

    print(
        f"[rank {args.rank}] sampling: temperature={args.temperature} top_p={args.top_p} "
        f"max_tokens={args.max_tokens} stop_token_ids={stop_token_ids}",
        flush=True,
    )

    outputs = llm.generate(rendered, sampling)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for local_idx, (prompt, result) in enumerate(zip(my_prompts, outputs)):
            out = result.outputs[0]
            token_ids = list(out.token_ids)
            text_clean = out.text
            text_full = tokenizer.decode(token_ids, skip_special_tokens=False)
            row = {
                "prompt_idx": start + local_idx,
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

    print(f"[rank {args.rank}] wrote {len(my_prompts)} rows → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
