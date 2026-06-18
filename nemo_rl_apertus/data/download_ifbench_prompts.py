# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample N user prompts from the public ifbench subset of NVIDIA's
Nemotron-RL-Ultra-Training-Blends and write them to a NeMo-RL-friendly JSONL.

Why ifbench: instruction-following constraints ("start and end with the same
word, sentence 13 word 13 = `computer`, exactly 2 paragraphs") are exactly the
prompts that empirically tickle Apertus 1.5 SFT's analysis-paralysis / no-EOS
failure mode, so they give the format reward a non-zero gradient.

The agent/judge/rubric metadata in the source dataset is intentionally dropped:
this recipe scores structural well-formedness, not content correctness.
"""

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO = "nvidia/Nemotron-RL-Ultra-Training-Blends"
DEFAULT_CONFIG = "ifbench"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # `datasets.load_dataset(..., "ifbench")` (streaming or not) fails on this
    # blend: the `id` column is heterogeneous (numeric in some rows, string in
    # others), and the Arrow JSON parser refuses the schema mid-batch. Pull the
    # raw .jsonl with huggingface_hub and parse it ourselves — no schema
    # enforcement, no surprises.
    src = hf_hub_download(
        repo_id=args.repo,
        filename=f"{args.config}.jsonl",
        repo_type="dataset",
    )

    collected: list[dict] = []
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
            sys_msgs = [m for m in inputs if m.get("role") == "system"]
            if not user_msgs:
                continue
            prompt = user_msgs[0].get("content")
            if not prompt:
                continue
            record = {"prompt": prompt}
            if sys_msgs and sys_msgs[0].get("content"):
                record["system"] = sys_msgs[0]["content"]
            collected.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(collected)
    take = collected[: args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in take:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(take)} prompts (sampled from {len(collected)} valid rows) "
        f"from {args.repo}/{args.config}.jsonl → {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
