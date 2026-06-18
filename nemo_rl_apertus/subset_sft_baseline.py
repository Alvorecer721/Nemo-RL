# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filter the SFT-base 10K pass@8/verified JSONL to the (config, prompt) subset
that the 2500-prompt DPO rollouts actually evaluated. Gives an equal-prompt
baseline for direct delta computation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dpo-jsonl",
        type=Path,
        required=True,
        help="One of the new (2500-prompt) DPO pass@8 outputs; defines the prompt set.",
    )
    p.add_argument(
        "--sft-jsonl",
        type=Path,
        required=True,
        help="SFT-base 10K pass@8 (or verified) JSONL to subset.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    keys = set()
    with open(args.dpo_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            keys.add((r["config"], r["prompt"]))
    print(f"[subset] {len(keys)} unique (config,prompt) keys from {args.dpo_jsonl}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    with open(args.sft_jsonl) as f, args.out.open("w") as out:
        for line in f:
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            if (r.get("config"), r.get("prompt")) in keys:
                out.write(line)
                kept += 1
    print(f"[subset] kept {kept} / {total} rows → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
