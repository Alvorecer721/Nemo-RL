# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract the prompts where Apertus has zero correct samples in pass@k.

Reads a verified pass@k JSONL (one row per generation, with ``is_correct`` and
optionally ``content_correct`` flags) and emits a small JSONL of unique prompts
(one row per prompt_idx) suitable for the Qwen teacher rollout script.

Two definitions of "all-fail":
  --mode format            : zero ``is_correct`` samples (default — picks up the doom-loop / format failures)
  --mode content_or_format : if the prompt is content-verifiable, require zero
                             ``content_correct``; else fall back to format.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--passk-jsonl",
        type=Path,
        required=True,
        help="Verified pass@k JSONL (post-verify_passk_offline).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Per-prompt JSONL of all-fail prompts.",
    )
    parser.add_argument(
        "--mode",
        choices=["format", "content_or_format"],
        default="format",
    )
    args = parser.parse_args()

    by_prompt = defaultdict(list)
    meta = {}
    with open(args.passk_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r["prompt_idx"]
            by_prompt[pid].append(r)
            meta[pid] = {
                "config": r["config"],
                "prompt": r["prompt"],
                "agent_ref": r.get("agent_ref"),
            }

    def is_chosen(row):
        if args.mode == "format":
            return bool(row.get("is_correct"))
        # content_or_format
        if row.get("content_verifiable"):
            return row.get("content_correct") is True
        return bool(row.get("is_correct"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_all_fail = 0
    by_agent = defaultdict(int)
    with args.out.open("w") as f:
        for pid, rows in by_prompt.items():
            n_total += 1
            if any(is_chosen(r) for r in rows):
                continue
            n_all_fail += 1
            m = meta[pid]
            by_agent[m.get("agent_ref") or "<unknown>"] += 1
            f.write(
                json.dumps(
                    {"prompt_idx": pid, "config": m["config"], "prompt": m["prompt"], "agent_ref": m.get("agent_ref")},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Total prompts: {n_total}")
    print(f"All-fail (mode={args.mode}): {n_all_fail} ({n_all_fail / max(1, n_total) * 100:.1f}%)")
    print(f"Wrote {n_all_fail} all-fail prompts → {args.out}")
    print()
    print("Breakdown by agent:")
    for ag, n in sorted(by_agent.items(), key=lambda kv: -kv[1]):
        print(f"  {ag:<60} {n}")


if __name__ == "__main__":
    sys.exit(main())
