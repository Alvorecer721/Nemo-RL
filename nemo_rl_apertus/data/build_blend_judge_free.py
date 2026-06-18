# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a balanced, judge-free slice of Nemotron-RL-Ultra-Training-Blends for
mixed-env GRPO smoke testing.

We keep only rows whose ``agent_ref.name`` resolves to an agent we have
a NeMo-Gym config for *and* that doesn't require an LLM judge / external
infra. Everything else (genrm, equivalence_llm_judge, swe, math_with_judge,
ns_tools, multichallenge, jailbreak, abstention, math_formal_lean, nvarc_*,
rdkit, reasoning_gym, etc.) is dropped.

Each output line is a raw blend row, preserving ``agent_ref`` and
``responses_create_params`` — the schema ``NemoGymDataset`` expects.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "nvidia/Nemotron-RL-Ultra-Training-Blends"
DEFAULT_CONFIGS = "ifbench,reasoning,rlhf,mopd,rlvr1,rlvr2"

JUDGE_FREE_AGENTS = (
    "instruction_following_simple_agent",
    "mcqa_simple_agent",
    "structured_outputs_simple_agent",
    "code_gen_simple_agent",
    "single_step_tool_use_with_argument_comparison_agent",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
    "calendar_simple_agent",
)


def _stringify_none(obj):
    # Apertus chat template concatenates ``tool.description`` and friends
    # directly; a None in any tool field raises at render time.
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {k: _stringify_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_none(x) for x in obj]
    return obj


def _normalize_row(row: dict) -> dict:
    p = row.get("responses_create_params") or {}
    if p.get("tools"):
        p["tools"] = _stringify_none(p["tools"])
        row["responses_create_params"] = p
    return row


def _load_blend_rows(repo: str, config: str) -> list[dict]:
    src = hf_hub_download(repo_id=repo, filename=f"{config}.jsonl", repo_type="dataset")
    rows: list[dict] = []
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS, help="comma-separated")
    ap.add_argument("--per-agent", type=int, default=80,
                    help="Target rows per agent (cap; some agents may have fewer)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    bucket: dict[str, list[dict]] = defaultdict(list)

    for cfg in configs:
        print(f"[load] {cfg}.jsonl", file=sys.stderr)
        rows = _load_blend_rows(args.repo, cfg)
        for r in rows:
            ar = r.get("agent_ref") or {}
            name = ar.get("name")
            if name in JUDGE_FREE_AGENTS:
                bucket[name].append(r)

    selected: list[dict] = []
    for agent in JUDGE_FREE_AGENTS:
        rows = bucket.get(agent, [])
        rng.shuffle(rows)
        take = rows[: args.per_agent]
        print(f"[select] {agent}: {len(take)}/{len(rows)} available", file=sys.stderr)
        selected.extend(take)

    rng.shuffle(selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in selected:
            f.write(json.dumps(_normalize_row(r)) + "\n")

    final = Counter(r["agent_ref"]["name"] for r in selected)
    print(f"[done] wrote {len(selected)} rows → {args.out}", file=sys.stderr)
    for k, v in final.most_common():
        print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
