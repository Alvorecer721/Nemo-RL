# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate pass@8 + verifier outputs across multiple DPO pilot checkpoints.

For each input file pair (passk + verified), prints format/structural metrics,
doom-loop bucketing, and per-agent content correctness. Then prints
side-by-side deltas across checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CJK = re.compile(r"[一-鿿]")


def load_passk(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def topline(rows):
    n = len(rows)
    if n == 0:
        return {}
    emit = sum(1 for r in rows if r.get("thinking_emitted"))
    closed = sum(1 for r in rows if r.get("thinking_closed"))
    iscor = sum(1 for r in rows if r.get("is_correct"))
    fr = Counter(r.get("finish_reason", "?") for r in rows)
    tok_sum = sum(r.get("n_output_tokens", 0) for r in rows)
    return {
        "n": n,
        "emit": emit / n,
        "closed": closed / n,
        "is_correct": iscor / n,
        "fr_length": fr.get("length", 0) / n,
        "fr_stop": fr.get("stop", 0) / n,
        "mean_toks": tok_sum / n,
    }


def doom_buckets(rows):
    n = len(rows)
    if n == 0:
        return {}
    severe = sum(1 for r in rows if r["ttr"] < 0.10 and r["n_output_tokens"] >= 50)
    degen = sum(
        1
        for r in rows
        if 0.10 <= r["ttr"] < 0.20 and r["n_output_tokens"] >= 50
    )
    pattern = sum(1 for r in rows if r["top_5gram_count"] >= 10)
    cjk = sum(
        1
        for r in rows
        if CJK.search(r["generation_text"]) and not CJK.search(r["prompt"])
    )
    return {
        "n": n,
        "severe": severe / n,
        "degen": degen / n,
        "pattern": pattern / n,
        "cjk": cjk / n,
    }


def per_agent(rows):
    agg = defaultdict(lambda: {"n": 0, "v": 0, "c": 0})
    for r in rows:
        a = r.get("agent_ref") or "<unmatched>"
        agg[a]["n"] += 1
        if r.get("content_verifiable"):
            agg[a]["v"] += 1
            if r.get("content_correct"):
                agg[a]["c"] += 1
    return agg


def per_prompt_pass(rows):
    by_prompt = defaultdict(list)
    for r in rows:
        if r.get("content_verifiable"):
            by_prompt[r["prompt_idx"]].append(bool(r.get("content_correct")))
    verifiable = {p: s for p, s in by_prompt.items() if s}
    n = len(verifiable)
    if n == 0:
        return None
    p1 = sum(1 for p, s in verifiable.items() if s[0]) / n
    p8 = sum(1 for p, s in verifiable.items() if any(s)) / n
    allp = sum(1 for p, s in verifiable.items() if all(s)) / n
    allf = sum(1 for p, s in verifiable.items() if not any(s)) / n
    return {
        "n_prompts": n,
        "pass1": p1,
        "pass8": p8,
        "allpass": allp,
        "allfail": allf,
        "mixed": 1 - allp - allf,
    }


def per_config_breakdown(rows):
    """Per-config (subset) format + content stats. Pass@8 over rows that have
    a verifier; for non-verifiable configs, only show format metrics."""
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r.get("config", "?")].append(r)
    out = {}
    for cfg, rs in by_cfg.items():
        n = len(rs)
        closed = sum(1 for r in rs if r.get("thinking_closed")) / n
        iscor = sum(1 for r in rs if r.get("is_correct")) / n
        fr_len = sum(1 for r in rs if r.get("finish_reason") == "length") / n
        mean_toks = sum(r.get("n_output_tokens", 0) for r in rs) / n
        v = [r for r in rs if r.get("content_verifiable")]
        if v:
            cc = sum(1 for r in v if r.get("content_correct")) / len(v)
        else:
            cc = None
        # per-prompt for this config
        by_p = defaultdict(list)
        for r in rs:
            if r.get("content_verifiable"):
                by_p[r["prompt_idx"]].append(bool(r.get("content_correct")))
        passes = {p: s for p, s in by_p.items() if s}
        if passes:
            p1 = sum(1 for p, s in passes.items() if s[0]) / len(passes)
            p8 = sum(1 for p, s in passes.items() if any(s)) / len(passes)
        else:
            p1 = p8 = None
        out[cfg] = {
            "n": n,
            "closed": closed,
            "is_correct": iscor,
            "fr_length": fr_len,
            "mean_toks": mean_toks,
            "cc_rate": cc,
            "n_verifiable": len(v),
            "pass1": p1,
            "pass8": p8,
        }
    return out


def fmt_pct(x):
    return f"{x*100:6.2f}%" if x is not None else "    n/a"


def print_topline_block(label, tl):
    print(f"  {label}: n={tl['n']}")
    print(f"    thinking_emitted : {fmt_pct(tl['emit'])}")
    print(f"    thinking_closed  : {fmt_pct(tl['closed'])}")
    print(f"    is_correct (fmt) : {fmt_pct(tl['is_correct'])}")
    print(f"    finish=length    : {fmt_pct(tl['fr_length'])}")
    print(f"    finish=stop      : {fmt_pct(tl['fr_stop'])}")
    print(f"    mean output toks : {tl['mean_toks']:6.0f}")


def print_doom_block(label, db):
    print(f"  {label}")
    print(f"    severe TTR<0.10  : {fmt_pct(db['severe'])}")
    print(f"    degen 0.10..0.20 : {fmt_pct(db['degen'])}")
    print(f"    pattern-lock     : {fmt_pct(db['pattern'])}")
    print(f"    CJK drift        : {fmt_pct(db['cjk'])}")


def print_per_agent(label, agg):
    print(f"  {label}")
    rows = [(a, d["n"], d["v"], d["c"]) for a, d in agg.items() if d["v"] > 0]
    rows.sort(key=lambda r: -r[2])
    print(f"    {'agent':<55} {'n':>6} {'verif':>7} {'corr':>6} {'%':>7}")
    for a, n, v, c in rows:
        print(f"    {a:<55} {n:>6} {v:>7} {c:>6} {c/max(1,v)*100:>6.2f}%")


def print_per_config(label, breakdown):
    print(f"  {label}")
    print(
        f"    {'config':<14} {'n':>6} {'closed':>8} {'is_c':>8} {'fr=L':>8} "
        f"{'cc':>8} {'p@1':>8} {'p@8':>8} {'mtoks':>6}"
    )
    for cfg in sorted(breakdown.keys()):
        d = breakdown[cfg]
        print(
            f"    {cfg:<14} {d['n']:>6} {fmt_pct(d['closed'])} "
            f"{fmt_pct(d['is_correct'])} {fmt_pct(d['fr_length'])} "
            f"{fmt_pct(d['cc_rate'])} {fmt_pct(d['pass1'])} "
            f"{fmt_pct(d['pass8'])} {d['mean_toks']:>6.0f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pair",
        action="append",
        required=True,
        help="label:passk.jsonl:verified.jsonl (repeatable)",
    )
    args = p.parse_args()

    checkpoints = []
    for spec in args.pair:
        label, passk_p, ver_p = spec.split(":")
        passk_path = Path(passk_p)
        ver_path = Path(ver_p) if ver_p else None
        passk_rows = load_passk(passk_path) if passk_path.exists() else []
        ver_rows = load_passk(ver_path) if ver_path and ver_path.exists() else []
        checkpoints.append((label, passk_rows, ver_rows))

    for label, passk_rows, ver_rows in checkpoints:
        print()
        print("=" * 70)
        print(f"CHECKPOINT: {label}")
        print(f"  passk rows    : {len(passk_rows)}")
        print(f"  verified rows : {len(ver_rows)}")
        print("=" * 70)
        if not passk_rows:
            print("  (no passk data yet)")
            continue
        tl = topline(passk_rows)
        print("\n[topline format metrics]")
        print_topline_block(label, tl)
        print("\n[doom-loop bucketing]")
        db = doom_buckets(passk_rows)
        print_doom_block(label, db)
        if ver_rows:
            print("\n[per-agent content correctness]")
            agg = per_agent(ver_rows)
            print_per_agent(label, agg)
            print("\n[per-prompt pass rates (verifiable)]")
            pp = per_prompt_pass(ver_rows)
            if pp:
                print(f"  verifiable prompts : {pp['n_prompts']}")
                print(f"  pass@1   : {fmt_pct(pp['pass1'])}")
                print(f"  pass@8   : {fmt_pct(pp['pass8'])}")
                print(f"  all-pass : {fmt_pct(pp['allpass'])}")
                print(f"  all-fail : {fmt_pct(pp['allfail'])}")
                print(f"  MIXED    : {fmt_pct(pp['mixed'])}")
            print("\n[per-config breakdown]")
            br = per_config_breakdown(ver_rows)
            print_per_config(label, br)
        else:
            print("\n[per-config breakdown — format only, verifier pending]")
            br = per_config_breakdown(passk_rows)
            print_per_config(label, br)

    # Side-by-side comparison if multiple checkpoints
    if len(checkpoints) > 1:
        print()
        print("=" * 70)
        print("CROSS-CHECKPOINT DIFF")
        print("=" * 70)
        print(
            f"\n{'metric':<22} "
            + " ".join(f"{c[0]:>14}" for c in checkpoints)
        )
        names = [
            ("thinking_closed", "closed"),
            ("is_correct (fmt)", "is_correct"),
            ("finish=length", "fr_length"),
            ("mean output toks", "mean_toks"),
        ]
        tls = [topline(c[1]) for c in checkpoints]
        for name, key in names:
            cells = []
            for tl in tls:
                if not tl:
                    cells.append(" " * 14)
                elif key == "mean_toks":
                    cells.append(f"{tl[key]:>14.0f}")
                else:
                    cells.append(f"{tl[key]*100:>13.2f}%")
            print(f"  {name:<20} " + " ".join(cells))
        print()
        dbs = [doom_buckets(c[1]) for c in checkpoints]
        for name, key in [
            ("severe doom", "severe"),
            ("degen", "degen"),
            ("pattern-lock", "pattern"),
            ("CJK drift", "cjk"),
        ]:
            cells = [f"{d[key]*100:>13.2f}%" if d else " " * 14 for d in dbs]
            print(f"  {name:<20} " + " ".join(cells))
        # Content-correctness side-by-side
        if any(c[2] for c in checkpoints):
            print()
            pps = [per_prompt_pass(c[2]) if c[2] else None for c in checkpoints]
            for name, key in [
                ("pass@1 (any)", "pass1"),
                ("pass@8 (any)", "pass8"),
                ("all-pass prompts", "allpass"),
                ("MIXED prompts", "mixed"),
                ("all-fail prompts", "allfail"),
            ]:
                cells = [
                    f"{pp[key]*100:>13.2f}%" if pp else " " * 14 for pp in pps
                ]
                print(f"  {name:<20} " + " ".join(cells))


if __name__ == "__main__":
    sys.exit(main())
