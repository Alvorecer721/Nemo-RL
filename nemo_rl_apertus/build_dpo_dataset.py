# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a DPO preference dataset from Apertus pass@k + Qwen teacher rollouts.

Inputs
------
1. ``--apertus-jsonl``   : Apertus pass@k file (one row per generation, with
                          ``is_correct`` format flag and optionally
                          ``content_correct`` from the verifier sweep).
2. ``--qwen-jsonl``      : Qwen teacher rollouts on the all-fail subset
                          (one row per generation; carries ``apertus_body`` —
                          the Qwen output already format-converted).

Output
------
JSONL of preference pairs:
  {
    "prompt_idx", "config", "prompt",
    "chosen", "rejected",
    "chosen_source": "apertus" | "qwen",
    "chosen_n_tokens", "rejected_n_tokens",
    "agent_ref" (if known),
  }

Strategy
--------
For each prompt:
  1. If Apertus has ≥1 sample matching ``--chosen-pred`` AND ≥1 sample matching
     ``--rejected-pred``: emit on-policy pair(s) (chosen=Apertus, rejected=Apertus).
  2. Else if Qwen has a sample (anything), pair Qwen-as-chosen with the worst
     Apertus as rejected.
  3. Else skip (no usable preference).

``--pair-mode`` chooses ``one`` (1 pair per prompt: best chosen × worst rejected)
or ``all`` (every chosen × every rejected combination). ``--length-match`` keeps
only pairs whose chosen/rejected n_output_tokens are within ±``--length-ratio``
of each other (default 0.5, i.e. neither side more than 1.5× the other) to limit
the length-collapse hazard of DPO.

``--chosen-pred`` and ``--rejected-pred`` are simple expressions over the
per-sample fields. Defaults capture the format reward: chosen = ``is_correct``
true (closed thinking + stop + TTR ≥ 0.20 + 5-gram < 10), rejected = the negation.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _len_match(a_tokens, b_tokens, ratio):
    if a_tokens <= 0 or b_tokens <= 0:
        return False
    lo = min(a_tokens, b_tokens)
    hi = max(a_tokens, b_tokens)
    return (hi - lo) / hi <= ratio


def _row_to_chosen_text(row, source):
    """Extract the chosen-string for DPO from a row.

    Apertus rows: the assistant body the model generated is ``generation_text_full``
    minus the trailing terminator the model emitted; DPO concatenates with the
    prompt's ``add_generation_prompt`` output, so we return the body without
    the prompt portion. We use ``generation_text_full`` (keeps specials) and
    append ``<|assistant_end|>`` to ensure the terminator is uniform.

    Qwen rows: the ``apertus_body`` field is already converted + terminated.
    """
    if source == "qwen":
        return row["apertus_body"]
    # Apertus path
    text = row.get("generation_text_full", "")
    # Strip any trailing assistant_end if vLLM included it on stop, then re-add
    # one canonically so the chosen string is always terminated identically.
    if text.endswith("<|assistant_end|>"):
        text = text[: -len("<|assistant_end|>")]
    return text.rstrip() + "<|assistant_end|>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apertus-jsonl",
        type=Path,
        required=True,
        help="Apertus pass@k rollouts.",
    )
    parser.add_argument(
        "--qwen-jsonl",
        type=Path,
        default=None,
        help="Optional Qwen teacher rollouts on the all-fail subset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the DPO pair JSONL.",
    )
    parser.add_argument(
        "--chosen-pred",
        choices=["format", "content", "content_or_format"],
        default="format",
        help="Apertus 'chosen' predicate. format = is_correct (closed + stop + TTR + ngram). "
        "content = content_correct (hard verifier). "
        "content_or_format = is_correct AND content_correct (when verifiable), else is_correct.",
    )
    parser.add_argument(
        "--rejected-pred",
        choices=["any_not_chosen", "format_broken", "content_wrong"],
        default="any_not_chosen",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["all", "one"],
        default="all",
        help="all = every (chosen × rejected) combination per prompt; one = single best/worst pair.",
    )
    parser.add_argument(
        "--length-ratio",
        type=float,
        default=0.5,
        help="Reject pairs where (max-min)/max > ratio (length-collapse guard).",
    )
    parser.add_argument(
        "--no-length-match",
        dest="length_match",
        action="store_false",
        default=True,
    )
    parser.add_argument("--max-pairs-per-prompt", type=int, default=64)
    args = parser.parse_args()

    # Load Apertus rollouts
    apertus_by_prompt = defaultdict(list)
    apertus_meta = {}
    with open(args.apertus_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r["prompt_idx"]
            apertus_by_prompt[pid].append(r)
            apertus_meta[pid] = {"config": r["config"], "prompt": r["prompt"], "agent_ref": r.get("agent_ref")}
    print(f"Loaded Apertus pass@k: {len(apertus_by_prompt)} prompts, "
          f"{sum(len(v) for v in apertus_by_prompt.values())} samples", flush=True)

    # Load Qwen teacher rollouts (optional)
    qwen_by_prompt = defaultdict(list)
    if args.qwen_jsonl:
        with open(args.qwen_jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = r["prompt_idx"]
                qwen_by_prompt[pid].append(r)
        print(f"Loaded Qwen teacher: {len(qwen_by_prompt)} prompts, "
              f"{sum(len(v) for v in qwen_by_prompt.values())} samples", flush=True)

    # Predicates
    def is_chosen(row):
        if args.chosen_pred == "format":
            return bool(row.get("is_correct"))
        if args.chosen_pred == "content":
            return row.get("content_correct") is True
        # content_or_format: when verifiable, require both; else just format.
        verif = row.get("content_verifiable", False)
        if verif:
            return bool(row.get("is_correct")) and row.get("content_correct") is True
        return bool(row.get("is_correct"))

    def is_rejected(row):
        if args.rejected_pred == "any_not_chosen":
            return not is_chosen(row)
        if args.rejected_pred == "format_broken":
            return not bool(row.get("is_correct"))
        # content_wrong
        return row.get("content_correct") is False

    # Build pairs
    pairs = []
    counts = {
        "prompts_total": 0,
        "prompts_with_pair": 0,
        "apertus_only_pairs": 0,
        "teacher_only_pairs": 0,
        "skipped_no_chosen_no_qwen": 0,
        "skipped_no_rejected": 0,
        "length_filtered": 0,
    }
    for pid, rows in apertus_by_prompt.items():
        counts["prompts_total"] += 1
        meta = apertus_meta[pid]
        chosens = [r for r in rows if is_chosen(r)]
        rejecteds = [r for r in rows if is_rejected(r)]

        # Sort chosens by token count desc (prefer longer "good" outputs to
        # discourage length-collapse) and rejecteds by token count asc.
        chosens.sort(key=lambda r: -r.get("n_output_tokens", 0))
        rejecteds.sort(key=lambda r: r.get("n_output_tokens", 0))

        chosen_source = "apertus"
        if not chosens:
            qwens = qwen_by_prompt.get(pid, [])
            if not qwens:
                counts["skipped_no_chosen_no_qwen"] += 1
                continue
            chosens = qwens
            chosen_source = "qwen"
        if not rejecteds:
            counts["skipped_no_rejected"] += 1
            continue

        local_pairs = []
        if args.pair_mode == "one":
            c = chosens[0]
            r = rejecteds[0]
            if args.length_match and not _len_match(c.get("n_output_tokens", 0), r.get("n_output_tokens", 0), args.length_ratio):
                counts["length_filtered"] += 1
                continue
            local_pairs.append((c, r))
        else:
            for c in chosens:
                for r in rejecteds:
                    if args.length_match and not _len_match(c.get("n_output_tokens", 0), r.get("n_output_tokens", 0), args.length_ratio):
                        continue
                    local_pairs.append((c, r))
                    if len(local_pairs) >= args.max_pairs_per_prompt:
                        break
                if len(local_pairs) >= args.max_pairs_per_prompt:
                    break
            if not local_pairs:
                counts["length_filtered"] += 1
                continue

        counts["prompts_with_pair"] += 1
        for c_row, r_row in local_pairs:
            pair = {
                "prompt_idx": pid,
                "config": meta["config"],
                "prompt": meta["prompt"],
                "agent_ref": meta.get("agent_ref"),
                "chosen": _row_to_chosen_text(c_row, chosen_source),
                "rejected": _row_to_chosen_text(r_row, "apertus"),
                "chosen_source": chosen_source,
                "chosen_n_tokens": c_row.get("n_output_tokens"),
                "rejected_n_tokens": r_row.get("n_output_tokens"),
            }
            pairs.append(pair)
        if chosen_source == "apertus":
            counts["apertus_only_pairs"] += len(local_pairs)
        else:
            counts["teacher_only_pairs"] += len(local_pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print()
    print(f"=== DPO pair dataset built ===")
    print(f"Wrote {len(pairs)} pairs → {args.out}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
