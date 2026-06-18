# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hard-verifier sweep over the pass@k rollouts.

Joins each rollout in ``rollouts_blend_passk_apertus_think_8k.jsonl`` to its
source row in the cached HF blend JSONLs (by prompt text), looks up the
``agent_ref.name``, and dispatches to a pure-Python verifier when the agent is
rule-based. Augments each row with:

  agent_ref         : the dispatch key from the source
  content_verifiable: bool — we have a rule-based verifier for this agent
  content_correct   : bool|null — true/false if verifiable, null otherwise
  verifier_reason   : string|null — diagnostic when we couldn't verify

Skipped (LLM-judge or agentic): multichallenge, inverse_if, abstention, genrm*,
equivalence_llm_judge, math_with_judge, swe_pivot*, nvarc_*, reasoning_gym,
jailbreak_*, math_formal_lean. The skip set comes from the schema audit — these
agents all carry ``judge_prompt_template`` / ``rubric`` (natural-language
criteria) so a hard verifier can't reproduce them.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Optional deps — install via the slurm wrapper. We gate each verifier on its
# corresponding library so a missing install only disables one verifier instead
# of failing the whole sweep.
try:
    from verifiable_instructions import instructions_registry as _ifeval_registry  # type: ignore
except ImportError:
    _ifeval_registry = None

try:
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


INNER_SUFFIX = "<|inner_suffix|>"
ASSISTANT_END = "<|assistant_end|>"
TOOLS_PREFIX = "<|tools_prefix|>"
TOOLS_SUFFIX = "<|tools_suffix|>"

_TOOLS_BLOCK_RE = re.compile(
    re.escape(TOOLS_PREFIX) + r"(.*?)" + re.escape(TOOLS_SUFFIX), re.DOTALL
)
_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py|json|yaml)?\s*\n(.*?)\n```", re.DOTALL
)
_BOXED_RE = re.compile(r"\\boxed\{\s*([^}]+?)\s*\}")
_ANSWER_COLON_RE = re.compile(r"(?i)answer\s*:\s*([A-Za-z0-9_\-+.]+)")


def extract_answer(generation_text_full):
    """Strip the thinking block + assistant_end suffix to get the model's final answer."""
    text = generation_text_full
    if INNER_SUFFIX in text:
        text = text.split(INNER_SUFFIX, 1)[1]
    text = text.split(ASSISTANT_END, 1)[0]
    return text.strip()


def extract_tool_calls(generation_text_full):
    """Apertus chat template emits tool calls as
    ``<|tools_prefix|>[{"name1": <args_json>}, ...]<|tools_suffix|>``.
    """
    out = []
    for m in _TOOLS_BLOCK_RE.finditer(generation_text_full):
        body = m.group(1).strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            # Best-effort: try wrapping in brackets if it looks like a bare object
            try:
                parsed = json.loads("[" + body + "]")
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, list):
            out.extend(parsed)
        elif isinstance(parsed, dict):
            out.append(parsed)
    return out


# ============== Verifiers ==============

def verify_instruction_following(rollout, src):
    if _ifeval_registry is None:
        return None, "verifiable_instructions not installed"
    answer = extract_answer(rollout["generation_text_full"])
    id_list = src.get("instruction_id_list") or []
    kw_list = src.get("kwargs") or []
    if not id_list:
        return None, "no instruction_id_list"
    if len(kw_list) < len(id_list):
        kw_list = list(kw_list) + [None] * (len(id_list) - len(kw_list))
    checks = []
    for iid, kw in zip(id_list, kw_list):
        try:
            cls = _ifeval_registry.INSTRUCTION_DICT.get(iid)
            if cls is None:
                checks.append(False)
                continue
            inst = cls(iid)
            filtered = {k: v for k, v in (kw or {}).items() if v is not None}
            inst.build_description(**filtered)
            checks.append(bool(inst.check_following(answer)))
        except Exception:
            checks.append(False)
    return all(checks), None


def verify_mcqa(rollout, src):
    expected = src.get("expected_answer")
    if not expected:
        return None, "no expected_answer"
    expected_upper = str(expected).strip().upper()
    answer = extract_answer(rollout["generation_text_full"])
    # Prefer \boxed{X}, then "Answer: X" — matches Gym's lenient parsers.
    for matcher in (_BOXED_RE, _ANSWER_COLON_RE):
        m = matcher.search(answer)
        if m:
            extracted = m.group(1).strip().upper()
            if not extracted:
                continue
            # Some mcqa expected_answer fields are full strings; compare as
            # case-insensitive substring if length differs significantly.
            if expected_upper == extracted:
                return True, None
            if len(expected_upper) <= 4 and len(extracted) <= 4:
                # Letter choices: must match exactly
                return False, None
            # Otherwise tolerate substring
            return expected_upper in extracted or extracted in expected_upper, None
    return False, None


def verify_structured_outputs(rollout, src):
    schema_str = src.get("schema_str")
    if not schema_str:
        return None, "no schema_str"
    try:
        schema = json.loads(schema_str)
    except json.JSONDecodeError:
        return None, "schema not valid json"
    schema_type = (src.get("schema_type") or "json").lower()
    answer = extract_answer(rollout["generation_text_full"])

    # Find a code block first (most reliable), then fall back to a curly-brace span.
    block_m = _CODE_BLOCK_RE.search(answer)
    candidate = block_m.group(1) if block_m else answer

    data = None
    if schema_type == "yaml":
        if yaml is None:
            return None, "yaml not installed"
        try:
            data = yaml.safe_load(candidate)
        except Exception:
            return False, None
    else:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            obj_m = re.search(r"\{.*\}", candidate, re.DOTALL)
            if not obj_m:
                return False, None
            try:
                data = json.loads(obj_m.group(0))
            except json.JSONDecodeError:
                return False, None

    if data is None:
        return False, None
    if jsonschema is None:
        return True, "schema-parse-only (jsonschema missing)"
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except jsonschema.ValidationError:
        return False, None
    except Exception as e:
        return None, f"jsonschema error: {type(e).__name__}"


def _args_equal(a, b):
    """Compare tool arguments after JSON-normalizing both sides."""
    def _norm(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return x
        return x
    return _norm(a) == _norm(b)


def verify_tool_use_comparison(rollout, src):
    expected = src.get("expected_action") or {}
    if expected.get("type") != "function_call":
        return None, "expected_action is not function_call"
    exp_name = expected.get("name", "")
    exp_args = expected.get("arguments", "{}")
    try:
        exp_args = json.loads(exp_args) if isinstance(exp_args, str) else exp_args
    except json.JSONDecodeError:
        exp_args = {}

    calls = extract_tool_calls(rollout["generation_text_full"])
    if not calls:
        return False, None

    for call in calls:
        if not isinstance(call, dict):
            continue
        # Standard format: {"name": "...", "arguments": ...}
        if "name" in call and "arguments" in call:
            if call["name"] == exp_name and _args_equal(call["arguments"], exp_args):
                return True, None
        # Apertus's compact format: {"name1": <args>}
        for k, v in call.items():
            if k in ("name", "arguments"):
                continue
            if k == exp_name and _args_equal(v, exp_args):
                return True, None
    return False, None


def _run_one_codegen_test(args):
    code, stdin, expected_stdout = args
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    return result.stdout.strip() == expected_stdout.strip()


def verify_code_gen(rollout, src):
    vm = src.get("verifier_metadata") or {}
    tests = vm.get("unit_tests") or {}
    inputs = tests.get("inputs") or []
    outputs = tests.get("outputs") or []
    if not inputs or not outputs or len(inputs) != len(outputs):
        return None, "no/mismatched unit tests"
    answer = extract_answer(rollout["generation_text_full"])
    block_m = _CODE_BLOCK_RE.search(answer)
    if not block_m:
        return False, None
    code = block_m.group(1)
    if not code.strip():
        return False, None
    # All tests must pass.
    for i, o in zip(inputs, outputs):
        if not _run_one_codegen_test((code, i, o)):
            return False, None
    return True, None


# ============== Dispatch ==============

VERIFIERS = {
    "instruction_following_simple_agent": verify_instruction_following,
    "mcqa_simple_agent": verify_mcqa,
    "structured_outputs_simple_agent": verify_structured_outputs,
    "structured_outputs_v3_simple_agent": verify_structured_outputs,
    "single_step_tool_use_with_argument_comparison_agent": verify_tool_use_comparison,
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent": verify_tool_use_comparison,
    "code_gen_simple_agent": verify_code_gen,
}


# ============== Source-row index ==============

def build_prompt_index(cache_root, configs):
    """Map (config, user_prompt_text) -> source row."""
    out = {}
    for cfg in configs:
        paths = glob.glob(f"{cache_root}/*/{cfg}.jsonl")
        if not paths:
            print(f"[index] {cfg}: no cached file (skipping)", flush=True)
            continue
        path = paths[0]
        n = 0
        with open(path) as f:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--passk-jsonl",
        type=Path,
        default=Path("logs/rollouts_blend_passk_apertus_think_8k.jsonl"),
    )
    parser.add_argument(
        "--source-cache",
        default="/iopsstor/scratch/cscs/nathanrchn/.cache/huggingface/hub/datasets--nvidia--Nemotron-RL-Ultra-Training-Blends/snapshots",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/rollouts_blend_passk_apertus_think_8k_verified.jsonl"),
    )
    parser.add_argument("--configs", default="ifbench,reasoning,rlhf,mopd,rlvr1,rlvr2")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    print(f"Indexing source JSONLs from {args.source_cache} ...", flush=True)
    prompt_to_src = build_prompt_index(args.source_cache, configs)
    print(f"Total indexed: {len(prompt_to_src)}", flush=True)
    print()

    # Library availability report up front so the slurm log shows which
    # verifiers actually ran with their full backing libraries.
    print(f"Library availability:")
    print(f"  verifiable_instructions (IFEval): {'YES' if _ifeval_registry else 'NO'}")
    print(f"  jsonschema:                       {'YES' if jsonschema else 'NO'}")
    print(f"  yaml:                             {'YES' if yaml else 'NO'}")
    print()

    counts = defaultdict(lambda: {"total": 0, "verified": 0, "correct": 0, "skipped_reason": defaultdict(int)})
    overall = {"total": 0, "verified": 0, "correct": 0}
    n_rows = 0
    n_unmatched = 0

    with args.passk_jsonl.open() as fin, args.out.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            content_correct = None
            verifier_reason = None
            agent = None

            src = prompt_to_src.get((row["config"], row["prompt"]))
            if src is None:
                n_unmatched += 1
                verifier_reason = "source row not matched"
            else:
                agent = (src.get("agent_ref") or {}).get("name")
                verifier = VERIFIERS.get(agent)
                if verifier is not None:
                    try:
                        content_correct, verifier_reason = verifier(row, src)
                    except Exception as e:
                        content_correct = None
                        verifier_reason = f"verifier exception: {type(e).__name__}: {e}"
                else:
                    verifier_reason = "no rule-based verifier for this agent"

            row["agent_ref"] = agent
            row["content_verifiable"] = content_correct is not None
            row["content_correct"] = content_correct
            if verifier_reason:
                row["verifier_reason"] = verifier_reason
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            overall["total"] += 1
            agent_key = agent or "<unmatched>"
            counts[agent_key]["total"] += 1
            if content_correct is None:
                if verifier_reason:
                    # Bucket by reason for diagnostics.
                    counts[agent_key]["skipped_reason"][verifier_reason] += 1
            else:
                counts[agent_key]["verified"] += 1
                overall["verified"] += 1
                if content_correct:
                    counts[agent_key]["correct"] += 1
                    overall["correct"] += 1

            if n_rows % 500 == 0:
                print(
                    f"  ... processed {n_rows} rows  "
                    f"(verified so far: {overall['verified']}, correct: {overall['correct']})",
                    flush=True,
                )

    print()
    print(f"=" * 80)
    print(f"SUMMARY  (total rows {n_rows}, unmatched {n_unmatched})")
    print(f"=" * 80)
    print(
        f"  verifiable rows : {overall['verified']:>5} / {overall['total']:>5}  "
        f"({overall['verified']/max(1,overall['total'])*100:.1f}%)"
    )
    print(
        f"  content_correct : {overall['correct']:>5} / {overall['verified']:>5}  "
        f"({overall['correct']/max(1,overall['verified'])*100:.1f}%)"
    )
    print()
    print("Per-agent (sorted by verified count):")
    for agent_key, c in sorted(counts.items(), key=lambda kv: -kv[1]["verified"]):
        if c["verified"] > 0:
            print(
                f"  {agent_key:<60s} verified {c['verified']:>4}/{c['total']:<4}  "
                f"correct {c['correct']:>4}/{c['verified']:<4} "
                f"({c['correct']/max(1,c['verified'])*100:5.1f}%)"
            )

    print()
    print("Top skip-reasons (no verifier dispatched):")
    skip_counter = defaultdict(int)
    for c in counts.values():
        for reason, n in c["skipped_reason"].items():
            skip_counter[reason] += n
    for reason, n in sorted(skip_counter.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:>5}  {reason}")

    return 0


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    sys.exit(main())
