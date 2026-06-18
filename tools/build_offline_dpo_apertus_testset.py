# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate a tiny offline-DPO preference set exercising the Apertus tools/thinking path.

Produces the ``PreferenceDataset`` schema (``{context, completions:[{rank, completion}]}``) that
``ToolThinkingPreferenceProcessor`` (``data.processor: apertus_tool_thinking_preference``) consumes,
covering the full matrix **thinking × tools × multi-turn** so a small ``run_dpo_apertus.py`` probe
hits every branch of detection + developer-block assembly:

* thinking is expressed with the **inline** ``<|inner_prefix|>…<|inner_suffix|>`` markers (string
  ``content`` throughout — so the row schema stays homogeneous and HF ``datasets`` loads it);
* tool **calls** use the message-level ``tool_calls`` field (``arguments`` JSON-encoded) and tool
  **results** are ``role: "tool"`` turns, both kept in *context* (earlier turns) so the final turn
  is a clean assistant reply with a well-defined prompt/response split — except the single-turn tool
  combos, whose trained response *is* the tool call;
* ``enable_thinking`` is set explicitly ``true`` on thinking rows (their inline ``<|inner_prefix|>``
  thinking renders but is not auto-detected); no-think rows alternate explicit ``false`` / omitted
  (auto-detected as ``false`` — no ``thoughts`` block), so the auto-detect path is still exercised;
* ``tools`` schemas are emitted as a **JSON string** per row (``--tools-as-json-string``, default)
  because Arrow can't unify arbitrary ``parameters`` schemas across rows — the matching processor
  decodes them; pass ``--tools-as-json-string=false`` to emit raw lists.

Each datum carries a ``combo`` tag (ignored by the processor) for inspection. Deterministic: no RNG.

Usage::

    python tools/build_offline_dpo_apertus_testset.py --out data/dpo_apertus_testset \
        --n-per-combo 3 --val-fraction 0.25
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

INNER = "<|inner_prefix|>"  # Apertus thinking open
OUTER = "<|inner_suffix|>"  # Apertus thinking close

# Complete OpenAI function specs (name + description + typed parameters) — deliberately varied
# `parameters.properties` shapes, which is exactly what Arrow cannot unify into one struct.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '2 * (3 + 4)'"}
            },
            "required": ["expression"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and return the top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "How many results"},
            },
            "required": ["query"],
        },
    },
}
TOOLS = [WEATHER_TOOL, CALC_TOOL, SEARCH_TOOL]


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _tool_result(text: str) -> dict[str, Any]:
    return {"role": "tool", "content": text}


def _assistant_tool_call(
    name: str, args: dict[str, Any], thinking: Optional[str] = None
) -> dict[str, Any]:
    """Assistant turn that calls a tool (message-level ``tool_calls``; string ``content``)."""
    content = f"{INNER}{thinking}{OUTER}" if thinking else ""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


def _think(thoughts: str, answer: str) -> str:
    """Inline-thinking assistant content: a thought span then the visible answer."""
    return f"{INNER}{thoughts}{OUTER}{answer}"


# Small content banks (indexed by sample number for deterministic variety).
_QA = [
    ("What is the capital of France?", "The capital of France is Paris.", "It is Berlin."),
    ("Name a primary color.", "Red is a primary color.", "Purple is a primary color."),
    ("What is 6 times 7?", "6 times 7 is 42.", "6 times 7 is 36."),
    ("Spell 'cat' backwards.", "Backwards, 'cat' is 'tac'.", "It is 'cta'."),
]
_FOLLOWUP = [
    ("And its population, roughly?", "Paris has roughly 2.1 million people.", "About 50 people."),
    ("Give another one.", "Blue is another primary color.", "Brown is another one."),
    ("And 7 times 8?", "7 times 8 is 56.", "7 times 8 is 49."),
    ("Now spell 'dog' backwards.", "'dog' backwards is 'god'.", "It is 'dgo'."),
]
_THOUGHTS = [
    "Let me recall the fact.",
    "I should reason step by step.",
    "Let me compute carefully.",
    "Think about the letters.",
]


def _combos(i: int) -> list[dict[str, Any]]:
    """Build one datum per combo for sample index ``i`` (8 data — the full matrix)."""
    q, good, bad = _QA[i % len(_QA)]
    fq, fgood, fbad = _FOLLOWUP[i % len(_FOLLOWUP)]
    th = _THOUGHTS[i % len(_THOUGHTS)]
    # Alternate explicit thinking flag vs. omitted (auto-detect) by sample parity.
    explicit = i % 2 == 0
    tool = TOOLS[i % len(TOOLS)]
    tname = tool["function"]["name"]

    def datum(combo, context, chosen, rejected, tools=None, enable_thinking="omit"):
        d: dict[str, Any] = {
            "combo": combo,
            "context": context,
            "completions": [
                {"rank": 0, "completion": chosen},
                {"rank": 1, "completion": rejected},
            ],
        }
        if tools is not None:
            d["tools"] = tools
        if enable_thinking != "omit":
            d["enable_thinking"] = enable_thinking
        return d

    def _args_for(name: str, variant: str) -> dict[str, Any]:
        """Per-tool call arguments for the round-trip context ('rt') and good/bad completions."""
        return {
            "get_weather": {"rt": {"city": "Bern"}, "good": {"city": "Paris"}, "bad": {"city": "Atlantis"}},
            "calculate": {"rt": {"expression": "2+2"}, "good": {"expression": "6*7"}, "bad": {"expression": "6+7"}},
            "web_search": {
                "rt": {"query": q, "top_k": 3},
                "good": {"query": q, "top_k": 3},
                "bad": {"query": "unrelated", "top_k": 1},
            },
        }[name][variant]

    # A closed tool round-trip placed in *context* (turns before the final clean reply).
    rt = [
        _user(q),
        _assistant_tool_call(tname, _args_for(tname, "rt")),
        _tool_result("tool result: ok"),
        _assistant("Here is what I found."),
        _user(fq),
    ]
    good_args = _args_for(tname, "good")
    bad_args = _args_for(tname, "bad")

    return [
        # 1. no-think / no-tool / single
        datum("nothink-notool-single", [_user(q)], [_assistant(good)], [_assistant(bad)],
              enable_thinking=(False if explicit else "omit")),
        # 2. no-think / no-tool / multi
        datum("nothink-notool-multi", [_user(q), _assistant(good), _user(fq)],
              [_assistant(fgood)], [_assistant(fbad)],
              enable_thinking=(False if explicit else "omit")),
        # 3. think / no-tool / single
        datum("think-notool-single", [_user(q)],
              [_assistant(_think(th, good))], [_assistant(_think(th, bad))],
              enable_thinking=True),
        # 4. think / no-tool / multi
        datum("think-notool-multi", [_user(q), _assistant(good), _user(fq)],
              [_assistant(_think(th, fgood))], [_assistant(_think(th, fbad))],
              enable_thinking=True),
        # 5. no-think / tool / single  (trained response IS the tool call)
        datum("nothink-tool-single", [_user(q)],
              [_assistant_tool_call(tname, good_args)], [_assistant_tool_call(tname, bad_args)],
              tools=[tool], enable_thinking=(False if explicit else "omit")),
        # 6. no-think / tool / multi   (tool round-trip in context, final plain reply)
        datum("nothink-tool-multi", rt, [_assistant(fgood)], [_assistant(fbad)],
              tools=[tool], enable_thinking=(False if explicit else "omit")),
        # 7. think / tool / single     (trained response = thinking + tool call)
        datum("think-tool-single", [_user(q)],
              [_assistant_tool_call(tname, good_args, thinking=th)],
              [_assistant_tool_call(tname, bad_args, thinking=th)],
              tools=[tool], enable_thinking=True),
        # 8. think / tool / multi      (tool round-trip in context, final thinking + reply)
        datum("think-tool-multi", rt,
              [_assistant(_think(th, fgood))], [_assistant(_think(th, fbad))],
              tools=[tool], enable_thinking=True),
    ]


def build_rows(n_per_combo: int, tools_as_json_string: bool) -> list[dict[str, Any]]:
    """Build ``8 * n_per_combo`` preference data covering the full matrix."""
    rows: list[dict[str, Any]] = []
    for i in range(n_per_combo):
        rows.extend(_combos(i))
    if tools_as_json_string:
        for r in rows:
            if "tools" in r:
                r["tools"] = json.dumps(r["tools"])
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    """Generate the train (+ optional validation) JSONL test set."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output dir (writes train.jsonl [+ validation.jsonl])")
    ap.add_argument("--n-per-combo", type=int, default=3, help="Samples per combo (×8 combos)")
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.25,
        help="Fraction per combo held out as validation.jsonl (0 = train only; the probe recipe needs a val split)",
    )
    ap.add_argument(
        "--tools-as-json-string",
        default="true",
        choices=["true", "false"],
        help="Emit datum['tools'] as a JSON string (Arrow-safe; default) vs. a raw list.",
    )
    args = ap.parse_args()

    rows = build_rows(args.n_per_combo, args.tools_as_json_string == "true")
    os.makedirs(args.out, exist_ok=True)

    if args.val_fraction > 0.0:
        # Stratified split: hold out the last `n_val_per_combo` samples of EACH combo, so both
        # splits cover the full thinking × tools × multi/single-turn matrix (a flat stride would
        # alias with the 8-combo period and leave most combos out of val).
        n_val_per_combo = max(1, round(args.n_per_combo * args.val_fraction))
        by_combo: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_combo.setdefault(r["combo"], []).append(r)
        val_ids = {id(r) for items in by_combo.values() for r in items[-n_val_per_combo:]}
        train = [r for r in rows if id(r) not in val_ids]
        val = [r for r in rows if id(r) in val_ids]
        _write_jsonl(train, os.path.join(args.out, "train.jsonl"))
        _write_jsonl(val, os.path.join(args.out, "validation.jsonl"))
        print(f"wrote {len(train)} train + {len(val)} validation rows -> {args.out}")
    else:
        _write_jsonl(rows, os.path.join(args.out, "train.jsonl"))
        print(f"wrote {len(rows)} train rows -> {args.out}/train.jsonl")


if __name__ == "__main__":
    main()
