# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Qwen-style assistant outputs to Apertus assistant-body format.

Qwen3.5 emits:
  - Thinking blocks:  ``<think>...</think>``
  - Tool calls:       ``<tool_call>{"name": N, "arguments": A}</tool_call>``  (Hermes-style)
  - End of turn:      ``<|im_end|>``

Apertus emits:
  - Thinking blocks:  ``<|inner_prefix|>...<|inner_suffix|>``
  - Tool calls:       ``<|tools_prefix|>[{"N": A}, ...]<|tools_suffix|>``  (inverted JSON)
  - End of turn:      ``<|assistant_end|>``

This converter takes a Qwen assistant response body (what the model wrote between
``<|im_start|>assistant`` and ``<|im_end|>``) and returns a string in Apertus's
assistant-body format, ready to be used as the ``chosen`` text in a DPO
preference pair where the prompt was rendered with Apertus's chat template.

Why this matters for DPO: the loss computes logprobs over the *exact* tokenized
chosen string, so emitting ``<|im_end|>`` instead of ``<|assistant_end|>`` would
teach Apertus to use Qwen's terminator — a regression we'd then have to undo.
"""

import json
import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_QWEN_TERMINATOR_RE = re.compile(r"<\|im_end\|>.*", re.DOTALL)
# A contiguous run of Qwen tool-calls (possibly separated by whitespace) — we
# collapse it into a single Apertus tool block.
_TOOL_RUN_RE = re.compile(
    r"(?:<tool_call>.*?</tool_call>\s*)+", re.DOTALL
)


APERTUS_ASSISTANT_END = "<|assistant_end|>"
APERTUS_INNER_PREFIX = "<|inner_prefix|>"
APERTUS_INNER_SUFFIX = "<|inner_suffix|>"
APERTUS_TOOLS_PREFIX = "<|tools_prefix|>"
APERTUS_TOOLS_SUFFIX = "<|tools_suffix|>"


def _parse_tool_call(raw: str):
    """Parse a Hermes-style tool-call JSON object. Returns ``{name: args}`` (Apertus
    inverted format) on success, or ``None`` on parse failure."""
    try:
        obj = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", {})
    # arguments may be a JSON string or already a dict — normalize to a value.
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            # Keep the raw string if it isn't valid JSON; the Apertus envelope
            # supports either, and the verifier normalizes both sides.
            pass
    return {name: args}


def _convert_thinking(text: str) -> str:
    """Replace ``<think>X</think>`` blocks with Apertus's inner_prefix/suffix wrapping."""
    return _THINK_RE.sub(
        lambda m: f"{APERTUS_INNER_PREFIX}{m.group(1)}{APERTUS_INNER_SUFFIX}",
        text,
    )


def _convert_tool_calls(text: str) -> str:
    """Collapse runs of Hermes-style tool calls into a single Apertus tool block."""
    def _replace_run(match):
        run = match.group(0)
        calls = []
        for tc_raw in _TOOL_CALL_RE.findall(run):
            converted = _parse_tool_call(tc_raw)
            if converted is not None:
                calls.append(converted)
        if not calls:
            return run  # leave the original alone if we couldn't parse anything
        return (
            APERTUS_TOOLS_PREFIX
            + json.dumps(calls, ensure_ascii=False)
            + APERTUS_TOOLS_SUFFIX
        )

    return _TOOL_RUN_RE.sub(_replace_run, text)


def _strip_qwen_terminator(text: str) -> str:
    """Drop ``<|im_end|>`` and anything after it (Qwen models occasionally trail
    after the terminator; Apertus doesn't expect that)."""
    return _QWEN_TERMINATOR_RE.sub("", text)


def convert_qwen_body_to_apertus(qwen_body: str, append_terminator: bool = True) -> str:
    """Convert one Qwen assistant response body to Apertus assistant-body format.

    Parameters
    ----------
    qwen_body : str
        What Qwen generated between ``<|im_start|>assistant\\n`` and ``<|im_end|>``.
        Whitespace is preserved (we don't strip leading newlines that Qwen uses).
    append_terminator : bool, optional
        Append ``<|assistant_end|>`` at the end. Default True for DPO chosen
        strings; pass False if you want the body only.
    """
    text = _strip_qwen_terminator(qwen_body)
    text = _convert_thinking(text)
    text = _convert_tool_calls(text)
    text = text.rstrip()
    if append_terminator:
        text = text + APERTUS_ASSISTANT_END
    return text


# ---- Smoke test when invoked as a script (CI-friendly, no external deps). ----

def _smoke():
    cases = [
        # 1. Pure text — no thinking, no tool calls.
        ("Hello world<|im_end|>", "Hello world<|assistant_end|>"),
        # 2. Thinking block.
        (
            "<think>Let me think</think>The answer is 42<|im_end|>",
            "<|inner_prefix|>Let me think<|inner_suffix|>The answer is 42<|assistant_end|>",
        ),
        # 3. Tool call (Hermes → Apertus inverted).
        (
            '<tool_call>{"name": "search", "arguments": {"q": "apertus"}}</tool_call><|im_end|>',
            '<|tools_prefix|>[{"search": {"q": "apertus"}}]<|tools_suffix|><|assistant_end|>',
        ),
        # 4. Multiple tool calls collapsed.
        (
            '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call><|im_end|>',
            '<|tools_prefix|>[{"a": {"x": 1}}, {"b": {"y": 2}}]<|tools_suffix|><|assistant_end|>',
        ),
        # 5. Thinking + tool call.
        (
            '<think>which tool?</think><tool_call>{"name": "go", "arguments": {}}</tool_call><|im_end|>',
            '<|inner_prefix|>which tool?<|inner_suffix|><|tools_prefix|>[{"go": {}}]<|tools_suffix|><|assistant_end|>',
        ),
        # 6. Arguments-as-string from upstream (some tool emitters serialize args twice).
        (
            '<tool_call>{"name": "x", "arguments": "{\\"k\\": 1}"}</tool_call><|im_end|>',
            '<|tools_prefix|>[{"x": {"k": 1}}]<|tools_suffix|><|assistant_end|>',
        ),
    ]
    failures = 0
    for inp, want in cases:
        got = convert_qwen_body_to_apertus(inp)
        if got != want:
            failures += 1
            print(f"FAIL\n  in:   {inp!r}\n  want: {want!r}\n  got:  {got!r}")
        else:
            print(f"ok: {inp[:60]!r}")
    if failures:
        raise SystemExit(f"{failures} smoke failure(s)")
    print("All smoke cases passed.")


if __name__ == "__main__":
    _smoke()
