# Copyright (c) 2026, the Apertus project.
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
"""Unit tests for the extensible preference-data processor interface.

The module is import-light (lazy nemo_rl imports), so the abstract base, the
pretokenized path (overrides build_message_log -> no tokenizer), and the
text-format parse logic are all testable without the full runtime. The text
__call__ path (default tokenizing build) and register() need the real
nemo_rl.data stack and are skipped where it isn't importable.
"""

import pytest
import torch

from nemo_rl_apertus.data_processors import (
    PretokenizedPreferenceProcessor,
    PreferenceDataProcessor,
    RankedCompletionsPreferenceProcessor,
    ToolThinkingPreferenceProcessor,
    render_preference_pair_with_template,
)


# ---------------------------------------------------------------------------
# ToolThinkingPreferenceProcessor — fake chat-template tokenizer
# ---------------------------------------------------------------------------
def _content_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):  # Apertus structured {blocks:[{type:...}]}
        return "|".join(b.get("type", "") for b in content.get("blocks", []))
    return ""


class _CTTok:
    """Fake tokenizer mirroring the Apertus chat template + tokenizer.

    apply_chat_template emits the BOS itself ('<s>') like the real Apertus template, reflects
    tools/enable_thinking in a [T?D?] developer tag (a list of tools also renders their names, while
    a non-list truthy `tools` renders ':RAW' — so a missing JSON-string decode is caught), and
    renders each turn append-only (assistant opener 'A:' == the generation prompt, so the prompt
    stays a string-prefix of the full render).
    __call__ is a char-level encoder: a leading '<s>' maps to the BOS id (1), and
    add_special_tokens=True would prepend ANOTHER BOS (so a double-BOS regression is caught); it
    never adds an EOS — like the Apertus tokenizer (add_eos_token=False).
    """

    bos_token_id = 1
    add_bos_token = True

    def apply_chat_template(
        self, messages, tools=None, enable_thinking=None, tokenize=False, add_generation_prompt=False
    ):
        names = (
            ",".join(t.get("function", {}).get("name", "?") for t in tools)
            if isinstance(tools, list)
            else ("RAW" if tools else "")
        )
        tag = f"[T{int(bool(tools))}D{int(bool(enable_thinking))}{(':' + names) if names else ''}]"
        out = ["<s>", tag]
        for m in messages:
            text = _content_text(m)
            if m["role"] == "assistant":
                out.append("A:" + text + ":endA")
            else:
                out.append(m["role"][:1].upper() + ":" + text + ":end")
        s = "".join(out)
        if add_generation_prompt:
            s += "A:"
        return s

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = []
        if add_special_tokens:
            ids.append(1)  # an extra BOS on top of the template's -> double-BOS if code uses True
        if text.startswith("<s>"):
            ids.append(1)  # the template's own BOS, recognized as the special id
            text = text[len("<s>"):]
        ids += [ord(c) for c in text]
        return {"input_ids": torch.tensor([ids])}


def _pref_datum(chosen, rejected, tools=None, **extra):
    datum = {
        "context": [{"role": "user", "content": "q"}],
        "completions": [
            {"rank": 0, "completion": [{"role": "assistant", "content": chosen}]},
            {"rank": 1, "completion": [{"role": "assistant", "content": rejected}]},
        ],
    }
    if tools is not None:
        datum["tools"] = tools
    datum.update(extra)
    return datum


def test_tool_thinking_basic_split_no_tools_no_thinking():
    out = ToolThinkingPreferenceProcessor()(_pref_datum("good", "bad"), None, _CTTok(), 1000, 0)
    chosen = out["message_log_chosen"]
    assert [m["role"] for m in chosen] == ["context", "assistant"]
    assert "[T0D0]" in chosen[0]["content"]  # tools + Deliberation both off in the developer block
    assert chosen[-1]["content"] == "good:endA"  # response = final turn's generated span (no EOS)
    assert out["message_log_rejected"][-1]["content"] == "bad:endA"
    assert out["loss_multiplier"] == 1.0
    # prompt is an exact token-prefix of the full render (BOS once, no duplicate/dropped tokens)
    assert chosen[0]["token_ids"][0].item() == 1  # BOS
    assert chosen[1]["token_ids"][0].item() != 1  # response has no second BOS


def test_tool_thinking_single_bos_no_duplicate():
    # Regression guard: the template emits the BOS, so the code must tokenize with
    # add_special_tokens=False — otherwise every sample would start <s><s> (double BOS).
    chosen = ToolThinkingPreferenceProcessor()(
        _pref_datum("good", "bad"), None, _CTTok(), 1000, 0
    )["message_log_chosen"]
    n_bos = int((chosen[0]["token_ids"] == 1).sum()) + int((chosen[1]["token_ids"] == 1).sum())
    assert n_bos == 1  # exactly one BOS across prompt + response
    assert chosen[0]["token_ids"][0].item() == 1  # at the very start


def test_tool_thinking_inline_marker_not_autodetected():
    # Inline <|inner_prefix|> strings are NOT auto-detected (correct-data assumption): without an
    # explicit flag or a structured thoughts block, thinking stays disabled.
    out = ToolThinkingPreferenceProcessor()(
        _pref_datum("<|inner_prefix|>think<|inner_suffix|>ans", "bad"), None, _CTTok(), 1000, 0
    )
    assert "D0]" in out["message_log_chosen"][0]["content"]  # Deliberation disabled (not auto-enabled)


def test_tool_thinking_tools_enable_developer_block():
    out = ToolThinkingPreferenceProcessor()(
        _pref_datum("ok", "no", tools=[{"type": "function", "function": {"name": "f"}}]),
        None, _CTTok(), 1000, 0,
    )
    assert "[T1D0:f]" in out["message_log_chosen"][0]["content"]  # Tool Capabilities enabled


def test_tool_thinking_tools_as_json_string_decoded():
    # Tool schemas are commonly stored per-row as a JSON string (Arrow can't unify arbitrary
    # `parameters` schemas), so the processor must json-decode them to a list — not pass the raw
    # string to the template (which would render ':RAW' here and iterate chars in the real template).
    import json

    tools = [{"type": "function", "function": {"name": "f"}}]
    out = ToolThinkingPreferenceProcessor()(
        _pref_datum("ok", "no", tools=json.dumps(tools)), None, _CTTok(), 1000, 0
    )
    assert "[T1D0:f]" in out["message_log_chosen"][0]["content"]  # decoded list, name rendered


def test_tool_thinking_empty_json_string_tools_is_none():
    # "[]" decodes to [] -> falsy -> no tools (an undecoded "[]" string would be truthy -> T1).
    out = ToolThinkingPreferenceProcessor()(
        _pref_datum("ok", "no", tools="[]"), None, _CTTok(), 1000, 0
    )
    assert "[T0D0]" in out["message_log_chosen"][0]["content"]


def test_tool_thinking_autodetect_thoughts_block():
    datum = _pref_datum({"blocks": [{"type": "thoughts"}, {"type": "response"}]}, "bad")
    out = ToolThinkingPreferenceProcessor()(datum, None, _CTTok(), 1000, 0)
    assert "D1]" in out["message_log_chosen"][0]["content"]  # Deliberation auto-enabled


def test_tool_thinking_explicit_flag_overrides():
    on = ToolThinkingPreferenceProcessor()(_pref_datum("a", "b", enable_thinking=True), None, _CTTok(), 1000, 0)
    off = ToolThinkingPreferenceProcessor()(_pref_datum("a", "b", enable_thinking=False), None, _CTTok(), 1000, 0)
    assert "D1]" in on["message_log_chosen"][0]["content"]
    assert "D0]" in off["message_log_chosen"][0]["content"]


def test_tool_thinking_tool_use_without_schemas_raises():
    datum = {
        "context": [{"role": "user", "content": "q"}],
        "completions": [
            {"rank": 0, "completion": [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}]},
            {"rank": 1, "completion": [{"role": "assistant", "content": "no"}]},
        ],
    }
    with pytest.raises(ValueError, match="tool calls but"):
        ToolThinkingPreferenceProcessor()(datum, None, _CTTok(), 1000, 0)


def test_tool_thinking_overlength_masked():
    out = ToolThinkingPreferenceProcessor()(_pref_datum("a", "b"), None, _CTTok(), max_seq_length=2, idx=0)
    assert out["loss_multiplier"] == 0.0


def test_render_split_raises_on_non_prefix():
    class _BadTok(_CTTok):
        # full render does NOT start with the prompt render -> must raise
        def apply_chat_template(self, messages, **kw):
            return "PROMPTX" if kw.get("add_generation_prompt") else "DIFFERENT"

    with pytest.raises(ValueError, match="token-prefix"):
        render_preference_pair_with_template(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], _BadTok()
        )


def _tok_turn(role, ids):
    return {"role": role, "token_ids": ids}


# ---------------------------------------------------------------------------
# abstract base
# ---------------------------------------------------------------------------
def test_base_is_abstract():
    with pytest.raises(TypeError):
        PreferenceDataProcessor()  # parse() is abstract


# ---------------------------------------------------------------------------
# PretokenizedPreferenceProcessor (skip tokenization)
# ---------------------------------------------------------------------------
def test_pretokenized_call_builds_spec():
    datum = {
        "chosen": [_tok_turn("user", [1, 2]), _tok_turn("assistant", [3, 4, 5])],
        "rejected": [_tok_turn("user", [1, 2]), _tok_turn("assistant", [6])],
    }
    out = PretokenizedPreferenceProcessor()(
        datum, task_data_spec=None, tokenizer=None, max_seq_length=100, idx=7
    )
    assert out["idx"] == 7
    assert out["loss_multiplier"] == 1.0
    assert out["length_chosen"] == 5 and out["length_rejected"] == 3
    assert [m["role"] for m in out["message_log_chosen"]] == ["user", "assistant"]
    last = out["message_log_chosen"][-1]
    assert isinstance(last["token_ids"], torch.Tensor)
    assert last["token_ids"].dtype == torch.long
    assert last["token_ids"].tolist() == [3, 4, 5]
    assert last["content"] == ""  # default when not supplied


def test_pretokenized_overlength_is_masked():
    datum = {
        "chosen": [_tok_turn("user", [1, 2]), _tok_turn("assistant", [3, 4, 5])],  # len 5
        "rejected": [_tok_turn("user", [1, 2]), _tok_turn("assistant", [6])],  # len 3
    }
    out = PretokenizedPreferenceProcessor()(datum, None, None, max_seq_length=2, idx=0)
    assert out["loss_multiplier"] == 0.0  # over max_seq_length -> masked
    assert out["length_chosen"] <= 2 and out["length_rejected"] <= 2


def test_pretokenized_no_max_seq_length():
    datum = {
        "chosen": [_tok_turn("assistant", [1, 2, 3])],
        "rejected": [_tok_turn("assistant", [4])],
    }
    out = PretokenizedPreferenceProcessor()(datum, None, None, max_seq_length=None, idx=0)
    assert out["loss_multiplier"] == 1.0  # None disables the length gate


def test_pretokenized_custom_keys():
    datum = {
        "win": [{"r": "user", "ids": [1]}, {"r": "assistant", "ids": [2, 3]}],
        "lose": [{"r": "user", "ids": [1]}, {"r": "assistant", "ids": [4]}],
    }
    proc = PretokenizedPreferenceProcessor(
        chosen_key="win", rejected_key="lose", role_key="r", token_ids_key="ids"
    )
    out = proc(datum, None, None, 100, 0)
    assert out["message_log_chosen"][-1]["token_ids"].tolist() == [2, 3]
    assert out["message_log_rejected"][-1]["token_ids"].tolist() == [4]


# ---------------------------------------------------------------------------
# RankedCompletionsPreferenceProcessor.parse (text format)
# ---------------------------------------------------------------------------
def test_ranked_parse_lower_rank_is_chosen():
    proc = RankedCompletionsPreferenceProcessor()
    datum = {
        "context": [{"role": "user", "content": "q"}],
        "completions": [
            {"rank": 1, "completion": [{"role": "assistant", "content": "bad"}]},
            {"rank": 0, "completion": [{"role": "assistant", "content": "good"}]},
        ],
    }
    chosen, rejected = proc.parse(datum)
    assert chosen[0]["content"] == "q"  # context prepended to both
    assert chosen[-1]["content"] == "good"  # lower rank -> chosen
    assert rejected[-1]["content"] == "bad"


def test_ranked_parse_tie_and_bad_count():
    proc = RankedCompletionsPreferenceProcessor()
    with pytest.raises(NotImplementedError):
        proc.parse({"context": [], "completions": [{"rank": 0, "completion": []}, {"rank": 0, "completion": []}]})
    with pytest.raises(ValueError):
        proc.parse({"context": [], "completions": [{"rank": 0, "completion": []}]})


# ---------------------------------------------------------------------------
# registry wiring (needs the real nemo_rl.data stack)
# ---------------------------------------------------------------------------
def test_register_into_registry():
    try:
        from nemo_rl.data.processors import PROCESSOR_REGISTRY
    except Exception:  # noqa: BLE001 - decord/full runtime absent off-cluster
        pytest.skip("nemo_rl.data.processors not importable in this environment")
    proc = PretokenizedPreferenceProcessor()
    name = "test_pretok_pref_apertus"
    PROCESSOR_REGISTRY.pop(name, None)
    try:
        returned = proc.register(name)
        assert returned is proc
        assert PROCESSOR_REGISTRY[name] is proc  # callable instance is the processor
    finally:
        PROCESSOR_REGISTRY.pop(name, None)
