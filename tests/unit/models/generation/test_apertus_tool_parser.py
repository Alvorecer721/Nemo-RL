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
import json

import pytest

P = "<|tools_prefix|>"
S = "<|tools_suffix|>"


class _StubTok:
    """extract_tool_calls never tokenizes; the base just stores the tokenizer."""

    def __bool__(self):
        return True

    def get_vocab(self):
        return {}

    def get_added_vocab(self):
        return {}


@pytest.fixture
def parser():
    # Imported inside the fixture so the module collects without the vllm extra.
    from nemo_rl.models.generation.vllm.apertus_tool_parser import ApertusToolParser

    return ApertusToolParser(_StubTok())


@pytest.mark.vllm
def test_registers_under_apertus():
    import nemo_rl.models.generation.vllm.apertus_tool_parser  # noqa: F401  registers via decorator
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    assert "apertus" in ToolParserManager.list_registered()


@pytest.mark.vllm
@pytest.mark.parametrize(
    "output, expected",
    [
        (f'{P}[{{"get_weather": {{"city": "Paris"}}}}]{S}', [("get_weather", {"city": "Paris"})]),
        (f'{P}[{{"a": {{}}}}, {{"b": {{"x": 1}}}}]{S}', [("a", {}), ("b", {"x": 1})]),
        (f'{P}[{{"calc": {{"expr": "a}}b"}}}}]{S}', [("calc", {"expr": "a}b"})]),
        (f'{P}[{{"f": {{"x": 1}}}}]', [("f", {"x": 1})]),  # missing suffix (truncated)
        (
            f'{P}[{{"search": {{"q": "a", "filters": {{"k": [1, 2]}}}}}}]{S}',
            [("search", {"q": "a", "filters": {"k": [1, 2]}})],
        ),
    ],
)
def test_extract_success(parser, output, expected):
    r = parser.extract_tool_calls(output, request=None)
    assert r.tools_called
    got = [(c.function.name, json.loads(c.function.arguments)) for c in r.tool_calls]
    assert got == expected


@pytest.mark.vllm
def test_string_valued_args_passthrough(parser):
    r = parser.extract_tool_calls(f'{P}[{{"f": "raw string args"}}]{S}', request=None)
    assert r.tools_called
    assert r.tool_calls[0].function.arguments == "raw string args"


@pytest.mark.vllm
def test_content_before_call_is_preserved(parser):
    r = parser.extract_tool_calls(f'Let me check. {P}[{{"f": {{}}}}]{S}', request=None)
    assert r.tools_called
    assert r.content == "Let me check. "


@pytest.mark.vllm
def test_empty_object_yields_no_calls(parser):
    r = parser.extract_tool_calls(f"{P}[{{}}]{S}", request=None)
    assert not r.tools_called
    assert r.tool_calls == []


@pytest.mark.vllm
def test_plain_text_is_not_a_tool_call(parser):
    r = parser.extract_tool_calls("The answer is 4.", request=None)
    assert not r.tools_called
    assert r.content == "The answer is 4."


@pytest.mark.vllm
def test_malformed_json_degrades_gracefully(parser):
    r = parser.extract_tool_calls(f'{P}[{{"f": ]{S}', request=None)
    assert not r.tools_called
    assert r.tool_calls == []
    assert r.content is not None and P in r.content


@pytest.mark.vllm
def test_non_list_payload_degrades_gracefully(parser):
    r = parser.extract_tool_calls(f'{P}{{"f": {{}}}}{S}', request=None)
    assert not r.tools_called
    assert r.tool_calls == []


@pytest.mark.vllm
def test_adjust_request_forces_skip_special_tokens_with_tools(parser):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": "d",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        }
    ]
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice="auto",
        skip_special_tokens=True,
    )
    assert parser.adjust_request(req).skip_special_tokens is False


@pytest.mark.vllm
def test_adjust_request_leaves_skip_special_tokens_without_tools(parser):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        skip_special_tokens=True,
    )
    assert parser.adjust_request(req).skip_special_tokens is True


@pytest.mark.vllm
def test_streaming_emits_complete_tool_call_at_block_close(parser):
    full = f'{P}[{{"f": {{"x": 1}}}}]{S}'
    dm = parser.extract_tool_calls_streaming("", full, full, [], [], [], request=None)
    assert dm is not None and dm.tool_calls
    fn = dm.tool_calls[0].function
    assert fn.name == "f"
    assert json.loads(fn.arguments) == {"x": 1}


@pytest.mark.vllm
def test_streaming_passes_content_through(parser):
    dm = parser.extract_tool_calls_streaming("", "hello world", "hello world", [], [], [], request=None)
    assert dm is not None and dm.content == "hello world" and not dm.tool_calls
