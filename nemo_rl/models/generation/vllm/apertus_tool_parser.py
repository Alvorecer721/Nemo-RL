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

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

logger = init_logger(__name__)


@ToolParserManager.register_module("apertus")
class ApertusToolParser(ToolParser):
    """Parser for Apertus tool calls: ``<|tools_prefix|>[{"<name>": <args>}, ...]<|tools_suffix|>``.

    The tool name is the dict KEY and the arguments are the VALUE — inverted
    relative to the hermes/mistral/llama families, so none of the built-in
    vLLM parsers match.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.start = "<|tools_prefix|>"
        self.end = "<|tools_suffix|>"
        self._stream_emitted = False
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser constructor."
            )

    @staticmethod
    def _calls(array_obj: list) -> list[ToolCall]:
        out: list[ToolCall] = []
        for el in array_obj:
            if not isinstance(el, dict) or not el:
                continue
            name, args = next(iter(el.items()))
            out.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=args
                        if isinstance(args, str)
                        else json.dumps(args, ensure_ascii=False),
                    ),
                )
            )
        return out

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        start = model_output.find(self.start)
        if start == -1:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        try:
            body_start = start + len(self.start)
            end = model_output.find(self.end, body_start)
            raw = (
                model_output[body_start:end] if end != -1 else model_output[body_start:]
            )
            array_obj = json.loads(raw.strip())
            if not isinstance(array_obj, list):
                raise ValueError("Apertus tool-call payload is not a JSON array")
            calls = self._calls(array_obj)
            content = model_output[:start]
            return ExtractedToolCallInformation(
                tools_called=bool(calls), tool_calls=calls, content=content or None
            )
        except (json.JSONDecodeError, ValueError):
            logger.exception("Error in extracting Apertus tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request: ChatCompletionRequest,
    ):
        if self.start not in current_text:
            return DeltaMessage(content=delta_text)
        if self.start not in previous_text:
            pre = current_text[: current_text.find(self.start)]
            if len(pre) > len(previous_text):
                return DeltaMessage(content=pre[len(previous_text) :])
        # Apertus serializes tool args as one inverted-JSON block; rather than diff
        # re-serialized partial JSON (non-monotonic), emit the whole call once the
        # block closes, reusing the tested non-streaming parse.
        if self.end not in current_text or self._stream_emitted:
            return None
        self._stream_emitted = True
        info = self.extract_tool_calls(current_text, request)
        if not info.tools_called:
            return None
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=i,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=c.function.name, arguments=c.function.arguments
                    ).model_dump(exclude_none=True),
                )
                for i, c in enumerate(info.tool_calls)
            ]
        )
