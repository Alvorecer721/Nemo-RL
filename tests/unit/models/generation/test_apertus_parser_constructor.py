# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the plugin constructor without importing a GPU inference runtime."""

import ast
from pathlib import Path


def test_parser_accepts_and_forwards_request_tools():
    source = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/vllm/apertus_tool_parser.py"
    )
    module = ast.parse(source.read_text())
    cls = next(
        n
        for n in module.body
        if isinstance(n, ast.ClassDef) and n.name == "ApertusToolParser"
    )
    cls.decorator_list = []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]

    class ToolParser:
        # vLLM 0.26 and 0.29 both expose this base constructor contract.
        def __init__(self, tokenizer, tools=None):
            self.model_tokenizer = tokenizer
            self.tools = tools

    namespace = dict(ToolParser=ToolParser, TokenizerLike=object, Tool=object)
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace
    )
    tokenizer = object()
    tools = [object()]
    parser = namespace["ApertusToolParser"](tokenizer, tools)
    assert parser.model_tokenizer is tokenizer
    assert parser.tools is tools
    assert namespace["ApertusToolParser"](tokenizer).model_tokenizer is tokenizer
