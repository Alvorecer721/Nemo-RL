#!/usr/bin/env python3
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

"""Load Apertus with baked vLLM, generate tokens, and validate its tool parser."""

import gc
import json
import os


def main() -> None:
    from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat

    ensure_vllm_source_compat()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    assert vllm.__version__ == "0.25.1", vllm.__version__
    assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
    print(f"vllm_version={vllm.__version__}")
    print(f"gpu={torch.cuda.get_device_name(0)}")

    llm = LLM(
        model=os.environ["APERTUS_CKPT"],
        tokenizer=os.environ["APERTUS_TOKENIZER"],
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ["APERTUS_TOKENIZER"])
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Give one short sentence explaining what reinforcement learning is.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=24,
            ignore_eos=True,
            skip_special_tokens=False,
        ),
    )
    generated = outputs[0].outputs[0]
    assert len(generated.token_ids) == 24, outputs
    assert generated.text.strip(), outputs
    print(f"generated_tokens={len(generated.token_ids)}")
    print(f"generated_text={generated.text!r}")

    # Validate the plugin against vLLM 0.25.1 protocol objects in this runtime.
    from nemo_rl.models.generation.vllm.apertus_tool_parser import ApertusToolParser
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    class StubTokenizer:
        def __bool__(self):
            return True

        def get_vocab(self):
            return {}

        def get_added_vocab(self):
            return {}

    assert "apertus" in ToolParserManager.list_registered()
    parser = ApertusToolParser(StubTokenizer())
    parsed = parser.extract_tool_calls(
        '<|tools_prefix|>[{"get_weather":{"city":"Zurich"}}]<|tools_suffix|>',
        request=None,
    )
    assert parsed.tools_called and len(parsed.tool_calls) == 1, parsed
    call = parsed.tool_calls[0]
    assert call.function.name == "get_weather", call
    assert json.loads(call.function.arguments) == {"city": "Zurich"}, call
    print("apertus_tool_parser=OK")

    del llm
    gc.collect()
    print("model_generation=OK")


if __name__ == "__main__":
    main()
