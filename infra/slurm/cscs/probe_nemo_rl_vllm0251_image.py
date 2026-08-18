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
import time
import types
from pathlib import Path


def check_fp8_refit_storage() -> None:
    """Keep stable FP8 parameter storage across repeated refits."""
    import torch
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    from nemo_rl.models.generation.vllm.quantization import fp8

    layer = types.SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False),
        weight_scale_inv=torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False),
    )
    method = types.SimpleNamespace(
        block_quant=True,
        quant_config=types.SimpleNamespace(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
        ),
    )
    original_process = fp8_utils.process_fp8_weight_block_strategy
    original_post_process = fp8.maybe_post_process_fp8_weight_block
    try:
        fp8_utils.process_fp8_weight_block_strategy = lambda weight, scale: (
            torch.ones_like(weight),
            torch.ones_like(scale),
        )
        fp8.maybe_post_process_fp8_weight_block = lambda _layer: None

        weight_ptr = layer.weight.data.data_ptr()
        scale_ptr = layer.weight_scale_inv.data.data_ptr()
        weight_param = layer.weight
        scale_param = layer.weight_scale_inv
        for _ in range(3):
            fp8.process_weights_after_loading(method, layer)

        assert layer.weight.data.data_ptr() == weight_ptr
        assert layer.weight_scale_inv.data.data_ptr() == scale_ptr
        assert layer.weight is weight_param
        assert layer.weight_scale_inv is scale_param
        assert torch.equal(layer.weight.data, torch.ones(4, 4))
        assert torch.equal(layer.weight_scale_inv.data, torch.ones(1, 1))
    finally:
        fp8_utils.process_fp8_weight_block_strategy = original_process
        fp8.maybe_post_process_fp8_weight_block = original_post_process
    print("fp8_refit_storage=OK")


def main() -> None:
    import nemo_rl
    from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat

    nemo_rl_root = Path(nemo_rl.__file__).resolve()
    assert nemo_rl_root.is_relative_to("/opt/nemo-rl"), nemo_rl_root
    print(f"baked_nemo_rl={nemo_rl_root}")
    ensure_vllm_source_compat()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    assert vllm.__version__ == "0.25.1", vllm.__version__
    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    enforce_eager = os.environ.get("ENFORCE_EAGER", "true").lower() == "true"
    num_prompts = int(os.environ.get("PROBE_NUM_PROMPTS", "1"))
    max_tokens = int(os.environ.get("PROBE_MAX_TOKENS", "24"))
    assert tensor_parallel_size >= 1, tensor_parallel_size
    assert torch.cuda.device_count() >= tensor_parallel_size, (
        torch.cuda.device_count(),
        tensor_parallel_size,
    )
    for device_index in range(tensor_parallel_size):
        assert torch.cuda.get_device_name(device_index) == "NVIDIA GH200 120GB"
    print(f"vllm_version={vllm.__version__}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"tensor_parallel_size={tensor_parallel_size}")
    print(f"enforce_eager={enforce_eager}")
    check_fp8_refit_storage()

    llm = LLM(
        model=os.environ["APERTUS_CKPT"],
        tokenizer=os.environ["APERTUS_TOKENIZER"],
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.50,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="ray" if tensor_parallel_size > 1 else None,
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
    generation_start = time.perf_counter()
    outputs = llm.generate(
        [prompt] * num_prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
            skip_special_tokens=False,
        ),
    )
    generation_seconds = time.perf_counter() - generation_start
    total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    assert len(outputs) == num_prompts, outputs
    assert total_generated_tokens == num_prompts * max_tokens, outputs
    generated = outputs[0].outputs[0]
    assert len(generated.token_ids) == max_tokens, outputs
    assert generated.text.strip(), outputs
    print(f"generated_prompts={len(outputs)}")
    print(f"generated_tokens={total_generated_tokens}")
    print(f"generation_seconds={generation_seconds:.3f}")
    print(
        f"generation_tokens_per_second={total_generated_tokens / generation_seconds:.2f}"
    )
    print(f"generated_text={generated.text!r}")

    # Validate the plugin against vLLM 0.25.1 protocol objects in this runtime.
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    from nemo_rl.models.generation.vllm.apertus_tool_parser import ApertusToolParser

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
