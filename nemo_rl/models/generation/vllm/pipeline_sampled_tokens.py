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

"""Correct vLLM 0.29 pipeline feedback when execution changes CUDA streams."""

import sys
from importlib.metadata import version

_MODULE = "vllm.v1.worker.gpu.pp_utils"
_SOURCE = "v1/worker/gpu/pp_utils.py"
_MARKER = "\n_NRL_PIPELINE_SAMPLED_TOKENS_PATCH = 1\n"
_SOURCE_EDITS = (
    (
        "        self.device = device\n"
        "        self.main_stream = torch.cuda.current_stream(device)\n"
        "        self.broadcast_stream = torch.cuda.Stream(device)\n",
        "        self.device = device\n"
        "        self.broadcast_stream = torch.cuda.Stream(device)\n",
    ),
    (
        "        self.main_stream.wait_event(slot.event)\n",
        "        # Warmup and decoding can use different streams. Wait and retain\n"
        "        # received storage on the stream that actually consumes it.\n"
        "        consumer_stream = torch.cuda.current_stream(self.device)\n"
        "        consumer_stream.wait_event(slot.event)\n"
        "        for tensor in (\n"
        "            slot.sampled_tokens,\n"
        "            slot.num_sampled,\n"
        "            slot.num_rejected,\n"
        "            idx_mapping,\n"
        "        ):\n"
        "            tensor.record_stream(consumer_stream)\n",
    ),
    (
        "        num_reqs = input_batch.num_reqs\n"
        "        with torch.cuda.stream(self.broadcast_stream):\n"
        "            self.broadcast_stream.wait_stream(self.main_stream)\n",
        "        num_reqs = input_batch.num_reqs\n"
        "        main_stream = torch.cuda.current_stream(self.device)\n"
        "        with torch.cuda.stream(self.broadcast_stream):\n"
        "            self.broadcast_stream.wait_stream(main_stream)\n",
    ),
    (
        "            # Must record_stream since these were allocated on broadcast stream but\n"
        "            # later used on the main stream.\n"
        "            sampled_tokens.record_stream(self.main_stream)\n"
        "            combined.record_stream(self.main_stream)\n",
        "            # The consuming stream is registered when this slot is consumed.\n",
    ),
    (
        "        assert sampled_token_ids.dtype == torch.int64\n\n"
        "        if current_platform.is_xpu():\n"
        "            self.main_stream.synchronize()\n\n"
        "        with torch.cuda.stream(self.broadcast_stream):\n"
        "            self.broadcast_stream.wait_stream(self.main_stream)\n",
        "        assert sampled_token_ids.dtype == torch.int64\n"
        "        main_stream = torch.cuda.current_stream(self.device)\n\n"
        "        if current_platform.is_xpu():\n"
        "            main_stream.synchronize()\n\n"
        "        with torch.cuda.stream(self.broadcast_stream):\n"
        "            self.broadcast_stream.wait_stream(main_stream)\n",
    ),
)


def patch_pipeline_sampled_tokens() -> None:
    """Install the stream fix before PPHandler is imported; reject source drift."""
    # Break the cycle: patches.py exposes this installer with the other patches.
    from nemo_rl.models.generation.vllm.patches import (
        _get_vllm_file,
        _locked_file_patch,
    )

    installed_version = version("vllm")
    if installed_version.split("+", 1)[0] != "0.29.0":
        raise RuntimeError(
            "Pipeline sampled-token compatibility patch requires vLLM 0.29.0; "
            f"found {installed_version}. Check upstream support before updating it."
        )
    module = sys.modules.get(_MODULE)
    if (
        module is not None
        and getattr(module, "_NRL_PIPELINE_SAMPLED_TOKENS_PATCH", None) != 1
    ):
        raise RuntimeError(
            f"{_MODULE} was already imported without the pipeline sampled-token fix. "
            "Install the compatibility patch before importing the model runner."
        )
    with _locked_file_patch(_get_vllm_file(_SOURCE)) as (source, write_back):
        updated = source
        for old, new in _SOURCE_EDITS:
            if updated.count(new) == 1:
                continue
            if updated.count(old) != 1:
                raise RuntimeError(
                    "Pipeline sampled-token patch anchor mismatch; "
                    "no source files have been changed."
                )
            updated = updated.replace(old, new, 1)
        if _MARKER not in updated:
            updated += _MARKER
        compile(updated, _SOURCE, "exec")
        if updated != source:
            write_back(updated)
