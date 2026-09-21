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

"""Preserve TP reductions across optimized GLM/DeepSeek pipeline boundaries."""

import sys
from importlib.metadata import version

_MODULE = "vllm.models.deepseek_v32.nvidia.model"
_SOURCE = "models/deepseek_v32/nvidia/model.py"
_MARKER = "\n_NRL_PIPELINE_HIDDEN_STATES_PATCH = 1\n"
_SOURCE_EDITS = (
    (
        "from vllm.distributed import get_pp_group\n",
        "from vllm.distributed import get_pp_group, tensor_model_parallel_all_reduce\n",
    ),
    (
        '            prefix=f"{prefix}.layers",\n'
        "        )\n\n"
        "        if get_pp_group().is_last_rank:\n",
        '            prefix=f"{prefix}.layers",\n'
        "        )\n"
        "        if self.start_layer == self.end_layer:\n"
        "            raise ValueError(\n"
        '                "Optimized GLM/DeepSeek pipeline stages must each contain "\n'
        '                "at least one decoder layer. Check VLLM_PP_LAYER_PARTITION "\n'
        '                "and pipeline_parallel_size."\n'
        "            )\n\n"
        "        if get_pp_group().is_last_rank:\n",
    ),
    (
        "        attn_in: torch.Tensor | None = None,\n"
        "    ) -> tuple[torch.Tensor, torch.Tensor]:\n",
        "        attn_in: torch.Tensor | None = None,\n"
        "        input_is_reduced: bool = False,\n"
        "    ) -> tuple[torch.Tensor, torch.Tensor]:\n",
    ),
    (
        "        elif self.use_sequence_parallel:\n"
        "            hidden_states, residual = self.input_layernorm(hidden_states, residual)\n",
        "        elif self.use_sequence_parallel or input_is_reduced:\n"
        "            hidden_states, residual = self.input_layernorm(hidden_states, residual)\n",
    ),
    (
        "            hidden_states, residual = layer(positions, hidden_states, residual, attn_in)\n",
        "            hidden_states, residual = layer(\n"
        "                positions, hidden_states, residual, attn_in,\n"
        "                input_is_reduced=(\n"
        "                    idx == self.start_layer and not get_pp_group().is_first_rank\n"
        "                ),\n"
        "            )\n",
    ),
    (
        "            return IntermediateTensors(\n"
        '                {"hidden_states": hidden_states, "residual": residual}\n'
        "            )\n",
        "            # PP's TP slice/all-gather requires replicated activations. Move\n"
        "            # the next layer's input reduction before the PP boundary; its\n"
        "            # input norm must then consume this sum without reducing again.\n"
        "            hidden_states = tensor_model_parallel_all_reduce(hidden_states)\n"
        "            return IntermediateTensors(\n"
        '                {"hidden_states": hidden_states, "residual": residual}\n'
        "            )\n",
    ),
)


def patch_pipeline_hidden_states() -> None:
    """Install the boundary fix before model import; reject unsupported sources."""
    # Break the cycle: patches.py exposes this installer with the other patches.
    from nemo_rl.models.generation.vllm.patches import (
        _get_vllm_file,
        _locked_file_patch,
    )

    installed_version = version("vllm")
    if installed_version.split("+", 1)[0] != "0.29.0":
        raise RuntimeError(
            "Pipeline hidden-state compatibility patch requires vLLM 0.29.0; "
            f"found {installed_version}. Check upstream support before updating it."
        )
    module = sys.modules.get(_MODULE)
    if (
        module is not None
        and getattr(module, "_NRL_PIPELINE_HIDDEN_STATES_PATCH", None) != 1
    ):
        raise RuntimeError(
            f"{_MODULE} was already imported without the pipeline hidden-state fix. "
            "Install the compatibility patch before importing the model."
        )
    with _locked_file_patch(_get_vllm_file(_SOURCE)) as (source, write_back):
        updated = source
        for old, new in _SOURCE_EDITS:
            if updated.count(new) == 1:
                continue
            if updated.count(old) != 1:
                raise RuntimeError(
                    "Pipeline hidden-state patch anchor mismatch; "
                    "no source files have been changed."
                )
            updated = updated.replace(old, new, 1)
        if _MARKER not in updated:
            updated += _MARKER
        compile(updated, _SOURCE, "exec")
        if updated != source:
            write_back(updated)
