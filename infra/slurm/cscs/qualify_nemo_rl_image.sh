#!/bin/bash
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

# Qualify an assembled image independently of build and export.
set -euo pipefail
: "${SQSH_PATH:?Set SQSH_PATH to the assembled image}"
NVTE_WITH_NCCL_EP=${NVTE_WITH_NCCL_EP:-0}
[[ "$NVTE_WITH_NCCL_EP" == 0 || "$NVTE_WITH_NCCL_EP" == 1 ]] || exit 2
[[ -s "$SQSH_PATH" ]] || { echo "Missing assembled image: $SQSH_PATH" >&2; exit 1; }
# Verify the dependency/API boundary that motivated this image. A writable
# overlay is required because NeMo-RL applies narrowly scoped vLLM source
# compatibility patches at worker startup.
enroot start --root --rw "$SQSH_PATH" \
    /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python - <<'PY'
from importlib.metadata import version

from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat

ensure_vllm_source_compat()

import openai
import vllm
import xgrammar
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tool_parsers import utils as tool_parser_utils

if vllm.__version__ != "0.26.0":
    raise RuntimeError(f"Expected vLLM 0.26.0, found {vllm.__version__}")
if any(
    symbol is None
    for symbol in (OnlineRenderer, ServingTokenization, tool_parser_utils.NamespaceTool)
):
    raise RuntimeError("Required vLLM 0.25 APIs are unavailable")
print("vLLM:", vllm.__version__)
print("OpenAI:", openai.__version__)
print("xgrammar:", version("xgrammar"))
print("vLLM 0.25 renderer, tokenization, and tool-parser imports: OK")
PY

# TE is compiled independently in the Megatron worker environment. Verify the
# exact MCore pairing and the TE APIs used by the dense and grouped-MoE paths.
enroot start --root --rw "$SQSH_PATH" \
    /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
    - "$NVTE_WITH_NCCL_EP" <<'PY'
import ctypes
import sys
from importlib.metadata import version

import megatron.core.extensions.transformer_engine as mcore_te

if not mcore_te.HAVE_TE:
    raise RuntimeError("MCore did not discover Transformer Engine")

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

expected_te = "2.18.0+27486e03"
actual_te = version("transformer-engine")
if actual_te != expected_te:
    raise RuntimeError(f"Expected Transformer Engine {expected_te}, found {actual_te}")

expect_nccl_ep = bool(int(sys.argv[1]))
ep_symbols = [
    name
    for name in ("ep_initialize", "ep_finalize", "ep_get_zero_copy")
    if hasattr(tex, name)
]
if expect_nccl_ep:
    if len(ep_symbols) != 3:
        raise RuntimeError(f"NCCL-EP was enabled but only found symbols: {ep_symbols}")
    nccl = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_GLOBAL)
    nccl_version = ctypes.c_int()
    if nccl.ncclGetVersion(ctypes.byref(nccl_version)) != 0:
        raise RuntimeError("ncclGetVersion failed")
    if nccl_version.value < 23004:
        raise RuntimeError(f"NCCL-EP requires NCCL >=2.30.4, found {nccl_version.value}")
elif ep_symbols:
    raise RuntimeError(f"NCCL-EP was expected to be disabled, found symbols: {ep_symbols}")

if any(
    symbol is None
    for symbol in (mcore_te.TELinear, te.DotProductAttention, te.GroupedLinear)
):
    raise RuntimeError("Required MCore/Transformer Engine APIs are unavailable")
print("Transformer Engine:", actual_te)
print("MCore TE adapters: OK; NCCL-EP enabled:", expect_nccl_ep)
PY

echo "Native image API qualification passed: $SQSH_PATH"
