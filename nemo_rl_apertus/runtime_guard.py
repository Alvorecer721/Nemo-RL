# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Fail-fast guard: Apertus runs must import *our* nemo_rl, not the stock copy.

The Apertus deltas live in this repo's working tree and its Bridge submodule, not the container's baked ``/opt/nemo-rl`` copy:

* the xIELU dummy-load fix (``nemo_rl/models/huggingface/common.py``), and
* the Bridge refit-emit for the xIELU beta/eps buffers (``ApertusBridge.maybe_modify_converted_hf_weight`` in ``apertus_bridge.py``).

If a launcher runs from a directory / ``PYTHONPATH`` that does not point at this checkout, ``import nemo_rl`` silently falls back to stock ``/opt/nemo-rl`` and those deltas are absent.
The dangerous case is online training (GRPO / online-DPO): without the xIELU fix the refit omits the xIELU beta/eps buffers, vLLM's dummy-load leaves them as noise, and the run *silently* regresses to Generation KL Error ~0.79 — no error raised.

This converts that silent misconfiguration into a loud startup failure.
The check is cheap and side-effect-free, so it runs on every Apertus entrypoint regardless of online/offline — the invariant "we run our nemo_rl" is universal even though the xIELU symptom is online-only.
"""

from __future__ import annotations


def assert_apertus_runtime() -> None:
    """Raise if the Apertus deltas are missing from the imported runtime.

    Checks (1) ``is_apertus_model`` exists in our nemo_rl (absent in the stock ``/opt/nemo-rl``), and (2) the forked Megatron-Bridge defines the xIELU beta/eps refit-emit override — the base class ships a no-op, so we check it is on ``ApertusBridge`` itself (``vars``), not merely inherited. (2) matters because vLLM dummy-load relies on the refit carrying beta/eps; a stale Bridge submodule would silently regress KL.
    """
    import nemo_rl
    from nemo_rl.models.huggingface import common

    if not hasattr(common, "is_apertus_model"):
        raise RuntimeError(
            "Apertus runtime guard failed: the imported nemo_rl is the stock copy, not the Apertus checkout.\n"
            f"  nemo_rl loaded from: {nemo_rl.__file__}\n"
            "  It is missing the Apertus deltas (xIELU dummy-load + Bridge refit-emit).\n"
            "  Online training would silently regress to Generation KL Error ~0.79 with no error raised.\n"
            "  Fix: run from your Nemo-RL checkout, or set PYTHONPATH=<repo> so `import nemo_rl` resolves to it."
        )

    from megatron.bridge.models.apertus.apertus_bridge import ApertusBridge

    if "maybe_modify_converted_hf_weight" not in vars(ApertusBridge):
        raise RuntimeError(
            "Apertus runtime guard failed: the Megatron-Bridge submodule is missing the xIELU beta/eps refit-emit (ApertusBridge.maybe_modify_converted_hf_weight).\n"
            f"  ApertusBridge loaded from: {ApertusBridge.__module__}\n"
            "  With vLLM dummy-load the refit would not carry beta/eps and Generation KL would silently regress to ~0.79.\n"
            "  Fix: update the submodule — git submodule update --init --recursive."
        )
