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

The Apertus deltas live in this repo's working tree, not the container's baked ``/opt/nemo-rl`` copy:

* the xIELU dummy-load fix (``nemo_rl/models/huggingface/common.py`` + the Bridge refit-emit in ``apertus_bridge.py``), and
* the raw-Megatron-checkpoint loader (#2329, ``nemo_rl/models/megatron/setup.py``).

If a launcher runs from a directory / ``PYTHONPATH`` that does not point at this checkout, ``import nemo_rl`` silently falls back to stock ``/opt/nemo-rl`` and those deltas are absent.
The dangerous case is online training (GRPO / online-DPO): without the xIELU fix the refit omits the xIELU beta/eps buffers, vLLM's dummy-load leaves them as noise, and the run *silently* regresses to Generation KL Error ~0.79 — no error raised.

This converts that silent misconfiguration into a loud startup failure.
The check is cheap and side-effect-free, so it runs on every Apertus entrypoint regardless of online/offline — the invariant "we run our nemo_rl" is universal even though the xIELU symptom is online-only.
"""

from __future__ import annotations


def assert_apertus_runtime() -> None:
    """Raise if the imported ``nemo_rl`` is the stock copy, not the Apertus checkout.

    ``is_apertus_model`` exists only in our nemo_rl, so its absence reliably signals the stock ``/opt/nemo-rl`` was imported instead of this checkout.
    """
    import nemo_rl
    from nemo_rl.models.huggingface import common

    if not hasattr(common, "is_apertus_model"):
        raise RuntimeError(
            "Apertus runtime guard failed: the imported nemo_rl is the stock copy, not the Apertus checkout.\n"
            f"  nemo_rl loaded from: {nemo_rl.__file__}\n"
            "  It is missing the Apertus deltas (xIELU dummy-load + Bridge refit-emit + raw-Megatron checkpoint loader).\n"
            "  Online training would silently regress to Generation KL Error ~0.79 with no error raised.\n"
            "  Fix: run from your Nemo-RL checkout, or set PYTHONPATH=<repo> so `import nemo_rl` resolves to it."
        )
