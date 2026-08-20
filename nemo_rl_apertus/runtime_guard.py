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

import ast
import importlib.util
from pathlib import Path


def _bridge_apertus_module_path() -> Path:
    """Locate the apertus_bridge.py that ``import megatron.bridge`` would resolve, without executing the package.

    ``find_spec`` follows the same ``sys.path`` resolution as a real import but only imports the
    code-free ``megatron`` namespace parent — executing ``megatron/bridge/__init__.py`` would pull
    the full model zoo (+~17s warm / +~110s cold in the guard-only launcher process).
    """
    spec = importlib.util.find_spec("megatron.bridge")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Apertus runtime guard failed: megatron.bridge is not importable.\n"
            "  Fix: initialize the Bridge submodule — git submodule update --init --recursive — "
            "and ensure PYTHONPATH includes <repo>/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src."
        )
    return (
        Path(next(iter(spec.submodule_search_locations)))
        / "models"
        / "apertus"
        / "apertus_bridge.py"
    )


def assert_apertus_runtime() -> None:
    """Raise if the Apertus deltas are missing from the imported runtime.

    Checks (1) ``is_apertus_model`` exists in our nemo_rl (absent in the stock ``/opt/nemo-rl``), and (2) the Bridge's ``apertus_bridge.py`` defines the xIELU beta/eps refit-emit override on ``ApertusBridge`` itself — the base class ships a no-op, so an inherited-only method means a stale submodule. (2) matters because vLLM dummy-load relies on the refit carrying beta/eps; a stale Bridge would silently regress KL. The Bridge check is static (spec resolution + AST) so the guard stays cheap in the launcher's guard-only process.
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

    bridge_module = _bridge_apertus_module_path()
    if not bridge_module.is_file():
        raise RuntimeError(
            "Apertus runtime guard failed: the resolved Megatron-Bridge has no apertus_bridge module (stock or stale Bridge).\n"
            f"  Expected: {bridge_module}\n"
            "  Fix: update the submodule — git submodule update --init --recursive."
        )

    tree = ast.parse(bridge_module.read_text())
    defines_refit_emit = any(
        isinstance(node, ast.ClassDef)
        and node.name == "ApertusBridge"
        and any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "maybe_modify_converted_hf_weight"
            for item in node.body
        )
        for node in ast.walk(tree)
    )
    if not defines_refit_emit:
        raise RuntimeError(
            "Apertus runtime guard failed: the Megatron-Bridge submodule is missing the xIELU beta/eps refit-emit (ApertusBridge.maybe_modify_converted_hf_weight).\n"
            f"  ApertusBridge resolved from: {bridge_module}\n"
            "  With vLLM dummy-load the refit would not carry beta/eps and Generation KL would silently regress to ~0.79.\n"
            "  Fix: update the submodule — git submodule update --init --recursive."
        )
