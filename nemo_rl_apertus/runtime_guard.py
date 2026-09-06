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

"""Fail fast when an Apertus run resolves an incompatible local runtime.

The fork keeps xIELU ``beta``/``eps`` as engine-owned architecture constants:
vLLM excludes them from weight state and the Bridge must not synthesize refit
keys for them.  A stale NeMo-RL or Bridge checkout can silently violate that
contract and corrupt online generation, so every Apertus entrypoint checks the
contract before allocating model state.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def _bundled_bridge_apertus_module_path() -> Path:
    """Return the Bridge source shipped beside NeMo-RL in source and release trees."""
    return (
        Path(__file__).resolve().parents[1]
        / "3rdparty"
        / "Megatron-Bridge-workspace"
        / "Megatron-Bridge"
        / "src"
        / "megatron"
        / "bridge"
        / "models"
        / "apertus"
        / "apertus_bridge.py"
    )


def _bridge_apertus_module_path() -> Path:
    """Locate the active or bundled Apertus Bridge source without importing it.

    ``find_spec`` follows the same ``sys.path`` resolution as a real import but only imports the
    code-free ``megatron`` namespace parent — executing ``megatron/bridge/__init__.py`` would pull
    the full model zoo (+~17s warm / +~110s cold in the guard-only launcher process). Release
    images intentionally install Bridge only in the frozen Megatron worker environment, so the
    launcher interpreter falls back to the same bundled source tree used to build that worker.
    """
    try:
        spec = importlib.util.find_spec("megatron.bridge")
    except ModuleNotFoundError:
        spec = None
    if spec is not None and spec.submodule_search_locations:
        return (
            Path(next(iter(spec.submodule_search_locations)))
            / "models"
            / "apertus"
            / "apertus_bridge.py"
        )

    bundled_module = _bundled_bridge_apertus_module_path()
    if bundled_module.is_file():
        return bundled_module
    raise RuntimeError(
        "Apertus runtime guard failed: no Megatron-Bridge source is available.\n"
        f"  Expected bundled source: {bundled_module}\n"
        "  Fix: initialize the Bridge submodule — git submodule update --init --recursive — "
        "or ensure PYTHONPATH includes <repo>/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src."
    )


def assert_apertus_runtime() -> None:
    """Raise if the Apertus deltas are missing from the imported runtime.

    The NeMo-RL marker distinguishes this checkout from the image's stock copy.
    The Bridge source must declare that the engine owns xIELU static state and
    must not retain the legacy synthetic-weight override.  The Bridge check is
    static so the guard remains cheap in launcher preflight processes.
    """
    import nemo_rl
    from nemo_rl.models.huggingface import common

    if not hasattr(common, "is_apertus_model"):
        raise RuntimeError(
            "Apertus runtime guard failed: the imported nemo_rl is the stock copy, not the Apertus checkout.\n"
            f"  nemo_rl loaded from: {nemo_rl.__file__}\n"
            "  It is missing the Apertus engine-owned xIELU state contract.\n"
            "  Online training could silently corrupt generation after refit.\n"
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
    engine_owns_static_state = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name)
                and target.id == "APERTUS_XIELU_STATIC_STATE_OWNER"
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value == "engine"
        for node in tree.body
    )
    defines_legacy_refit_emit = any(
        isinstance(node, ast.ClassDef)
        and node.name == "ApertusBridge"
        and any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "maybe_modify_converted_hf_weight"
            for item in node.body
        )
        for node in ast.walk(tree)
    )
    if not engine_owns_static_state:
        raise RuntimeError(
            "Apertus runtime guard failed: the Megatron-Bridge submodule does not declare engine-owned xIELU static state.\n"
            f"  ApertusBridge resolved from: {bridge_module}\n"
            '  Expected: APERTUS_XIELU_STATIC_STATE_OWNER = "engine".\n'
            "  Fix: update the submodule — git submodule update --init --recursive."
        )
    if defines_legacy_refit_emit:
        raise RuntimeError(
            "Apertus runtime guard failed: ApertusBridge still defines the legacy xIELU synthetic-weight refit override.\n"
            f"  ApertusBridge resolved from: {bridge_module}\n"
            "  Engine-owned beta/eps constants must not also travel through the weight stream.\n"
            "  Fix: update the submodule — git submodule update --init --recursive."
        )
