# Copyright (c) 2026, the Apertus project.
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

"""Tests for the Apertus runtime guard (nemo_rl_apertus/runtime_guard.py).

The guard keys off ``is_apertus_model`` in ``nemo_rl.models.huggingface.common`` (present only in our nemo_rl)
and the refit-emit override on the Bridge's ``ApertusBridge``.
All cases drive both signals explicitly via monkeypatch rather than depending on which nemo_rl or Bridge is loaded.
"""

import sys
import types

import pytest

from nemo_rl_apertus.runtime_guard import assert_apertus_runtime


def _stub_bridge_module(monkeypatch, with_refit_emit: bool):
    """Install a stub apertus_bridge module whose ApertusBridge optionally defines the refit-emit."""
    ns = {"maybe_modify_converted_hf_weight": lambda self, *a, **k: None} if with_refit_emit else {}
    bridge_cls = type("ApertusBridge", (), ns)
    module = types.ModuleType("megatron.bridge.models.apertus.apertus_bridge")
    module.ApertusBridge = bridge_cls
    monkeypatch.setitem(sys.modules, "megatron.bridge.models.apertus.apertus_bridge", module)


def test_guard_passes_when_deltas_present(monkeypatch):
    """Marker present and Bridge defines the refit-emit (a healthy checkout) -> no-op."""
    from nemo_rl.models.huggingface import common

    monkeypatch.setattr(common, "is_apertus_model", lambda name: False, raising=False)
    _stub_bridge_module(monkeypatch, with_refit_emit=True)
    assert_apertus_runtime()  # must not raise


def test_guard_raises_when_marker_absent(monkeypatch):
    """is_apertus_model absent (the stock /opt copy) -> loud RuntimeError."""
    from nemo_rl.models.huggingface import common

    monkeypatch.delattr(common, "is_apertus_model", raising=False)
    with pytest.raises(RuntimeError, match="stock copy"):
        assert_apertus_runtime()


def test_guard_raises_when_bridge_lacks_refit_emit(monkeypatch):
    """Marker present but a stale Bridge without the refit-emit override -> loud RuntimeError."""
    from nemo_rl.models.huggingface import common

    monkeypatch.setattr(common, "is_apertus_model", lambda name: False, raising=False)
    _stub_bridge_module(monkeypatch, with_refit_emit=False)
    with pytest.raises(RuntimeError, match="refit-emit"):
        assert_apertus_runtime()
