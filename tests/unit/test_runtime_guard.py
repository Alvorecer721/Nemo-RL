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

The guard keys off ``is_apertus_model`` in ``nemo_rl.models.huggingface.common`` (present only in our
nemo_rl) and a static check that the Bridge's ``apertus_bridge.py`` defines the refit-emit override.
Cases drive the guard's signals explicitly — the marker via monkeypatch, the Bridge via a temp
apertus_bridge.py behind the path seam — rather than depending on which nemo_rl or Bridge is loaded.
"""

from pathlib import Path

import pytest

from nemo_rl_apertus import runtime_guard
from nemo_rl_apertus.runtime_guard import assert_apertus_runtime

BRIDGE_WITH_REFIT_EMIT = """
class ApertusBridge:
    def maybe_modify_converted_hf_weight(self):
        pass
"""

BRIDGE_WITHOUT_REFIT_EMIT = """
class ApertusBridge:
    pass
"""


@pytest.fixture
def marker_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the nemo_rl-side marker look like a healthy Apertus checkout."""
    from nemo_rl.models.huggingface import common

    monkeypatch.setattr(common, "is_apertus_model", lambda name: False, raising=False)


def _point_guard_at_bridge_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    """Route the guard's Bridge-path seam at a temp apertus_bridge.py with the given source."""
    bridge_module = tmp_path / "apertus_bridge.py"
    bridge_module.write_text(source)
    monkeypatch.setattr(runtime_guard, "_bridge_apertus_module_path", lambda: bridge_module)


def test_guard_passes_when_deltas_present(marker_present, monkeypatch, tmp_path):
    """Marker present and the Bridge source defines the refit-emit (a healthy checkout) -> no-op."""
    _point_guard_at_bridge_source(monkeypatch, tmp_path, BRIDGE_WITH_REFIT_EMIT)
    assert_apertus_runtime()  # must not raise


def test_guard_raises_when_marker_absent(monkeypatch):
    """is_apertus_model absent (the stock /opt copy) -> loud RuntimeError."""
    from nemo_rl.models.huggingface import common

    monkeypatch.delattr(common, "is_apertus_model", raising=False)
    with pytest.raises(RuntimeError, match="stock copy"):
        assert_apertus_runtime()


def test_guard_raises_when_bridge_lacks_refit_emit(marker_present, monkeypatch, tmp_path):
    """Marker present but a stale Bridge without the refit-emit override -> loud RuntimeError."""
    _point_guard_at_bridge_source(monkeypatch, tmp_path, BRIDGE_WITHOUT_REFIT_EMIT)
    with pytest.raises(RuntimeError, match="refit-emit"):
        assert_apertus_runtime()


def test_guard_raises_when_bridge_module_missing(marker_present, monkeypatch, tmp_path):
    """Resolved Bridge has no apertus_bridge.py at all (wholly stock Bridge) -> loud RuntimeError."""
    monkeypatch.setattr(
        runtime_guard, "_bridge_apertus_module_path", lambda: tmp_path / "apertus_bridge.py"
    )
    with pytest.raises(RuntimeError, match="no apertus_bridge module"):
        assert_apertus_runtime()
