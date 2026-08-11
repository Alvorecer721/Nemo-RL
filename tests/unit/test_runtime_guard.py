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

The guard keys off ``is_apertus_model`` in ``nemo_rl.models.huggingface.common`` (present only in our nemo_rl).
Both cases drive the marker explicitly via monkeypatch rather than depending on which nemo_rl is loaded.
"""

import pytest

from nemo_rl_apertus.runtime_guard import assert_apertus_runtime


def test_guard_passes_when_marker_present(monkeypatch):
    """is_apertus_model present (as in a real Apertus checkout) -> no-op."""
    from nemo_rl.models.huggingface import common

    monkeypatch.setattr(common, "is_apertus_model", lambda name: False, raising=False)
    assert_apertus_runtime()  # must not raise


def test_guard_raises_when_marker_absent(monkeypatch):
    """is_apertus_model absent (the stock /opt copy) -> loud RuntimeError."""
    from nemo_rl.models.huggingface import common

    monkeypatch.delattr(common, "is_apertus_model", raising=False)
    with pytest.raises(RuntimeError, match="stock copy"):
        assert_apertus_runtime()
