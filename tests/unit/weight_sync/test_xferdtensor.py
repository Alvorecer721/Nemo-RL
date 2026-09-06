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
"""Tests for the xferdtensor reshard path selection."""

import pytest
import torch
from torch.distributed._tensor import Replicate, Shard

from nemo_rl.weight_sync import xferdtensor as xfer
from nemo_rl.weight_sync import xferdtensor_python


class _Mesh:
    def __init__(self, ranks):
        self.mesh = torch.tensor(ranks, dtype=torch.int64).reshape(1, -1)


class _Communicator:
    def __init__(self, device_api_support):
        self.queries = 0
        self._device_api_support = device_api_support

    @property
    def device_api_support(self):
        self.queries += 1
        if isinstance(self._device_api_support, Exception):
            raise self._device_api_support
        return self._device_api_support


class _ProcessGroup:
    def __init__(self, device_api_support):
        self.rank = 0
        self.nccl_communicator = _Communicator(device_api_support)


@pytest.fixture(autouse=True)
def _fresh_path_state(monkeypatch):
    monkeypatch.delenv("NRL_XFERDTENSOR_GOLDEN", raising=False)
    monkeypatch.delenv("NRL_XFERDTENSOR_PYTHON", raising=False)
    monkeypatch.setattr(xfer, "_XFERDTENSOR_PATH_LOGGED", False)


def _xferdtensor_with_stubs(monkeypatch, process_group):
    calls = []
    monkeypatch.setattr(xfer, "_reshard", lambda *a, **k: calls.append("native"))
    monkeypatch.setattr(
        xferdtensor_python,
        "xferdtensor_python_impl",
        lambda *a, **k: calls.append("python"),
    )
    src = xfer.DTensorRef(torch.zeros(2, 4), (4, 4))
    xfer.xferdtensor(
        src,
        _Mesh([0, 1]),
        [Replicate(), Shard(0)],
        None,
        _Mesh([2, 3]),
        [Replicate(), Shard(1)],
        process_group,
    )
    return calls


def test_native_op_used_when_communicator_supports_device_api(monkeypatch):
    calls = _xferdtensor_with_stubs(monkeypatch, _ProcessGroup(True))
    assert calls == ["native"]


def test_python_path_when_communicator_lacks_device_api(monkeypatch, capsys):
    process_group = _ProcessGroup(False)
    calls = _xferdtensor_with_stubs(monkeypatch, process_group)
    assert calls == ["python"]
    out = capsys.readouterr().out
    assert "reshard path: xferdtensor_python (exact-transfer)" in out
    assert "device_api_support=False" in out


def test_python_override_skips_capability_query(monkeypatch):
    monkeypatch.setenv("NRL_XFERDTENSOR_PYTHON", "1")
    process_group = _ProcessGroup(AssertionError("must not query"))
    calls = _xferdtensor_with_stubs(monkeypatch, process_group)
    assert calls == ["python"]
    assert process_group.nccl_communicator.queries == 0


def test_replacement_communicator_selects_its_own_capability(monkeypatch):
    # Model Python reusing an address after the previous communicator is freed.
    monkeypatch.setattr(xfer, "id", lambda obj: 7, raising=False)
    process_group = _ProcessGroup(False)
    assert _xferdtensor_with_stubs(monkeypatch, process_group) == ["python"]
    process_group.nccl_communicator = _Communicator(True)
    assert _xferdtensor_with_stubs(monkeypatch, process_group) == ["native"]


def test_capability_query_failure_propagates(monkeypatch):
    process_group = _ProcessGroup(RuntimeError("communicator unavailable"))
    with pytest.raises(RuntimeError, match="communicator unavailable"):
        _xferdtensor_with_stubs(monkeypatch, process_group)


def test_missing_communicator_uses_python_path(monkeypatch):
    process_group = _ProcessGroup(False)
    process_group.nccl_communicator = None
    assert _xferdtensor_with_stubs(monkeypatch, process_group) == ["python"]
