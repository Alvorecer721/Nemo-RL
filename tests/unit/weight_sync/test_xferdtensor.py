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


@pytest.mark.parametrize("ranks", [[[2, 4]], [[3, 2]], [[[2, 3]]]])
def test_native_unsupported_mesh_uses_exact_transfer(monkeypatch, ranks):
    """NCCL M2N requires at most two axes and contiguous row-major ranks.

    vLLM DP replicas of a PP stage can be separated by other stages' ranks.
    They still must refit successfully when the parent has device API support.
    """
    calls = []
    monkeypatch.setattr(xfer, "_reshard", lambda *a, **k: calls.append("native"))
    monkeypatch.setattr(
        xferdtensor_python,
        "xferdtensor_python_impl",
        lambda *a, **k: calls.append("python"),
    )
    destination = _Mesh([2, 3])
    destination.mesh = torch.tensor(ranks)
    xfer.xferdtensor(
        xfer.DTensorRef(torch.zeros(2, 4), (4, 4)),
        _Mesh([0, 1]),
        [Replicate(), Shard(0)],
        None,
        destination,
        [Replicate()] * destination.mesh.ndim,
        _ProcessGroup(True),
    )
    assert calls == ["python"]


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


def test_offstage_receive_participates_without_allocating(monkeypatch):
    from nemo_rl.weight_sync.xferdtensor import receive_resharded_param

    calls = []
    monkeypatch.setattr(xfer, "xferdtensor", lambda *args: calls.append(args))
    info = {
        "name": "model.layers.1.mlp.down_proj.weight",
        "global_shape": (8, 4),
        "dtype": "torch.bfloat16",
        "src_mesh_info": _Mesh([0]),
        "dst_mesh_info": _Mesh([1]),
        "src_placements": [Replicate(), Replicate()],
        "dst_placements": [Replicate(), Replicate()],
    }
    group = _ProcessGroup(False)
    group.rank = 2
    receive_resharded_param(info, None, group, None, device=torch.device("cpu"))
    assert len(calls) == 1
    assert calls[0][3]._local_tensor is None
    assert calls[0][3].shape == (8, 4)
    group.rank = 1
    with pytest.raises(ValueError, match="destination"):
        receive_resharded_param(info, None, group, None, device=torch.device("cpu"))


def test_native_idle_rank_supplies_explicit_shape_and_dtype(monkeypatch):
    recorded = []
    monkeypatch.setattr(xfer, "_reshard", lambda *a, **kw: recorded.append((a, kw)))
    metadata = xfer.DTensorRef(
        None, (8, 4), dtype=torch.bfloat16, device=torch.device("cpu")
    )
    xfer.xferdtensor(
        None,
        _Mesh([0]),
        [Replicate(), Replicate()],
        metadata,
        _Mesh([1, 2]),
        [Replicate(), Shard(0)],
        _ProcessGroup(True),
    )
    args, kwargs = recorded[0]
    assert args[:2] == (None, None)
    assert kwargs["dst_local_shape"] == (4, 4)
    assert kwargs["dst_dtype"] == torch.bfloat16
