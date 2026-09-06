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
"""Unit test for ``_broadcast_batched_data_dict`` on a 2-rank gloo group.

Exercises the helper that backs ``_fetch(fetch_policy="leader_broadcast")``.
Runs on CPU (gloo) so it stays in the no-GPU Tier 1 lane.
"""

from __future__ import annotations

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.worker_mixin import _broadcast_batched_data_dict
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _noncontiguous_int16_routes() -> torch.Tensor:
    routes = torch.tensor(
        [[[-32768, -1], [127, 128]], [[255, 256], [1024, 32767]]],
        dtype=torch.int16,
    ).transpose(0, 1)
    assert not routes.is_contiguous()
    return routes


def _in_gloo_group(body, rank: int, world_size: int, tmp_init_file: str, q):
    """Run ``body(rank)`` in a gloo group, reporting the outcome via ``q``."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        body(rank)
        q.put((rank, "ok"))
    except Exception as e:  # pragma: no cover — surface failures to parent
        q.put((rank, f"err: {type(e).__name__}: {e}"))
    finally:
        dist.destroy_process_group()


def _run_two_ranks(body, tmp_init_file: str):
    """Spawn two ranks over ``body`` and require both to report ok."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_in_gloo_group, args=(body, rank, 2, tmp_init_file, q))
        for rank in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    results = sorted([q.get() for _ in range(2)])
    assert results == [(0, "ok"), (1, "ok")], results


def _pixel_rows():
    return [
        torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
        torch.arange(1 * 5 * 4, dtype=torch.float32).reshape(1, 5, 4) + 100,
        None,
    ]


def _packed(rows):
    return PackedTensor(
        [r.clone() if r is not None else None for r in rows],
        dim_to_pack=0,
        pad_to_max_shape=True,
    )


def _round_trip_body(rank: int):
    # ``pixel_values`` is the case that mattered: a PackedTensor is not a
    # torch.Tensor, so before the ``packed_wire`` branch it rode the object
    # list and ``broadcast_object_list`` pickled the pixels into device memory.
    # Rows differ in their trailing dims and one sample has no media, which is
    # what the format exists for.
    rows = _pixel_rows()
    data = (
        BatchedDataDict(
            {
                "input_ids": torch.arange(12, dtype=torch.long).reshape(3, 4),
                "input_lengths": torch.tensor([4, 3, 2], dtype=torch.int32),
                "routed_experts": _noncontiguous_int16_routes(),
                "scalar_meta": "step_42",
                "pixel_values": _packed(rows),
            }
        )
        if rank == 0
        else None
    )

    out = _broadcast_batched_data_dict(
        data, is_leader=(rank == 0), src=0, group=dist.group.WORLD
    )

    assert torch.equal(
        out["input_ids"], torch.arange(12, dtype=torch.long).reshape(3, 4)
    )
    assert torch.equal(out["input_lengths"], torch.tensor([4, 3, 2], dtype=torch.int32))
    assert torch.equal(out["routed_experts"], _noncontiguous_int16_routes())
    assert out["routed_experts"].dtype == torch.int16
    assert out["scalar_meta"] == "step_42"

    packed = out["pixel_values"]
    assert isinstance(packed, PackedTensor), type(packed).__name__
    # Compare on logical rows, not ``.tensors``: ``from_wire`` returns segments
    # flat with a CSR row map, so an empty row contributes no entry there.
    expected = _packed(rows)
    assert (
        packed.logical_segment_counts_by_row()
        == expected.logical_segment_counts_by_row()
        == [1, 1, 0]
    )
    assert torch.equal(packed.as_tensor(), expected.as_tensor())


def _all_empty_body(rank: int):
    # One DP shard of a mixed image/text batch can hold only media-free
    # samples. ``pixel_values`` is still in ``meta.fields``, so the shard
    # rebuilds an empty PackedTensor -- and the key must survive the broadcast,
    # since consumers branch on the key set.
    data = (
        BatchedDataDict(
            {
                "input_ids": torch.arange(8, dtype=torch.long).reshape(2, 4),
                "pixel_values": PackedTensor(
                    [None, None], dim_to_pack=0, pad_to_max_shape=True
                ),
            }
        )
        if rank == 0
        else None
    )

    out = _broadcast_batched_data_dict(
        data, is_leader=(rank == 0), src=0, group=dist.group.WORLD
    )

    assert set(out.keys()) == {"input_ids", "pixel_values"}, sorted(out.keys())
    packed = out["pixel_values"]
    assert isinstance(packed, PackedTensor), type(packed).__name__
    assert packed.logical_segment_counts_by_row() == [0, 0]
    assert packed.as_tensor() is None
    assert packed.pad_to_max_shape is True


def test_leader_broadcast_round_trip(tmp_path):
    _run_two_ranks(_round_trip_body, str(tmp_path / "init"))


def _tensor_payload(device: str) -> dict[str, torch.Tensor]:
    payload = {}
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        values = torch.arange(6, device=device).reshape(2, 3).to(dtype)
        payload[f"{dtype}_matrix"] = values
        payload[f"{dtype}_transposed"] = values.T
        payload[f"{dtype}_sliced"] = values[:, 1:]
        payload[f"{dtype}_scalar"] = torch.tensor(1, dtype=dtype, device=device)
        payload[f"{dtype}_empty"] = torch.empty(2, 0, 3, dtype=dtype, device=device)
    # Exhaust the signed int16 domain so a wrong wire dtype cannot silently
    # truncate negative sentinels or large expert indices.
    payload["int16_domain"] = torch.arange(
        -32768, 32768, dtype=torch.int32, device=device
    ).to(torch.int16)
    return payload


def _tensor_round_trip_body(rank: int, *, source_device: str = "cpu") -> None:
    expected = _tensor_payload(source_device)
    original_values = {key: tensor.clone() for key, tensor in expected.items()}
    data = BatchedDataDict(expected) if rank == 0 else None
    original_strides = {key: tensor.stride() for key, tensor in expected.items()}

    out = _broadcast_batched_data_dict(
        data, is_leader=(rank == 0), src=0, group=dist.group.WORLD
    )

    assert out.keys() == expected.keys()
    for key, tensor in expected.items():
        actual = out[key]
        assert actual.dtype == tensor.dtype, key
        assert actual.shape == tensor.shape, key
        assert actual.device == tensor.device, key
        assert torch.equal(actual, original_values[key]), key
        if rank == 0:
            assert out is data
            assert actual is tensor, key
            assert actual.stride() == original_strides[key], key


def test_leader_broadcast_preserves_tensor_values_layout_and_dtype(tmp_path):
    """Collectives preserve scalars, empty fields and strided tensor values."""
    _run_two_ranks(_tensor_round_trip_body, str(tmp_path / "init_tensors"))


def test_leader_broadcast_keeps_media_free_packed_key(tmp_path):
    """An all-empty packed field keeps its key on both sides of the broadcast.

    ``to_wire`` answers "is there payload", not "is there a field". Deriving
    the broadcast key set from it made a media-free shard emit a different key
    set than the same shard on the independent-fetch path.
    """
    _run_two_ranks(_all_empty_body, str(tmp_path / "init_empty"))


def test_get_replica_group_default_is_none():
    """TQWorkerMixin._get_replica_group must default to None.

    The base default lets ``_fetch(fetch_policy="leader_broadcast")``
    fall back to the independent path when no backend override exists
    (Phase 1 / FSDP2 with TP=CP=PP=1).
    """
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin

    class _Stub(TQWorkerMixin):
        pass

    assert _Stub()._get_replica_group() is None


def _nccl_int16_worker(rank: int, world_size: int) -> None:
    if rank == 0:
        data = BatchedDataDict({"routed_experts": _noncontiguous_int16_routes()})
    else:
        data = None

    out = _broadcast_batched_data_dict(
        data,
        is_leader=(rank == 0),
        src=0,
        group=dist.group.WORLD,
    )

    expected = _noncontiguous_int16_routes()
    assert out["routed_experts"].device.type == "cpu"
    assert out["routed_experts"].dtype == torch.int16
    assert torch.equal(out["routed_experts"], expected)


def test_leader_broadcast_int16_round_trip_nccl(distributed_test_runner):
    """NCCL transport handles non-contiguous Router Replay routes."""
    distributed_test_runner(_nccl_int16_worker, world_size=2, backend="nccl")


def _nccl_tensor_worker(rank: int, world_size: int, *, source_device: str) -> None:
    _tensor_round_trip_body(rank, source_device=source_device)


@pytest.mark.parametrize("source_device", ["cpu", "cuda"])
def test_leader_broadcast_preserves_tensor_devices_nccl(
    distributed_test_runner, source_device
):
    """NCCL preserves the source device type and the leader's original views."""
    distributed_test_runner(
        partial(_nccl_tensor_worker, source_device=source_device),
        world_size=2,
        backend="nccl",
    )
