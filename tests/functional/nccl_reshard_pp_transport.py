# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Four-GPU refit transport gate: changed weights, off-stage ranks and replicas."""

import argparse
import os
import socket
from datetime import timedelta

import torch
import torch.multiprocessing as mp
from torch.distributed.tensor import Replicate, Shard

from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
from nemo_rl.weight_sync.nccl_reshard_utils import MeshInfo
from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor
from nemo_rl.weight_sync.xferdtensor_python import clear_xferdtensor_python_caches


def worker(rank: int, port: int, mode: str) -> None:
    os.environ["NRL_XFERDTENSOR_PYTHON"] = str(int(mode == "python"))
    os.environ["NRL_XFERDTENSOR_GOLDEN"] = str(int(mode == "golden"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    group = StatelessProcessGroup("127.0.0.1", port + 1, rank, 4)
    group.init_nccl_communicator(device=rank)
    if mode == "native" and not group.nccl_communicator.device_api_support:
        raise RuntimeError("Native M2N unavailable on this communicator")
    stream = torch.cuda.current_stream()
    cases = [
        ("pp_stage0", [0, 1], [Shard(0)], [2], [Replicate()]),
        ("pp_stage1", [0, 1], [Shard(0)], [3], [Replicate()]),
        ("replicas_idle", [0], [Replicate()], [1, 2], [Replicate()]),
        ("pp1_control", [0, 1], [Shard(0)], [2, 3], [Shard(1)]),
        ("noncontiguous_fallback", [0], [Replicate()], [1, 3], [Shard(1)]),
        ("source_replicas", [0, 1], [Replicate()], [2, 3], [Shard(1)]),
    ]
    for iteration in range(3):
        for case, src_ranks, src_placements, dst_ranks, dst_placements in cases:
            reference = (
                torch.arange(64, device=device, dtype=torch.bfloat16).reshape(8, 8)
                + iteration * 32
            )
            src = None
            dst = None
            if rank in src_ranks:
                local = (
                    reference.chunk(len(src_ranks), dim=0)[
                        src_ranks.index(rank)
                    ].clone()
                    if isinstance(src_placements[0], Shard)
                    else reference.clone()
                )
                src = DTensorRef(local, reference.shape)
            if rank in dst_ranks:
                expected = (
                    reference.chunk(len(dst_ranks), dim=1)[dst_ranks.index(rank)]
                    if isinstance(dst_placements[0], Shard)
                    else reference
                )
                dst = DTensorRef(torch.full_like(expected, -1), reference.shape)
            elif src is None:
                dst = DTensorRef(
                    None, reference.shape, dtype=reference.dtype, device=device
                )
            xferdtensor(
                src,
                MeshInfo(torch.tensor(src_ranks)),
                src_placements,
                dst,
                MeshInfo(torch.tensor(dst_ranks)),
                dst_placements,
                group,
                stream,
            )
            torch.cuda.synchronize()
            if rank in dst_ranks:
                torch.testing.assert_close(dst._local_tensor, expected, atol=0, rtol=0)
            torch.distributed.barrier()
            if rank == 0:
                print(f"PASS {mode} {case} iteration={iteration}", flush=True)
    clear_xferdtensor_python_caches(group)
    torch.distributed.barrier()
    group.abort()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["python", "native", "golden"], default="python"
    )
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args.mode), nprocs=4, join=True)
