# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-GPU reproduction for concurrent nccl_reshard PP-stage groups.

The topology is the smallest one that preserves the production overlap:

* global rank 0 is the source for PP stage 0 and joins communicator 0;
* global rank 1 is the source for PP stage 1 and joins communicator 1;
* global rank 2 is the generation receiver and joins both communicators.

The receiver queues the stage transfers with the same stream/event schedule as
``VllmInternalWorkerExtension._nccl_reshard_refit``.  The payload goes through
the repository's real ``StatelessProcessGroup`` and Python exact-transfer
``xferdtensor`` fallback, without loading a model or starting Ray/vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Replicate

from nemo_rl.distributed.refit_watchdog import RefitAbortWatchdog, RefitAborted
from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
from nemo_rl.weight_sync.nccl_reshard_utils import MeshInfo
from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor


WORLD_SIZE = 3
GENERATION_RANK = 2
STAGE_COUNT = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, choices=(1, 2), required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--transfers-per-stage", type=int, default=32)
    parser.add_argument("--tensor-mib", type=int, default=8)
    parser.add_argument("--iteration-timeout-s", type=float, default=10.0)
    parser.add_argument("--coordination-timeout-s", type=float, default=30.0)
    parser.add_argument("--jitter-ms", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    for name in (
        "iterations",
        "transfers_per_stage",
        "tensor_mib",
        "progress_every",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.iteration_timeout_s <= 0 or args.coordination_timeout_s <= 0:
        raise ValueError("timeouts must be positive")
    if args.jitter_ms < 0:
        raise ValueError("--jitter-ms must be non-negative")
    return args


def _find_free_ports(count: int) -> list[int]:
    sockets = []
    try:
        for _ in range(count):
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            candidate.bind(("127.0.0.1", 0))
            sockets.append(candidate)
        return [int(candidate.getsockname()[1]) for candidate in sockets]
    finally:
        for candidate in sockets:
            candidate.close()


def _broadcast_ports(rank: int) -> list[int]:
    payload: list[Optional[list[int]]] = [
        _find_free_ports(STAGE_COUNT) if rank == 0 else None
    ]
    dist.broadcast_object_list(payload, src=0)
    ports = payload[0]
    if ports is None or len(ports) != STAGE_COUNT:
        raise RuntimeError(f"failed to distribute {STAGE_COUNT} rendezvous ports")
    return ports


def _build_groups(
    rank: int, ports: list[int], device: int
) -> dict[int, StatelessProcessGroup]:
    stages = range(STAGE_COUNT) if rank == GENERATION_RANK else (rank,)
    groups = {}
    for stage in stages:
        group_rank = 1 if rank == GENERATION_RANK else 0
        print(
            f"rank={rank} group_init stage={stage} port={ports[stage]} "
            f"group_rank={group_rank}/2",
            flush=True,
        )
        group = StatelessProcessGroup(
            master_address="127.0.0.1",
            port=ports[stage],
            rank=group_rank,
            world_size=2,
        )
        group.init_nccl_communicator(device=device)
        groups[stage] = group
        print(f"rank={rank} group_ready stage={stage}", flush=True)
    return groups


def _wait_for_events(
    events: list[torch.cuda.Event], timeout_s: float, *, label: str
) -> None:
    deadline = time.monotonic() + timeout_s
    while not all(event.query() for event in events):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{label} did not retire within {timeout_s:.1f}s")
        time.sleep(0.01)
    torch.cuda.synchronize()


def _transfer_stage(
    *,
    group: StatelessProcessGroup,
    source: Optional[torch.Tensor],
    destination: Optional[torch.Tensor],
    stream: torch.cuda.Stream,
    transfers: int,
    global_shape: tuple[int, ...],
) -> None:
    source_ref = DTensorRef(source, global_shape) if source is not None else None
    destination_ref = (
        DTensorRef(destination, global_shape) if destination is not None else None
    )
    source_mesh = MeshInfo(torch.tensor([0]))
    destination_mesh = MeshInfo(torch.tensor([1]))
    placements = [Replicate()]
    for _ in range(transfers):
        xferdtensor(
            source_ref,
            source_mesh,
            placements,
            destination_ref,
            destination_mesh,
            placements,
            group,
            stream,
        )


def _expected_value(stage: int, iteration: int) -> float:
    return float(stage * 16 + iteration % 16)


def _run_source(
    *,
    rank: int,
    group: StatelessProcessGroup,
    source: torch.Tensor,
    args: argparse.Namespace,
) -> list[float]:
    durations = []
    stream = torch.cuda.current_stream()
    rng = random.Random(args.seed + rank)
    for iteration in range(args.iterations):
        dist.barrier()
        if args.jitter_ms:
            time.sleep(rng.uniform(0.0, args.jitter_ms) / 1000.0)
        source.fill_(_expected_value(rank, iteration))
        started = time.perf_counter()
        with RefitAbortWatchdog(group, args.iteration_timeout_s) as guard:
            _transfer_stage(
                group=group,
                source=source,
                destination=None,
                stream=stream,
                transfers=args.transfers_per_stage,
                global_shape=tuple(source.shape),
            )
            event = torch.cuda.Event()
            event.record(stream)
            _wait_for_events(
                [event],
                args.iteration_timeout_s + 2.0,
                label=f"rank {rank} iteration {iteration}",
            )
        if guard.fired:
            raise RefitAborted(f"rank {rank} iteration {iteration} watchdog fired")
        durations.append(time.perf_counter() - started)
        dist.barrier()
    return durations


def _run_receiver(
    *,
    groups: dict[int, StatelessProcessGroup],
    destinations: dict[int, torch.Tensor],
    args: argparse.Namespace,
) -> list[float]:
    durations = []
    for iteration in range(args.iterations):
        dist.barrier()
        started = time.perf_counter()
        events: dict[int, torch.cuda.Event] = {}
        num_streams = min(args.streams, STAGE_COUNT)
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        with RefitAbortWatchdog(
            [groups[stage] for stage in range(STAGE_COUNT)],
            args.iteration_timeout_s,
        ) as guard:
            for index, stage in enumerate(range(STAGE_COUNT)):
                previous = index - num_streams
                if previous in events:
                    _wait_for_events(
                        [events[previous]],
                        args.iteration_timeout_s + 2.0,
                        label=f"receiver iteration {iteration} stage {previous}",
                    )
                stage_stream = streams[index % num_streams]
                with torch.cuda.stream(stage_stream):
                    _transfer_stage(
                        group=groups[stage],
                        source=None,
                        destination=destinations[stage],
                        stream=stage_stream,
                        transfers=args.transfers_per_stage,
                        global_shape=tuple(destinations[stage].shape),
                    )
                    event = torch.cuda.Event()
                    event.record(stage_stream)
                    events[index] = event
            _wait_for_events(
                list(events.values()),
                args.iteration_timeout_s + 2.0,
                label=f"receiver iteration {iteration}",
            )
        if guard.fired:
            raise RefitAborted(f"receiver iteration {iteration} watchdog fired")

        for stage, destination in destinations.items():
            observed = float(destination[0].item())
            expected = _expected_value(stage, iteration)
            if observed != expected:
                raise RuntimeError(
                    f"iteration {iteration} stage {stage}: received {observed}, "
                    f"expected {expected}"
                )
        duration = time.perf_counter() - started
        durations.append(duration)
        if (iteration + 1) % args.progress_every == 0:
            print(
                f"receiver_progress iteration={iteration + 1}/{args.iterations} "
                f"last_s={duration:.6f}",
                flush=True,
            )
        dist.barrier()
    return durations


def main() -> None:
    args = _parse_args()
    dist.init_process_group(
        backend="gloo", timeout=timedelta(seconds=args.coordination_timeout_s)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected world size {WORLD_SIZE}, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(
            f"expected at least {WORLD_SIZE} visible GPUs, got {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)

    implicit_order = os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT")
    if implicit_order not in ("0", "1"):
        raise RuntimeError(
            "NCCL_LAUNCH_ORDER_IMPLICIT must be explicitly set to 0 or 1 before "
            "communicator construction"
        )
    requested_streams = os.environ.get("NRL_REFIT_NUM_STREAMS")
    if requested_streams != str(args.streams):
        raise RuntimeError(
            f"NRL_REFIT_NUM_STREAMS={requested_streams!r} does not match "
            f"--streams={args.streams}"
        )

    tensor_bytes = args.tensor_mib * 1024 * 1024
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    if tensor_bytes % element_size:
        raise RuntimeError("tensor byte count is not divisible by BF16 element size")
    numel = tensor_bytes // element_size

    ports = _broadcast_ports(rank)
    groups = _build_groups(rank, ports, local_rank)
    dist.barrier()
    if rank == 0:
        print(
            "repro_config="
            + json.dumps(
                {
                    "implicit_order": int(implicit_order),
                    "iterations": args.iterations,
                    "jitter_ms": args.jitter_ms,
                    "streams": args.streams,
                    "tensor_mib": args.tensor_mib,
                    "transfers_per_stage": args.transfers_per_stage,
                    "world_size": world_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if rank == GENERATION_RANK:
        destinations = {
            stage: torch.empty(numel, device="cuda", dtype=torch.bfloat16)
            for stage in range(STAGE_COUNT)
        }
        durations = _run_receiver(groups=groups, destinations=destinations, args=args)
    else:
        source = torch.empty(numel, device="cuda", dtype=torch.bfloat16)
        durations = _run_source(rank=rank, group=groups[rank], source=source, args=args)

    gathered: list[Optional[dict[str, float]]] = [None] * world_size
    summary = {
        "max_iteration_s": max(durations),
        "mean_iteration_s": sum(durations) / len(durations),
    }
    dist.all_gather_object(gathered, summary)
    if rank == 0:
        print(
            "REFIT_ORDER_REPRO=PASS "
            + json.dumps(
                {
                    "implicit_order": int(implicit_order),
                    "iterations": args.iterations,
                    "rank_timings": gathered,
                    "streams": args.streams,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
