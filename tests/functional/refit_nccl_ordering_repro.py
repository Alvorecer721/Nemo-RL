# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded NCCL refit ordering probe for concurrent trainer PP-stage groups.

Ranks ``0..stages-1`` model trainer pipeline stages. The final rank models one
rollout worker and receives from every stage over a separate two-rank ordinary
NCCL communicator. The transfer loop uses the repository's current
``StatelessProcessGroup``, refit watchdog, and Python exact-transfer
``xferdtensor`` implementation without starting Ray, vLLM, or a model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, cast

import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Replicate

from nemo_rl.distributed.refit_watchdog import RefitAbortWatchdog
from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
from nemo_rl.weight_sync.nccl_reshard_utils import MeshInfo
from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor


PASS_PREFIX = "REFIT_ORDER_REPRO=PASS "
_PAYLOAD_MODULUS = 251
_STAGE_STRIDE = 37
_ITERATION_STRIDE = 17


@dataclass(frozen=True)
class ProbeConfig:
    """Validated settings shared by probe execution and log validation."""

    stages: int
    streams: int
    iterations: int
    transfers_per_stage: int
    tensor_mib: int
    iteration_timeout_s: float
    coordination_timeout_s: float
    jitter_ms: float
    progress_every: int
    seed: int
    validate_log: Optional[Path]

    @property
    def world_size(self) -> int:
        """Return trainer source ranks plus the single rollout destination rank."""
        return self.stages + 1

    @property
    def receiver_rank(self) -> int:
        """Return the global rank of the sole rollout destination."""
        return self.stages


class RankResult(TypedDict):
    """Completion evidence emitted by one distributed probe rank."""

    rank: int
    role: Literal["source", "receiver"]
    stage: Optional[int]
    completed_iterations: int
    transfer_calls: int
    validated_payloads: int
    hostname: str
    slurm_node_id: int
    slurm_local_id: int
    max_iteration_s: float
    mean_iteration_s: float


class ProbeCoreResult(TypedDict):
    """Proof that every requested rank operation completed."""

    completed_iterations: int
    iterations: int
    stages: int
    status: Literal["pass"]
    streams: int
    transfer_pairs: int
    transfers_per_stage: int
    validated_payloads: int
    world_size: int


class ProbeResult(ProbeCoreResult):
    """Launcher-visible result emitted only after communicator teardown."""

    teardown: Literal["complete"]


def parse_args(argv: Optional[Sequence[str]] = None) -> ProbeConfig:
    """Parse and validate the probe command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=int, choices=(2, 4), required=True)
    parser.add_argument("--streams", type=int, choices=(1, 2), required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--transfers-per-stage", type=int, default=32)
    parser.add_argument("--tensor-mib", type=int, default=8)
    parser.add_argument("--iteration-timeout-s", type=float, default=10.0)
    parser.add_argument("--coordination-timeout-s", type=float, default=30.0)
    parser.add_argument("--jitter-ms", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--validate-log",
        type=Path,
        help="validate a completed probe log instead of initializing distributed state",
    )
    args = parser.parse_args(argv)
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
    return ProbeConfig(**vars(args))


def validate_runtime_environment(
    config: ProbeConfig, environ: Mapping[str, str]
) -> None:
    """Reject implicit transport selection or a launcher/probe stream mismatch."""
    if environ.get("NRL_XFERDTENSOR_PYTHON") != "1":
        raise RuntimeError("NRL_XFERDTENSOR_PYTHON must be explicitly set to 1")
    requested_streams = environ.get("NRL_REFIT_NUM_STREAMS")
    if requested_streams != str(config.streams):
        raise RuntimeError(
            f"NRL_REFIT_NUM_STREAMS={requested_streams!r} does not match "
            f"--streams={config.streams}"
        )
    implicit_order = environ.get("NCCL_LAUNCH_ORDER_IMPLICIT")
    if implicit_order not in ("0", "1"):
        raise RuntimeError(
            "NCCL_LAUNCH_ORDER_IMPLICIT must be explicitly set to 0 or 1"
        )
    if environ.get("NRL_XFERDTENSOR_GOLDEN", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise RuntimeError("NRL_XFERDTENSOR_GOLDEN must not enable the golden path")


def fill_payload(tensor: torch.Tensor, *, stage: int, iteration: int) -> None:
    """Fill a tensor with an exactly representable, element-varying BF16 pattern."""
    indices = torch.arange(tensor.numel(), device=tensor.device, dtype=torch.int32)
    indices.remainder_(_PAYLOAD_MODULUS)
    indices.add_(stage * _STAGE_STRIDE + iteration * _ITERATION_STRIDE)
    indices.remainder_(_PAYLOAD_MODULUS)
    tensor.copy_(indices.reshape(tensor.shape))


def validate_payload(tensor: torch.Tensor, *, stage: int, iteration: int) -> None:
    """Validate every received element against the stage and iteration pattern."""
    expected = torch.empty_like(tensor)
    fill_payload(expected, stage=stage, iteration=iteration)
    observed_flat = tensor.flatten()
    expected_flat = expected.flatten()
    mismatch = torch.nonzero(observed_flat != expected_flat, as_tuple=False)
    if mismatch.numel() == 0:
        return
    flat_index = int(mismatch[0, 0].item())
    observed = observed_flat[flat_index].item()
    wanted = expected_flat[flat_index].item()
    raise RuntimeError(
        f"iteration {iteration} stage {stage}: element {flat_index} expected "
        f"{wanted}, observed {observed}"
    )


def _require_result_field(
    *, result: Mapping[str, Any], field: str, expected: Any, rank: int
) -> None:
    observed = result.get(field)
    if observed != expected:
        raise RuntimeError(f"rank {rank} {field}={observed!r}, expected {expected!r}")


def validate_rank_results(
    rank_results: Sequence[RankResult], config: ProbeConfig
) -> ProbeCoreResult:
    """Validate all rank completion records and return the final PASS payload."""
    if len(rank_results) != config.world_size:
        raise RuntimeError(
            f"received {len(rank_results)} rank results, expected {config.world_size}"
        )
    by_rank = {result.get("rank"): result for result in rank_results}
    if set(by_rank) != set(range(config.world_size)):
        raise RuntimeError(f"rank results are missing or duplicated: {sorted(by_rank)}")

    placements = [
        (result.get("hostname"), result.get("slurm_local_id"))
        for result in rank_results
    ]
    if len(set(placements)) != len(placements):
        raise RuntimeError(f"duplicate placement in rank results: {placements}")

    expected_source_calls = config.iterations * config.transfers_per_stage
    for rank in range(config.stages):
        result = by_rank[rank]
        _require_result_field(result=result, field="role", expected="source", rank=rank)
        _require_result_field(result=result, field="stage", expected=rank, rank=rank)
        _require_result_field(
            result=result, field="slurm_node_id", expected=0, rank=rank
        )
        _require_result_field(
            result=result, field="slurm_local_id", expected=rank, rank=rank
        )
        _require_result_field(
            result=result,
            field="completed_iterations",
            expected=config.iterations,
            rank=rank,
        )
        _require_result_field(
            result=result,
            field="transfer_calls",
            expected=expected_source_calls,
            rank=rank,
        )
        _require_result_field(
            result=result, field="validated_payloads", expected=0, rank=rank
        )

    receiver = by_rank[config.receiver_rank]
    expected_pairs = expected_source_calls * config.stages
    _require_result_field(
        result=receiver,
        field="role",
        expected="receiver",
        rank=config.receiver_rank,
    )
    _require_result_field(
        result=receiver, field="stage", expected=None, rank=config.receiver_rank
    )
    _require_result_field(
        result=receiver,
        field="slurm_node_id",
        expected=1,
        rank=config.receiver_rank,
    )
    _require_result_field(
        result=receiver,
        field="slurm_local_id",
        expected=0,
        rank=config.receiver_rank,
    )
    _require_result_field(
        result=receiver,
        field="completed_iterations",
        expected=config.iterations,
        rank=config.receiver_rank,
    )
    _require_result_field(
        result=receiver,
        field="transfer_calls",
        expected=expected_pairs,
        rank=config.receiver_rank,
    )
    _require_result_field(
        result=receiver,
        field="validated_payloads",
        expected=config.iterations * config.stages,
        rank=config.receiver_rank,
    )
    for rank, result in by_rank.items():
        for field in ("max_iteration_s", "mean_iteration_s"):
            value = result.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise RuntimeError(f"rank {rank} has invalid {field}={value!r}")

    return {
        "completed_iterations": config.iterations,
        "iterations": config.iterations,
        "stages": config.stages,
        "status": "pass",
        "streams": config.streams,
        "transfer_pairs": expected_pairs,
        "transfers_per_stage": config.transfers_per_stage,
        "validated_payloads": config.iterations * config.stages,
        "world_size": config.world_size,
    }


def validate_result_log(path: Path, config: ProbeConfig) -> ProbeResult:
    """Require one complete, configuration-matching PASS record in a probe log."""
    pass_lines = [
        line.removeprefix(PASS_PREFIX)
        for line in path.read_text().splitlines()
        if line.startswith(PASS_PREFIX)
    ]
    if len(pass_lines) != 1:
        raise RuntimeError(
            f"expected exactly one {PASS_PREFIX.strip()} record in {path}, "
            f"found {len(pass_lines)}"
        )
    try:
        result = json.loads(pass_lines[0])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid PASS record in {path}: {error}") from error
    expected = {
        "completed_iterations": config.iterations,
        "iterations": config.iterations,
        "stages": config.stages,
        "status": "pass",
        "streams": config.streams,
        "teardown": "complete",
        "transfer_pairs": (
            config.iterations * config.transfers_per_stage * config.stages
        ),
        "transfers_per_stage": config.transfers_per_stage,
        "validated_payloads": config.iterations * config.stages,
        "world_size": config.world_size,
    }
    if result != expected:
        raise RuntimeError(f"PASS record does not match requested probe: {result!r}")
    return cast(ProbeResult, result)


def _reserve_endpoint() -> tuple[tuple[str, int], socket.socket]:
    reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    reservation.bind(("", 0))
    return (socket.gethostname(), int(reservation.getsockname()[1])), reservation


def _collect_endpoints(rank: int, config: ProbeConfig) -> list[tuple[str, int]]:
    reservation: Optional[socket.socket] = None
    local_endpoint: Optional[tuple[str, int]] = None
    if rank < config.stages:
        local_endpoint, reservation = _reserve_endpoint()
    gathered: list[Optional[tuple[str, int]]] = [None for _ in range(config.world_size)]
    try:
        dist.all_gather_object(gathered, local_endpoint)
    finally:
        if reservation is not None:
            reservation.close()
    endpoints = gathered[: config.stages]
    if any(endpoint is None for endpoint in endpoints):
        raise RuntimeError(f"failed to collect stage endpoints: {gathered}")
    return [endpoint for endpoint in endpoints if endpoint is not None]


def _build_groups(
    *,
    rank: int,
    endpoints: Sequence[tuple[str, int]],
    device: int,
    config: ProbeConfig,
    groups: dict[int, StatelessProcessGroup],
) -> None:
    stages: Sequence[int]
    if rank == config.receiver_rank:
        stages = range(config.stages)
    else:
        stages = (rank,)
    for stage in stages:
        master_address, port = endpoints[stage]
        group_rank = 1 if rank == config.receiver_rank else 0
        print(
            f"rank={rank} group_init stage={stage} address={master_address} "
            f"port={port} group_rank={group_rank}/2",
            flush=True,
        )
        group = StatelessProcessGroup(
            master_address=master_address,
            port=port,
            rank=group_rank,
            world_size=2,
        )
        groups[stage] = group
        group.init_nccl_communicator(device=device)
        print(f"rank={rank} group_ready stage={stage}", flush=True)


def _wait_for_events(
    events: Sequence[torch.cuda.Event], timeout_s: float, *, label: str
) -> None:
    deadline = time.monotonic() + timeout_s
    while not all(event.query() for event in events):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{label} did not retire within {timeout_s:.1f}s")
        time.sleep(0.01)


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


def _coordinate_iteration_result(
    *, rank: int, iteration: int, local_error: Optional[str], config: ProbeConfig
) -> None:
    errors: list[Optional[str]] = [None for _ in range(config.world_size)]
    dist.all_gather_object(errors, local_error)
    failures = [
        f"rank {failed_rank}: {error}"
        for failed_rank, error in enumerate(errors)
        if error
    ]
    if failures:
        raise RuntimeError(
            f"iteration {iteration} failed (reported by rank {rank}): "
            + "; ".join(failures)
        )


def _run_source(
    *,
    rank: int,
    group: StatelessProcessGroup,
    source: torch.Tensor,
    config: ProbeConfig,
) -> list[float]:
    durations = []
    stream = torch.cuda.current_stream()
    rng = random.Random(config.seed + rank)
    for iteration in range(config.iterations):
        dist.barrier()
        if config.jitter_ms:
            time.sleep(rng.uniform(0.0, config.jitter_ms) / 1000.0)
        fill_payload(source, stage=rank, iteration=iteration)
        started = time.perf_counter()
        local_error = None
        try:
            with RefitAbortWatchdog(group, config.iteration_timeout_s) as guard:
                _transfer_stage(
                    group=group,
                    source=source,
                    destination=None,
                    stream=stream,
                    transfers=config.transfers_per_stage,
                    global_shape=tuple(source.shape),
                )
                event = torch.cuda.Event()
                event.record(stream)
                _wait_for_events(
                    [event],
                    config.iteration_timeout_s + 2.0,
                    label=f"rank {rank} iteration {iteration}",
                )
            if guard.fired:
                local_error = "refit watchdog fired"
        except Exception as error:  # noqa: BLE001
            local_error = f"{type(error).__name__}: {error}"
        duration = time.perf_counter() - started
        _coordinate_iteration_result(
            rank=rank,
            iteration=iteration,
            local_error=local_error,
            config=config,
        )
        durations.append(duration)
    return durations


def _run_receiver(
    *,
    rank: int,
    groups: Mapping[int, StatelessProcessGroup],
    destinations: Mapping[int, torch.Tensor],
    config: ProbeConfig,
) -> list[float]:
    durations = []
    streams = [torch.cuda.Stream() for _ in range(config.streams)]
    for iteration in range(config.iterations):
        dist.barrier()
        started = time.perf_counter()
        local_error = None
        try:
            events: dict[int, torch.cuda.Event] = {}
            with RefitAbortWatchdog(
                [groups[stage] for stage in range(config.stages)],
                config.iteration_timeout_s,
            ) as guard:
                for index, stage in enumerate(range(config.stages)):
                    previous = index - config.streams
                    if previous in events:
                        _wait_for_events(
                            [events[previous]],
                            config.iteration_timeout_s + 2.0,
                            label=(f"receiver iteration {iteration} stage {previous}"),
                        )
                    stage_stream = streams[index % config.streams]
                    with torch.cuda.stream(stage_stream):
                        _transfer_stage(
                            group=groups[stage],
                            source=None,
                            destination=destinations[stage],
                            stream=stage_stream,
                            transfers=config.transfers_per_stage,
                            global_shape=tuple(destinations[stage].shape),
                        )
                        event = torch.cuda.Event()
                        event.record(stage_stream)
                        events[index] = event
                _wait_for_events(
                    list(events.values()),
                    config.iteration_timeout_s + 2.0,
                    label=f"receiver iteration {iteration}",
                )
            if guard.fired:
                local_error = "refit watchdog fired"
            else:
                for stage, destination in destinations.items():
                    validate_payload(destination, stage=stage, iteration=iteration)
        except Exception as error:  # noqa: BLE001
            local_error = f"{type(error).__name__}: {error}"
        duration = time.perf_counter() - started
        _coordinate_iteration_result(
            rank=rank,
            iteration=iteration,
            local_error=local_error,
            config=config,
        )
        durations.append(duration)
        if (iteration + 1) % config.progress_every == 0:
            print(
                f"receiver_progress iteration={iteration + 1}/{config.iterations} "
                f"last_s={duration:.6f}",
                flush=True,
            )
    return durations


def _rank_result(
    *, rank: int, durations: Sequence[float], config: ProbeConfig
) -> RankResult:
    is_receiver = rank == config.receiver_rank
    return {
        "rank": rank,
        "role": "receiver" if is_receiver else "source",
        "stage": None if is_receiver else rank,
        "completed_iterations": len(durations),
        "transfer_calls": (
            config.iterations
            * config.transfers_per_stage
            * (config.stages if is_receiver else 1)
        ),
        "validated_payloads": config.iterations * config.stages if is_receiver else 0,
        "hostname": socket.gethostname(),
        "slurm_node_id": int(os.environ.get("SLURM_NODEID", "0")),
        "slurm_local_id": int(os.environ.get("SLURM_LOCALID", str(rank))),
        "max_iteration_s": max(durations),
        "mean_iteration_s": sum(durations) / len(durations),
    }


def _abort_groups(groups: Mapping[int, StatelessProcessGroup]) -> None:
    for group in groups.values():
        group.abort()


def run_probe(config: ProbeConfig) -> None:
    """Run the distributed GPU probe and print PASS only after clean teardown."""
    validate_runtime_environment(config, os.environ)
    dist.init_process_group(
        backend="gloo", timeout=timedelta(seconds=config.coordination_timeout_s)
    )
    rank = dist.get_rank()
    groups: dict[int, StatelessProcessGroup] = {}
    final_result: Optional[ProbeCoreResult] = None
    try:
        world_size = dist.get_world_size()
        if world_size != config.world_size:
            raise RuntimeError(
                f"expected world size {config.world_size}, got {world_size}"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible_device_count = torch.cuda.device_count()
        if visible_device_count != 1 or local_rank != 0:
            raise RuntimeError(
                "the Slurm launcher must expose exactly one GPU per rank at "
                f"LOCAL_RANK=0, got {visible_device_count} GPUs and "
                f"LOCAL_RANK={local_rank}"
            )
        torch.cuda.set_device(local_rank)
        endpoints = _collect_endpoints(rank, config)
        _build_groups(
            rank=rank,
            endpoints=endpoints,
            device=local_rank,
            config=config,
            groups=groups,
        )
        dist.barrier()
        if rank == 0:
            print(
                "repro_config="
                + json.dumps(
                    {
                        "implicit_order": int(os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"]),
                        "iterations": config.iterations,
                        "jitter_ms": config.jitter_ms,
                        "stage_endpoints": endpoints,
                        "stages": config.stages,
                        "streams": config.streams,
                        "tensor_mib": config.tensor_mib,
                        "transfers_per_stage": config.transfers_per_stage,
                        "world_size": config.world_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        tensor_bytes = config.tensor_mib * 1024 * 1024
        element_size = torch.empty((), dtype=torch.bfloat16).element_size()
        if tensor_bytes % element_size:
            raise RuntimeError(
                "tensor byte count is not divisible by BF16 element size"
            )
        numel = tensor_bytes // element_size
        if rank == config.receiver_rank:
            destinations = {
                stage: torch.empty(numel, device="cuda", dtype=torch.bfloat16)
                for stage in range(config.stages)
            }
            durations = _run_receiver(
                rank=rank,
                groups=groups,
                destinations=destinations,
                config=config,
            )
        else:
            source = torch.empty(numel, device="cuda", dtype=torch.bfloat16)
            durations = _run_source(
                rank=rank,
                group=groups[rank],
                source=source,
                config=config,
            )

        gathered: list[Optional[RankResult]] = [None for _ in range(config.world_size)]
        dist.all_gather_object(
            gathered, _rank_result(rank=rank, durations=durations, config=config)
        )
        if rank == 0:
            rank_results = [result for result in gathered if result is not None]
            final_result = validate_rank_results(rank_results, config)

        _abort_groups(groups)
        groups.clear()
        dist.barrier()
    finally:
        _abort_groups(groups)
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        if final_result is None:
            raise RuntimeError("rank results were not validated")
        completed_result = cast(ProbeResult, {**final_result, "teardown": "complete"})
        print(PASS_PREFIX + json.dumps(completed_result, sort_keys=True), flush=True)


def main() -> None:
    """Validate an existing log or execute the distributed probe."""
    config = parse_args()
    if config.validate_log is not None:
        result = validate_result_log(config.validate_log, config)
        print("REFIT_ORDER_REPRO_LOG=VALID " + json.dumps(result, sort_keys=True))
        return
    run_probe(config)


if __name__ == "__main__":
    main()
