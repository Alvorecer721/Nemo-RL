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
"""Fault injection for the owned functional-test actors only."""

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import ray


def exit_pipeline_stage(worker: Any, stage: int, marker: str) -> None:
    # vLLM is installed only inside the generation worker environment.
    from vllm.distributed import get_pp_group

    if get_pp_group().rank_in_group == stage:
        Path(marker).write_text(f"pid={os.getpid()} stage={stage}\n")
        os._exit(86)


def arm_pipeline_stage_exit(worker: Any, stage: int, directory: str) -> None:
    # This callable executes inside an isolated vLLM test worker.
    from vllm.distributed import get_pp_group, get_tp_group

    from nemo_rl.distributed import refit_watchdog

    pp_rank = get_pp_group().rank_in_group
    tp_rank = get_tp_group().rank_in_group
    root = Path(directory)
    original = refit_watchdog.hold_refit_for_fault_injection

    def fail_once_in_receive() -> None:
        refit_watchdog.hold_refit_for_fault_injection = original
        (root / f"entered-{pp_rank}-{tp_rank}").write_text(str(os.getpid()))
        deadline = time.monotonic() + 60
        while not (root / "release").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Fault harness did not release the receive")
            time.sleep(0.02)
        exit_pipeline_stage(worker, stage, str(root / "exited"))

    refit_watchdog.hold_refit_for_fault_injection = fail_once_in_receive


def lose_stage_between_refits(
    generation: Any, synchronizer: Any, directory: Path, stage: int
) -> dict[str, Any]:
    """Lose one stage, exclude its complete engine, and rebuild both groups."""
    directory.mkdir(parents=True, exist_ok=False)
    marker = directory / "exited"
    victim = generation._refit_leader_workers()[0]
    start = time.perf_counter()
    try:
        ray.get(victim.exit_pipeline_stage.remote(stage, str(marker)), timeout=60)
    except ray.exceptions.RayError:
        assert marker.exists(), "The failure did not reach the intended PP stage"
    else:
        raise AssertionError("The engine reported success after its stage exited")
    assert synchronizer.reconcile_communicator([0])
    assert synchronizer._built_membership.surviving_shards == [1]
    return {"rebuild_seconds": time.perf_counter() - start, "absent_engine": 0}


def lose_stage_during_refit(
    generation: Any,
    refit: Callable[[], Any],
    *,
    directory: Path,
    stage: int,
    engine_world_size: int,
    deadline_seconds: float,
) -> dict[str, Any]:
    """Require bounded failure; never resume serving or reuse a lost CUDA context."""
    directory.mkdir(parents=True, exist_ok=False)
    victim = generation._refit_leader_workers()[0]
    ray.get(victim.arm_pipeline_stage_exit.remote(stage, str(directory)), timeout=30)
    errors: list[BaseException] = []

    def attempt_refit() -> None:
        try:
            refit()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=attempt_refit, daemon=True)
    start = time.perf_counter()
    thread.start()
    entered_deadline = time.monotonic() + 30
    while len(list(directory.glob("entered-*"))) != engine_world_size:
        if not thread.is_alive() or time.monotonic() > entered_deadline:
            raise AssertionError("Not every victim-engine rank entered the refit")
        time.sleep(0.02)
    (directory / "release").touch()
    thread.join(timeout=deadline_seconds)
    elapsed = time.perf_counter() - start
    assert not thread.is_alive(), f"Refit did not unwind within {deadline_seconds}s"
    assert (directory / "exited").exists(), "The intended PP stage did not exit"
    assert errors, "Refit reported success after losing a PP stage"
    return {
        "failure_seconds": elapsed,
        "error": str(errors[0]),
        "resumed_generation": False,
    }
