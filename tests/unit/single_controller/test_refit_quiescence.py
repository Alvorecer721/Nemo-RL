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

"""A refit must not mutate weights while an async engine can still use them."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ray.exceptions import RayActorError

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.distributed.refit_watchdog import RefitAborted
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetHealth,
    ShardState,
)
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from nemo_rl.weight_sync.membership import plan_refit_membership
from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
    NcclReshardWeightSynchronizer,
)


def _controller(
    monkeypatch,
    *,
    clear_cache=False,
    pause_fails=False,
    refit_fails=False,
    synchronizer_cls=NcclReshardWeightSynchronizer,
):
    calls = []
    state = {"paused": False, "version": 0}
    leader = MagicMock()

    def pause(**kwargs):
        calls.append(("pause", kwargs["clear_cache"]))
        if pause_fails:
            raise TimeoutError("engine did not pause")
        state["paused"] = True
        return True

    def refit(**kwargs):
        assert state["paused"], "refit reached live weights before engine quiescence"
        calls.append("refit")
        if refit_fails:
            raise RuntimeError("partial refit")

    def stamp(**kwargs):
        assert state["paused"]
        state["version"] = kwargs["version"]
        calls.append("stamp")

    def resume():
        assert state["version"] == 1
        calls.append("resume")
        state["paused"] = False
        return True

    leader.pause_generation_async.remote.side_effect = pause
    leader.resume_generation_async.remote.side_effect = resume
    leader.set_rollout_weight_version.remote.side_effect = stamp
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda refs, **kwargs: refs,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.wait",
        lambda refs, **kwargs: (refs, []),
    )
    generation = object.__new__(VllmGeneration)
    generation.cfg = {"vllm_cfg": {"async_engine": True}}
    generation.dp_size = 1
    generation._refit_membership = None
    generation.worker_group = SimpleNamespace(workers=[leader])
    generation.invalidate_kv_cache = MagicMock()
    generation.shutdown = MagicMock()

    cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(cls)
    ctrl._gen = generation
    ctrl._gen_fleet = None
    ctrl._async_cfg = SimpleNamespace(
        recompute_kv_cache_after_weight_updates=clear_cache,
        generation_fleet_health=SimpleNamespace(refit_timeout_s=None),
    )
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._inflight_by_group_id = {}
    ctrl._rollout_recovery_enabled = False
    ctrl._master_config = SimpleNamespace(
        env={}, token_capture=SimpleNamespace(enabled=True)
    )
    ctrl._weight_synchronizer = object.__new__(synchronizer_cls)
    ctrl._weight_synchronizer.sync_weights = refit
    ctrl._trainer = SimpleNamespace(sync_params_before_refit=MagicMock())
    ctrl._trainer_version = 1
    ctrl._rollout_manager = SimpleNamespace(
        suspend_request_deadlines=MagicMock(), resume_request_deadlines=MagicMock()
    )
    ctrl._timer = Timer()
    return ctrl, state, calls


@pytest.mark.parametrize("clear_cache", [False, True])
@pytest.mark.parametrize(
    "synchronizer_cls", [NcclReshardWeightSynchronizer, CollectiveWeightSynchronizer]
)
def test_refit_quiesces_engine_and_stamps_before_resuming(
    monkeypatch, clear_cache, synchronizer_cls
):
    ctrl, state, calls = _controller(
        monkeypatch, clear_cache=clear_cache, synchronizer_cls=synchronizer_cls
    )
    asyncio.run(ctrl._sync_weights())
    assert calls == [("pause", clear_cache), "refit", "stamp", "resume"]
    assert not state["paused"]
    assert ctrl._rollout_permitted.is_set()
    ctrl._gen.invalidate_kv_cache.assert_not_called()
    ctrl._rollout_manager.suspend_request_deadlines.assert_called_once()
    ctrl._rollout_manager.resume_request_deadlines.assert_called_once()


def test_failed_pause_never_starts_refit_or_reopens_dispatch(monkeypatch):
    ctrl, _, calls = _controller(monkeypatch, pause_fails=True)
    with pytest.raises(TimeoutError, match="engine did not pause"):
        asyncio.run(ctrl._sync_weights())
    assert calls == [("pause", False)]
    assert not ctrl._rollout_permitted.is_set()
    ctrl._rollout_manager.resume_request_deadlines.assert_not_called()


def test_partial_refit_leaves_engine_paused_and_dispatch_closed(monkeypatch):
    ctrl, state, calls = _controller(monkeypatch, refit_fails=True)
    with pytest.raises(RuntimeError, match="partial refit"):
        asyncio.run(ctrl._sync_weights())
    assert calls == [("pause", False), "refit"]
    assert state["paused"]
    assert not ctrl._rollout_permitted.is_set()
    ctrl._rollout_manager.resume_request_deadlines.assert_not_called()


def test_pause_settles_all_calls_before_recovering_a_dead_actor(monkeypatch):
    ctrl, state, calls = _controller(monkeypatch)
    dead = MagicMock()
    dead_ref = object()
    dead.pause_generation_async.remote.return_value = dead_ref
    ctrl._gen.dp_size = 2
    ctrl._gen.worker_group.workers.append(dead)

    def wait(refs, **kwargs):
        assert kwargs["num_returns"] == len(refs)
        calls.append("settled")
        return refs, []

    def get(refs, **kwargs):
        if any(ref is dead_ref for ref in refs):
            assert calls[-1] == "settled"
            raise RayActorError("actor died while pausing")
        return refs

    async def recover(_failure):
        assert state["paused"]
        calls.append("recover")
        ctrl._gen._refit_membership = plan_refit_membership(
            surviving_shards=[0], dp_size=2, total_gen_workers=2, train_world_size=1
        )

    monkeypatch.setattr("nemo_rl.models.generation.vllm.vllm_generation.ray.wait", wait)
    monkeypatch.setattr("nemo_rl.models.generation.vllm.vllm_generation.ray.get", get)
    ctrl._recover_from_failed_refit = recover
    ctrl._recovery_window = nullcontext
    ctrl._record_refit_landed = MagicMock()
    asyncio.run(ctrl._sync_weights())
    assert calls == [
        ("pause", False),
        "settled",
        "recover",
        ("pause", False),
        "settled",
        "refit",
        "stamp",
        "resume",
    ]
    ctrl._record_refit_landed.assert_called_once_with({0})
    dead.set_rollout_weight_version.remote.assert_not_called()
    dead.resume_generation_async.remote.assert_not_called()
    assert ctrl._rollout_permitted.is_set()


def test_pending_pause_is_fatal_and_cannot_arrive_after_retry_resume(monkeypatch):
    ctrl, state, calls = _controller(monkeypatch)
    ctrl._recover_from_failed_refit = AsyncMock()
    pending = object()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.wait",
        lambda refs, **kwargs: ([], [pending]),
    )
    with pytest.raises(TimeoutError, match="pause.*settle"):
        asyncio.run(ctrl._sync_weights())
    assert calls == [("pause", False)]
    assert state["paused"]
    assert not ctrl._rollout_permitted.is_set()
    ctrl._recover_from_failed_refit.assert_not_awaited()
    ctrl._rollout_manager.resume_request_deadlines.assert_not_called()


@pytest.mark.parametrize("phase", ["transfer", "stamp", "resume"])
def test_replacement_or_finalization_failure_keeps_dispatch_closed(monkeypatch, phase):
    ctrl, state, calls = _controller(monkeypatch)
    leader = ctrl._gen.worker_group.workers[0]
    replacement = MagicMock()
    if phase == "transfer":
        original = ctrl._weight_synchronizer.sync_weights

        def refit(**kwargs):
            original(**kwargs)
            ctrl._gen.worker_group.workers[0] = replacement

        ctrl._weight_synchronizer.sync_weights = refit
        expected = "participants changed"
    elif phase == "stamp":
        leader.set_rollout_weight_version.remote.side_effect = RuntimeError(
            "stamp failed"
        )
        expected = "stamp failed"
    else:
        leader.resume_generation_async.remote.side_effect = None
        leader.resume_generation_async.remote.return_value = False
        expected = "resume every"
    with pytest.raises(RuntimeError, match=expected):
        asyncio.run(ctrl._sync_weights())
    assert state["paused"]
    assert not ctrl._rollout_permitted.is_set()
    replacement.resume_generation_async.remote.assert_not_called()
    ctrl._rollout_manager.resume_request_deadlines.assert_not_called()


@pytest.mark.parametrize(
    "synchronizer_cls,send,receive",
    [
        (NcclReshardWeightSynchronizer, "nccl_reshard_refit", "nccl_reshard_refit"),
        (
            CollectiveWeightSynchronizer,
            "broadcast_weights_for_collective",
            "update_weights_from_collective",
        ),
    ],
)
def test_transport_uses_paused_actors_even_if_fleet_replaces_them(
    monkeypatch, synchronizer_cls, send, receive
):
    ctrl, _, _ = _controller(monkeypatch, synchronizer_cls=synchronizer_cls)
    generation = ctrl._gen
    targets = generation.capture_refit_targets()
    replacement = MagicMock()
    generation.get_collective_sender_spec = MagicMock(
        return_value=SimpleNamespace(buffer_size_bytes=1024, num_buffers=2)
    )
    policy = MagicMock()

    def enqueue(**kwargs):
        generation.worker_group.workers[0] = replacement
        return []

    getattr(policy, send).side_effect = enqueue
    sync = object.__new__(synchronizer_cls)
    sync._generation = generation
    sync._policy = policy
    sync._refit_timeout_s = None
    sync.sync_weights(generation_targets=targets)
    worker_method = (
        "nccl_reshard_refit_async"
        if receive == "nccl_reshard_refit"
        else "update_weights_from_collective_async"
    )
    getattr(targets.workers[0], worker_method).remote.assert_called_once()
    getattr(replacement, worker_method).remote.assert_not_called()


def test_refit_membership_includes_stale_and_excludes_mid_transfer_restart(monkeypatch):
    ctrl, _, _ = _controller(monkeypatch)
    leader = ctrl._gen.worker_group.workers[0]
    dead, stale = MagicMock(), MagicMock()
    for method in [
        "pause_generation_async",
        "set_rollout_weight_version",
        "resume_generation_async",
    ]:
        getattr(stale, method).remote.side_effect = getattr(
            leader, method
        ).remote.side_effect
    ctrl._gen.dp_size = 3
    ctrl._gen.worker_group.workers = [
        leader,
        MagicMock(),
        dead,
        MagicMock(),
        stale,
        MagicMock(),
    ]
    ctrl._gen._refit_membership = plan_refit_membership(
        surviving_shards=[0, 2], dp_size=3, total_gen_workers=6, train_world_size=1
    )
    fleet = GenerationFleetHealth(shard_count=3, policy=FleetHealthPolicy())
    fleet.record_actor_death(1)
    fleet.record_actor_death(2)
    fleet.mark_loaded(2)
    ctrl._gen_fleet = fleet
    ctrl._reconcile_refit_membership = AsyncMock()
    original = ctrl._weight_synchronizer.sync_weights

    def refit(**kwargs):
        assert kwargs["generation_targets"].shard_indices == (0, 2)
        original(**kwargs)
        fleet.mark_loaded(1)

    ctrl._weight_synchronizer.sync_weights = refit
    asyncio.run(ctrl._sync_weights())
    assert fleet.state_of(2) is ShardState.HEALTHY
    assert fleet.state_of(1) is ShardState.STALE
    for worker in [leader, stale]:
        worker.set_rollout_weight_version.remote.assert_called_once_with(version=1)
        worker.resume_generation_async.remote.assert_called_once()
    dead.pause_generation_async.remote.assert_not_called()
    dead.set_rollout_weight_version.remote.assert_not_called()
    dead.resume_generation_async.remote.assert_not_called()


def test_transfer_retry_keeps_survivors_paused_and_uses_rebuilt_membership(monkeypatch):
    ctrl, state, calls = _controller(monkeypatch)
    dead = MagicMock()
    dead.pause_generation_async.remote.return_value = True
    ctrl._gen.dp_size = 2
    ctrl._gen.worker_group.workers.append(dead)
    attempts = []

    def refit(**kwargs):
        assert state["paused"]
        attempts.append(kwargs["generation_targets"].shard_indices)
        calls.append("refit")
        if len(attempts) == 1:
            raise RefitAborted("participant lost")

    async def recover(_failure):
        assert state["paused"]
        calls.append("recover")
        ctrl._gen._refit_membership = plan_refit_membership(
            surviving_shards=[0], dp_size=2, total_gen_workers=2, train_world_size=1
        )

    ctrl._weight_synchronizer.sync_weights = refit
    ctrl._recover_from_failed_refit = recover
    ctrl._recovery_window = nullcontext
    ctrl._record_refit_landed = MagicMock()
    asyncio.run(ctrl._sync_weights())
    assert attempts == [(0, 1), (0,)]
    assert calls == [
        ("pause", False),
        "refit",
        "recover",
        ("pause", False),
        "refit",
        "stamp",
        "resume",
    ]
    ctrl._record_refit_landed.assert_called_once_with({0})
    dead.pause_generation_async.remote.assert_called_once()
    dead.set_rollout_weight_version.remote.assert_not_called()
    dead.resume_generation_async.remote.assert_not_called()
