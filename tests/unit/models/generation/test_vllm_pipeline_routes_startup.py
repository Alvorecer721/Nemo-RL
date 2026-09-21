# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node-local source preparation must finish before nested vLLM workers start."""

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm import vllm_generation


@pytest.mark.parametrize("caller_deferred", [False, True])
def test_every_actor_finishes_source_preparation_before_engine_start(
    monkeypatch, caller_deferred
):
    prepared = [False, False]
    engines = []

    def start_engine():
        assert all(prepared), (
            "Engine started before every node installed its source patch"
        )
        engines.append("loaded")

    class WorkerGroup:
        dp_size = 1

        def __init__(self, cluster, builder, **kwargs):
            if not builder.kwargs.get("defer_model_load"):
                prepared[0] = True
                start_engine()

        def run_all_workers_single_data(self, method, **kwargs):
            if method == "is_alive":
                assert not kwargs.get("run_rank_0_only_axes"), (
                    "Readiness must include nonleaders"
                )
                return [
                    lambda rank=rank: prepared.__setitem__(rank, True) or True
                    for rank in range(2)
                ]
            if method == "load_model":
                return [start_engine]
            raise AssertionError(method)

    monkeypatch.setattr(
        vllm_generation, "RayWorkerBuilder", lambda *a, **kw: SimpleNamespace(kwargs=kw)
    )
    monkeypatch.setattr(vllm_generation, "RayWorkerGroup", WorkerGroup)
    monkeypatch.setattr(
        vllm_generation.ray, "get", lambda futures: [f() for f in futures]
    )
    cls = vllm_generation.VllmGeneration
    monkeypatch.setattr(
        cls, "_get_tied_worker_bundle_indices", lambda *_: [(0, [0]), (1, [0])]
    )
    monkeypatch.setattr(cls, "_post_init", lambda _: None)
    monkeypatch.setattr(cls, "_collect_reserved_urls", lambda _: [])
    monkeypatch.setattr(cls, "_report_dp_openai_server_base_urls", lambda _: [])
    monkeypatch.setattr(cls, "_report_device_id", lambda _: [])
    config = {key: None for key in vllm_generation.VllmConfig.__required_keys__}
    config.update(
        model_name="model",
        top_k=None,
        top_p=1.0,
        colocated={"enabled": False},
        vllm_cfg={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 2,
            "expert_parallel_size": 1,
            "async_engine": True,
        },
        vllm_kwargs={"enable_return_routed_experts": True},
    )
    cluster = SimpleNamespace(
        world_size=lambda: 2,
        num_gpus_per_node=1,
        _init_placement_groups=lambda **_: None,
    )
    generation = cls(cluster, config, defer_model_load=caller_deferred)
    if caller_deferred:
        assert engines == []
        generation.load_and_start()
    assert engines == ["loaded"]
    assert prepared == [True, True]
    assert generation._defer_model_load is True


def test_recovery_waits_for_every_recreated_actor(monkeypatch):
    prepared = [False, False]
    events = []

    def start():
        assert all(prepared), "Recovery started before remote source preparation"
        events.append("loaded")

    workers = [
        SimpleNamespace(
            is_alive=SimpleNamespace(
                remote=lambda rank=rank: lambda: prepared.__setitem__(rank, True)
            ),
            load_model=SimpleNamespace(remote=lambda: start),
            post_init_async=SimpleNamespace(remote=lambda: lambda: None),
            report_dp_openai_server_base_url=SimpleNamespace(
                remote=lambda: lambda: "http://new"
            ),
        )
        for rank in range(2)
    ]
    generation = vllm_generation.VllmGeneration.__new__(vllm_generation.VllmGeneration)
    generation.worker_group = SimpleNamespace(
        workers=workers,
        get_dp_leader_worker_idx=lambda _: 0,
        recreate_worker=lambda _: None,
    )
    generation.model_parallel_size = 2
    generation._defer_model_load = True
    generation.cfg = {"vllm_cfg": {"async_engine": True}}
    generation.dp_openai_server_base_urls = ["http://old"]
    monkeypatch.setattr(
        vllm_generation.ray,
        "get",
        lambda refs: refs() if callable(refs) else [r() for r in refs],
    )
    assert generation.restart_shard(0) == "http://new"
    assert events == ["loaded"]
