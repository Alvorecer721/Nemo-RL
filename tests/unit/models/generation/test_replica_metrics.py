# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from ray import cloudpickle

from nemo_rl.models.generation.replica_metrics import CallMetrics, EngineMetrics


def test_call_metrics_transfer_starts_a_local_interval():
    calls = CallMetrics()
    calls.finish(calls.start(), completed=True)
    restored = cloudpickle.loads(cloudpickle.dumps(calls))
    # Ray transfers the generation handle to the controller before rollout.
    # Process-local locks and monotonic timestamps must not cross that boundary.
    assert restored.drain()["calls_completed"] == 0
    restored.finish(restored.start(), completed=True)
    assert restored.drain()["calls_completed"] == 1
    assert calls.drain()["calls_completed"] == 1


def test_call_metrics_cannot_transfer_unfinished_requests():
    calls = CallMetrics()
    started = calls.start()
    with pytest.raises(RuntimeError, match="in-flight"):
        cloudpickle.dumps(calls)
    calls.finish(started, completed=True)
    assert calls.drain()["calls_completed"] == 1


def test_call_window_keeps_live_calls_across_drains():
    ticks = [0.0]
    calls = CallMetrics(clock=lambda: ticks[0])
    ticks[0] = 2.0
    first = calls.start()
    ticks[0] = 4.0
    second = calls.start()
    ticks[0] = 6.0
    calls.finish(first, completed=True)
    ticks[0] = 8.0
    metrics = calls.drain()
    # 0*2 + 1*2 + 2*2 + 1*2 = 8 call-seconds / 8 seconds.
    assert metrics["calls_inflight_mean"] == 1
    assert metrics["calls_inflight"] == 1
    assert metrics["calls_inflight_max"] == 2
    assert metrics["calls_started"] == 2
    assert metrics["calls_completed"] == 1
    assert metrics["call_duration_s_mean"] == 4
    ticks[0] = 10.0
    calls.finish(second, completed=False)
    metrics = calls.drain()
    assert metrics["calls_started"] == 0
    assert metrics["calls_inflight_mean"] == 1
    assert metrics["calls_inflight"] == 0
    assert metrics["calls_interrupted"] == 1
    assert metrics["call_duration_s_max"] == 6


def test_engine_counter_rate_uses_elapsed_time_and_survives_drain():
    metrics = EngineMetrics()
    metrics.record(now=1, wall_time=101, running=4, waiting=2, kv=0.5, tokens=100)
    metrics.record(now=3, wall_time=103, running=0, waiting=0, kv=0.1, tokens=180)
    result = metrics.drain()
    assert result["requests_running_mean"] == 2
    assert result["requests_running_max"] == 4
    assert result["requests_waiting_mean"] == 1
    assert result["kv_cache_usage_fraction_max"] == 0.5
    assert result["idle_sample_fraction"] == 0.5
    assert result["generated_tokens"] == 80
    assert result["generation_tokens_per_s"] == 40
    assert result["sample_time_unix_s"] == 103
    assert metrics.drain() == {"engine_samples": 0}
    metrics.record(now=7, wall_time=107, running=1, waiting=0, kv=0.2, tokens=200)
    assert metrics.drain()["generation_tokens_per_s"] == 5


def test_counter_reset_and_missing_metrics_are_not_fake_zero_throughput():
    metrics = EngineMetrics()
    metrics.record(now=1, wall_time=101, tokens=100)
    metrics.record(now=2, wall_time=102, tokens=10)
    result = metrics.drain()
    assert result["counter_resets"] == 1
    assert "generation_tokens_per_s" not in result
    assert "requests_running_mean" not in result
    metrics.record(now=4, wall_time=104, tokens=30)
    assert metrics.drain()["generation_tokens_per_s"] == 10


def test_empty_engine_snapshot_is_missing_not_idle():
    metrics = EngineMetrics()
    metrics.record(now=1, wall_time=101)
    assert metrics.drain() == {"engine_samples": 0}


def test_many_engine_samples_are_reduced_without_losing_early_peaks():
    metrics = EngineMetrics()
    metrics.record(now=0, wall_time=100, running=100)
    for i in range(1, 10_000):
        metrics.record(now=i, wall_time=100 + i, running=0)
    result = metrics.drain()
    assert result["engine_samples"] == 10_000
    assert result["requests_running_mean"] == pytest.approx(0.01)
    assert result["requests_running_max"] == 100
