# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from collections import deque

from nemo_rl.models.generation.replica_metrics import EngineMetrics
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.utils.logger import TensorboardLogger


def _worker():
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"enable_vllm_metrics_logger": True}}
    worker._vllm_metrics_lock = threading.Lock()
    worker._replica_metrics = EngineMetrics()
    worker.inflight_batch_sizes = deque(maxlen=4096)
    worker.num_pending_samples = deque(maxlen=4096)
    worker.kv_cache_usage_perc = deque(maxlen=4096)
    worker.generation_tokens = deque(maxlen=4096)
    return worker


def test_worker_snapshot_reduction_drain_and_legacy_output():
    worker = _worker()
    worker._record_vllm_metric_snapshot(
        [
            ("vllm:num_requests_running", 2),
            ("vllm:num_requests_running", 3),
            ("vllm:num_requests_waiting", 4),
            ("vllm:kv_cache_usage_perc", 0.75),
            ("vllm:generation_tokens", 100),
        ]
    )
    legacy = worker.get_vllm_logger_metrics()
    assert legacy["inflight_batch_sizes"] == [2, 3]
    result = worker.drain_replica_metrics()
    assert result["requests_running_mean"] == 5
    assert result["requests_waiting_max"] == 4
    assert result["kv_cache_usage_fraction"] == 0.75
    assert worker.drain_replica_metrics() == {"engine_samples": 0}
    worker._record_vllm_metric_snapshot([("vllm:num_requests_running", 1)])
    assert worker.drain_replica_metrics()["requests_running_max"] == 1
    # Legacy consumers retain their own get/clear lifecycle.
    assert worker.get_vllm_logger_metrics()["inflight_batch_sizes"] == [2, 3, 1]
    worker.clear_vllm_logger_metrics()
    assert worker.get_vllm_logger_metrics()["inflight_batch_sizes"] == []
    assert legacy["inflight_batch_sizes"] == [2, 3]


def test_disabled_worker_metrics_are_empty():
    worker = _worker()
    worker.cfg["vllm_cfg"]["enable_vllm_metrics_logger"] = False
    assert worker.drain_replica_metrics() == {}


def test_worker_metrics_are_saved_as_per_replica_tensorboard_scalars(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    worker = _worker()
    worker._record_vllm_metric_snapshot([("vllm:num_requests_waiting", 7)])
    logger = TensorboardLogger({}, log_dir=str(tmp_path))
    logger.log_metrics(
        {
            "vllm/replica_2/" + key: value
            for key, value in worker.drain_replica_metrics().items()
        },
        step=3,
        prefix="train",
    )
    logger.writer.flush()
    events = EventAccumulator(str(tmp_path)).Reload()
    scalar = events.Scalars("train/vllm/replica_2/requests_waiting_max")
    assert [(item.step, item.value) for item in scalar] == [(3, 7)]
    logger.writer.close()
