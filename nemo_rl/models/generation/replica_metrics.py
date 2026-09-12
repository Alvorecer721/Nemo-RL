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

"""Constant-memory, interval metrics for native generation replicas.

Call counts describe controller requests, independently of the routing policy.
Engine statistics describe sampled vLLM queues. Neither changes scheduling.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Summary:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0
    last: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        self.last = value

    def metrics(self, name: str) -> dict[str, float]:
        if not self.count:
            return {}
        return {
            name: self.last,
            f"{name}_mean": self.total / self.count,
            f"{name}_max": self.maximum,
        }


class CallMetrics:
    """Thread-safe call accounting; draining does not reset live requests."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._begin = self._last = clock()
        self._inflight = self._peak = self._started = 0
        self._completed = self._interrupted = 0
        self._area = 0.0
        self._durations = _Summary()
        self._last_dispatch: float | None = None
        self._last_finish: float | None = None

    def __getstate__(self) -> dict[str, Callable[[], float]]:
        """Transfer idle accounting to Ray without process-local state.

        The generation handle moves from the driver to the controller before
        rollout starts. Locks and monotonic windows belong to the receiving
        process; transferring an active call would orphan its completion.
        """
        with self._lock:
            if self._inflight:
                raise RuntimeError("Cannot transfer metrics with in-flight calls")
            return {"clock": self._clock}

    def __setstate__(self, state: dict[str, Callable[[], float]]) -> None:
        self.__init__(clock=state["clock"])

    def _advance(self, now: float) -> None:
        self._area += self._inflight * (now - self._last)
        self._last = now

    def start(self) -> float:
        """Record dispatch and return a monotonic start time for finish()."""
        with self._lock:
            now = self._clock()
            self._advance(now)
            self._inflight += 1
            self._started += 1
            self._peak = max(self._peak, self._inflight)
            self._last_dispatch = time.time()
            return now

    def finish(self, started_at: float, *, completed: bool) -> None:
        """Record closure, including failure, cancellation and generator close."""
        with self._lock:
            now = self._clock()
            self._advance(now)
            self._inflight -= 1
            self._completed += int(completed)
            self._interrupted += int(not completed)
            self._durations.add(now - started_at)
            self._last_finish = time.time()

    def drain(self) -> dict[str, float]:
        """Return counts and time-weighted occupancy since the preceding drain."""
        with self._lock:
            now = self._clock()
            self._advance(now)
            elapsed = now - self._begin
            result = {
                "calls_inflight": float(self._inflight),
                "calls_inflight_max": float(self._peak),
                "calls_inflight_mean": self._area / elapsed
                if elapsed
                else float(self._inflight),
                "calls_started": float(self._started),
                "calls_completed": float(self._completed),
                "calls_interrupted": float(self._interrupted),
                "call_window_s": elapsed,
                **self._durations.metrics("call_duration_s"),
            }
            if self._last_dispatch is not None:
                result["last_dispatch_unix_s"] = self._last_dispatch
            if self._last_finish is not None:
                result["last_finish_unix_s"] = self._last_finish
            self._begin = now
            self._area = 0.0
            self._peak = self._inflight
            self._started = self._completed = self._interrupted = 0
            self._durations = _Summary()
            return result


class EngineMetrics:
    """Sample summaries, protected by the worker's existing metrics lock.

    Gauge means are sample means. Token rates use actual monotonic elapsed time,
    including idle periods, and retain a counter baseline across drains. Missing
    metrics and counter resets never masquerade as zero throughput.
    """

    def __init__(self) -> None:
        self._gauges: dict[str, _Summary] = {}
        self._samples = self._idle = self._queue_samples = 0
        self._token_delta = self._token_seconds = 0.0
        self._counter_resets = 0
        self._previous_tokens: tuple[float, float] | None = None
        self._sample_time = 0.0

    def record(
        self,
        *,
        now: float,
        wall_time: float,
        running: float | None = None,
        waiting: float | None = None,
        kv: float | None = None,
        tokens: float | None = None,
    ) -> None:
        """Record one engine snapshot. KV usage is a fraction in [0, 1]."""
        if all(value is None for value in (running, waiting, kv, tokens)):
            return
        self._samples += 1
        self._sample_time = wall_time
        for name, value in (
            ("requests_running", running),
            ("requests_waiting", waiting),
            ("kv_cache_usage_fraction", kv),
        ):
            if value is not None:
                self._gauges.setdefault(name, _Summary()).add(value)
        if running is not None and waiting is not None:
            self._queue_samples += 1
            self._idle += int(running == 0 and waiting == 0)
        if tokens is not None:
            if self._previous_tokens is not None:
                previous_time, previous_count = self._previous_tokens
                if tokens < previous_count:
                    self._counter_resets += 1
                elif now > previous_time:
                    self._token_delta += tokens - previous_count
                    self._token_seconds += now - previous_time
            self._previous_tokens = (now, tokens)

    def drain(self) -> dict[str, float]:
        """Consume summaries atomically under the caller's metrics lock."""
        result = {"engine_samples": float(self._samples)}
        if self._samples:
            result["sample_time_unix_s"] = self._sample_time
            result["counter_resets"] = float(self._counter_resets)
        for name, summary in self._gauges.items():
            result.update(summary.metrics(name))
        if self._queue_samples:
            result["idle_sample_fraction"] = self._idle / self._queue_samples
        if self._token_seconds:
            result["generated_tokens"] = self._token_delta
            result["generation_tokens_per_s"] = self._token_delta / self._token_seconds
            result["token_window_s"] = self._token_seconds
        self._gauges.clear()
        self._samples = self._idle = self._queue_samples = self._counter_resets = 0
        self._token_delta = self._token_seconds = 0.0
        return result
