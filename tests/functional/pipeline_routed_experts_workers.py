# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only observation of the routes actually captured on every PP stage."""

from collections.abc import Callable
from typing import Any

import ray

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


def make_route_observer() -> Callable[[Any], dict[str, Any]]:
    # Nest the RPC callable so cloudpickle sends it by value. The engine's
    # workers need vLLM, but do not need to import this test driver's module.
    def install_route_observer(worker: Any) -> dict[str, Any]:
        import socket

        from vllm.distributed import get_pp_group, get_tp_group
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            RoutedExpertsCaptureSource,
        )

        runner = worker.model_runner
        layers = sorted(
            {
                module.layer_id
                for module in runner.model.modules()
                if isinstance(module, (MoERunner, RoutedExpertsCaptureSource))
            }
        )
        assert layers, "Each tested stage must have an actual MoE router"
        assert runner.routed_experts_capturer is not None
        original = runner.execute_model
        records = []

        def observe(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(scheduler_output, *args, **kwargs)
            count = scheduler_output.total_num_scheduled_tokens
            if count:
                if worker.use_v2_model_runner:
                    token_ids = runner.input_buffers.input_ids[:count]
                    positions = runner.input_buffers.positions[:count]
                else:
                    token_ids = runner.input_ids.gpu[:count]
                    positions = runner.positions[:count]
                # Snapshot locally computed layers before GPUWorker can modify
                # or send the aggregate. CPU copies intentionally synchronize
                # this correctness-only experiment, including graph replay.
                routes = runner.routed_experts_capturer.get_device_buffer()[
                    :count, layers
                ]
                records.append(
                    {
                        "token_ids": token_ids.cpu().tolist(),
                        "positions": positions.cpu().tolist(),
                        "routes": routes.cpu().tolist(),
                    }
                )
            return result

        runner.execute_model = observe
        worker._test_route_observation = {
            "hostname": socket.gethostname(),
            "pp_rank": get_pp_group().rank_in_group,
            "tp_rank": get_tp_group().rank_in_group,
            "runner_v2": worker.use_v2_model_runner,
            "layers": layers,
            "records": records,
        }
        return {
            k: v for k, v in worker._test_route_observation.items() if k != "records"
        }

    return install_route_observer


@ray.remote
class PipelineRoutesWorker(VllmAsyncGenerationWorkerImpl):
    async def install_route_observer(self) -> list[dict[str, Any]]:
        return await self.llm.collective_rpc(make_route_observer(), args=())

    async def read_route_observations(self) -> list[dict[str, Any]]:
        def read_route_observations(worker: Any) -> dict[str, Any]:
            return worker._test_route_observation

        return await self.llm.collective_rpc(read_route_observations, args=())
