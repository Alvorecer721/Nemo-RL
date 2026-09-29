# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM-only actor extension for inspecting real destination storage."""

from typing import Any

import ray

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from tests.functional.nccl_reshard_pp_workers import inspect_vllm_destination


@ray.remote
class PipelineRefitVllmWorker(VllmAsyncGenerationWorkerImpl):
    async def inspect_refit_weights(self) -> list[dict[str, Any]]:
        return await self.llm.collective_rpc(inspect_vllm_destination, args=())
