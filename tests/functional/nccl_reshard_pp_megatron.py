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
"""Test-only Megatron actor RPCs for changed-weight lifecycle validation."""

from typing import Any

import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from tests.functional.nccl_reshard_pp_workers import (
    inspect_megatron_destination,
    mutate_policy,
)


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class PipelineRefitMegatronWorker(MegatronPolicyWorkerImpl):
    def mutate_refit_weights(self) -> int:
        return mutate_policy(self)

    def inspect_refit_weights(self) -> dict[str, Any]:
        return inspect_megatron_destination(self)

    def get_unreplayed_logprobs(
        self, *, data: BatchedDataDict[Any]
    ) -> BatchedDataDict[Any]:
        """Paired control with identical weights and ordinary Megatron routing."""
        return self.get_logprobs(data=data, require_router_replay=False)
