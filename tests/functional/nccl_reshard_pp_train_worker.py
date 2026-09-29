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
"""Test-only full Bridge export, independent of the local-view refit path."""

from pathlib import Path
from typing import Any

import ray
import torch
from safetensors.torch import save_file

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from tests.functional.nccl_reshard_pp_workers import (
    inspect_megatron_destination,
    mutate_policy,
)


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class PipelineTrainingRefitMegatronWorker(MegatronPolicyWorkerImpl):
    def mutate_refit_weights(self) -> int:
        return mutate_policy(self)

    def inspect_refit_weights(self) -> dict[str, Any]:
        return inspect_megatron_destination(self)

    def export_refit_reference(self, directory: str) -> dict[str, int]:
        """All source ranks gather; only rank zero writes bounded CPU shards.

        This intentionally uses Bridge's full HF conversion instead of the
        local-view mappings under test. It runs outside the refit timer.
        """
        writer = torch.distributed.get_rank() == 0
        pending = {}
        pending_bytes = 0
        shard = 0
        count = 0
        names = set()
        for name, tensor in self.megatron_bridge.export_hf_weights(
            self.model,
            cpu=False,
            show_progress=False,
        ):
            assert name not in names, f"Duplicate HF reference tensor: {name}"
            names.add(name)
            count += 1
            if not writer:
                continue
            # Copy before advancing the exporter, which may reuse its buffers.
            pending[name] = tensor.detach().to("cpu", copy=True).contiguous()
            pending_bytes += tensor.numel() * tensor.element_size()
            if pending_bytes >= 512 * 1024**2:
                save_file(
                    pending, Path(directory) / f"reference-{shard:05d}.safetensors"
                )
                pending.clear()
                pending_bytes = 0
                shard += 1
        if writer and pending:
            save_file(pending, Path(directory) / f"reference-{shard:05d}.safetensors")
            shard += 1
        torch.distributed.barrier()
        return {"tensors": count, "written_shards": shard}
