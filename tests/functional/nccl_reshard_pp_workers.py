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
"""Test-only inspection and deterministic mutations, outside timed refits."""

import re
from typing import Any

import torch

from nemo_rl.weight_sync.nccl_reshard_utils import (
    make_nccl_reshard_refit_info_wire_safe,
)
from tests.functional.nccl_reshard_pp_reference import digest


def mutate_policy(worker: Any) -> int:
    # Megatron is available only inside the policy worker environment.
    from megatron.core.utils import unwrap_model

    count = 0
    models = worker.model if isinstance(worker.model, (list, tuple)) else [worker.model]
    with torch.no_grad():
        for model in models:
            for param in unwrap_model(model).parameters():
                # Round once per update in the BF16 wire dtype, including FP64
                # router parameters, so the checkpoint oracle remains exact.
                param.copy_(param.to(torch.bfloat16).mul_(1.015625))
                count += param.numel()
    torch.cuda.synchronize()
    return count


def inspect_megatron_destination(worker: Any) -> dict[str, Any]:
    # Receive hooks can allocate fresh staging. Inspect the persistent model
    # through Bridge's local views instead, including each committed expert.
    _models, tasks = worker._build_generation_refit_tasks()
    names = set(worker.hf_to_local_param_map.specs)
    params = {}
    experts: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for task in tasks:
        for spec in task.conversion_task.local_hf_param_specs():
            if spec.name in names:
                assert not task.is_mxfp8, "The storage oracle requires BF16 weights"
                params[spec.name] = digest(spec.select(task.destination))
                continue
            match = re.fullmatch(
                r"(.*\.experts)\.(\d+)\.((?:gate|up|down)_proj)\.weight", spec.name
            )
            if match is None:
                continue
            grouped_name = f"{match[1]}.{match[3]}.weight"
            if grouped_name in names:
                assert not task.is_mxfp8, "The storage oracle requires BF16 weights"
                experts.setdefault(grouped_name, []).append(
                    (int(match[2]), spec.select(task.destination))
                )
    for name, pieces in experts.items():
        params[name] = digest(torch.stack([value for _, value in sorted(pieces)]))
    assert set(params) == names, (names - params.keys(), params.keys() - names)
    return {
        "rank": worker._generation_nccl_reshard_groups[0].rank,
        "params": params,
        "plan": make_nccl_reshard_refit_info_wire_safe(worker.nccl_reshard_refit_info),
    }


def inspect_vllm_destination(worker: Any) -> dict[str, Any]:
    # vLLM is available only inside its actor environment.
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    info = worker.nccl_reshard_refit_info
    mapping = worker._build_hf_to_gen_backend_mapping(info)
    # Read actual fused storage; receive hooks may allocate uninitialized staging.
    params = {
        name: digest(value if region is None else value[region])
        for name, (value, region) in mapping.items()
    }
    expert_ids = {}
    etp_rank, etp_size = 0, 1
    for name, module in worker.model_runner.model.named_modules():
        if not isinstance(module, RoutedExperts):
            continue
        assert module.quant_method.unquantized_backend.value == "TRITON", (
            "The raw BF16 expert oracle requires the Triton storage layout"
        )
        mapping = module.expert_map
        if mapping is None:
            ids = list(range(module.global_num_experts))
        else:
            pairs = sorted(
                (local, global_id)
                for global_id, local in enumerate(mapping.tolist())
                if local >= 0
            )
            assert [local for local, _ in pairs] == list(
                range(module.local_num_experts)
            )
            ids = [global_id for _, global_id in pairs]
        expert_ids[name] = ids
        parallel = module.moe_config.moe_parallel_config
        etp_rank, etp_size = parallel.tp_rank, parallel.tp_size
    parameters = list(
        worker.model_runner.model.named_parameters(remove_duplicate=False)
    )
    first_name = {}
    aliases = {}
    for name, value in parameters:
        canonical = first_name.setdefault(id(value), name)
        if canonical != name:
            aliases[name] = canonical
    config = worker.model_config.hf_config
    mla_config = (
        {
            name: getattr(config, name)
            for name in ("kv_lora_rank", "qk_nope_head_dim", "v_head_dim")
        }
        if config.model_type == "glm_moe_dsa"
        else None
    )
    return {
        "rank": worker.pp_comm_groups[0].rank,
        "params": params,
        "plan": make_nccl_reshard_refit_info_wire_safe(info),
        "tp_rank": get_tp_group().rank_in_group,
        "tp_size": get_tp_group().world_size,
        "expert_ids": expert_ids,
        "etp_rank": etp_rank,
        "etp_size": etp_size,
        "parameter_aliases": aliases,
        "mla_config": mla_config,
        "all_parameters": {name: digest(value) for name, value in parameters},
    }
