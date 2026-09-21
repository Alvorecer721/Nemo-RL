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
"""The runtime oracle must reject wrong expert ownership and fused ordering."""

import copy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests.functional.nccl_reshard_pp_reference import digest, verify_snapshots


def test_megatron_inspection_reads_persistent_experts():
    """A receive hook allocates staging, so it cannot inspect committed weights."""
    # The checkpoint-only tests do not need the heavy Megatron worker imports.
    from tests.functional.nccl_reshard_pp_workers import inspect_megatron_destination

    prefix = "model.layers.0.mlp.experts"
    name = f"{prefix}.gate_proj.weight"
    tensors = {
        expert: torch.full((4, 3), expert, dtype=torch.bfloat16) for expert in (10, 2)
    }

    def receive_staging(_base):
        raise AssertionError("Inspection must not allocate receive staging")

    tasks = []
    for expert, tensor in tensors.items():
        spec = SimpleNamespace(
            name=f"{prefix}.{expert}.gate_proj.weight",
            select=lambda value: value.chunk(2, dim=0)[0],
        )
        tasks.append(
            SimpleNamespace(
                destination=tensor,
                is_mxfp8=False,
                conversion_task=SimpleNamespace(
                    local_hf_param_specs=lambda spec=spec: (spec,)
                ),
            )
        )
    worker = SimpleNamespace(
        _build_generation_refit_tasks=lambda: ([], tasks),
        hf_to_local_param_map=SimpleNamespace(
            specs={name: SimpleNamespace(base=None, pre=receive_staging)}
        ),
        _generation_nccl_reshard_groups={0: SimpleNamespace(rank=4)},
        nccl_reshard_refit_info={"layer_names": [], "per_layer_params": {}},
    )
    expected = torch.stack([tensors[e][:2] for e in (2, 10)])
    assert inspect_megatron_destination(worker)["params"] == {name: digest(expected)}
    tensors[2].add_(1)
    expected = torch.stack([tensors[e][:2] for e in (2, 10)])
    assert inspect_megatron_destination(worker)["params"] == {name: digest(expected)}


def test_reference_checks_shared_parameter_aliases(tmp_path):
    """vLLM registers a Qwen MoE gate under both the block and fused module."""
    name = "model.layers.0.mlp.gate.weight"
    alias = "model.layers.0.mlp.experts.gate.weight"
    tensor = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)
    save_file({name: tensor}, tmp_path / "model.safetensors")
    snapshot = {
        "rank": 0,
        "plan": {"layer_names": [], "per_layer_params": {}},
        "params": {},
        "tp_rank": 0,
        "tp_size": 2,
        "parameter_aliases": {alias: name},
        "all_parameters": {name: digest(tensor), alias: digest(tensor)},
    }
    result = verify_snapshots([snapshot], str(tmp_path), 0)
    assert result["verified_vllm_parameters"] == 2
    snapshot["all_parameters"][alias] = digest(tensor + 1)
    with pytest.raises(AssertionError):
        verify_snapshots([snapshot], str(tmp_path), 0)


@pytest.mark.parametrize("shard_count", [2, 4])
def test_reference_shards_preserve_complete_coverage(tmp_path, shard_count):
    """Distributing the oracle must neither omit nor duplicate any comparison."""
    layers = [f"model.layers.{i}" for i in range(7)]
    hf = {
        f"{layer}.mlp.down_proj.weight": torch.full((2, 3), i, dtype=torch.bfloat16)
        for i, layer in enumerate(layers)
    }
    save_file(hf, tmp_path / "model.safetensors")
    plan = {
        "layer_names": layers,
        "per_layer_params": {
            layer: [
                {
                    "name": f"{layer}.mlp.down_proj.weight",
                    "dst_mesh_info": {"mesh": [i % 3]},
                    "dst_placements": [{}],
                }
            ]
            for i, layer in enumerate(layers)
        },
    }
    snapshots = [
        {
            "rank": rank,
            "plan": plan,
            "tp_rank": 0,
            "tp_size": 1,
            "params": {
                name: digest(value)
                for i, (name, value) in enumerate(hf.items())
                if i % 3 == rank
            },
        }
        for rank in range(3)
    ]
    for snapshot in snapshots:
        snapshot["all_parameters"] = copy.deepcopy(snapshot["params"])
    full = verify_snapshots(snapshots, str(tmp_path), 0)
    parts = [
        verify_snapshots(
            snapshots, str(tmp_path), 0, shard_index=index, shard_count=shard_count
        )
        for index in range(shard_count)
    ]
    assert {key: sum(part[key] for part in parts) for key in full} == full
    assert full == {
        "verified_local_shards": 7,
        "logical_bulk_bytes": 84,
        "verified_vllm_parameters": 7,
    }
    # The last layer and its owning rank are assigned independently. Both the
    # bulk check and the independent raw-storage check must still reject it.
    for field in ("params", "all_parameters"):
        corrupted = copy.deepcopy(snapshots)
        corrupted[0][field][f"{layers[-1]}.mlp.down_proj.weight"]["sha256"] = "stale"
        with pytest.raises(AssertionError):
            for index in range(shard_count):
                verify_snapshots(
                    corrupted,
                    str(tmp_path),
                    0,
                    shard_index=index,
                    shard_count=shard_count,
                )


@pytest.mark.parametrize("updates", [0, 2])
@pytest.mark.parametrize("mla_wrapper", ["", ".mla_attn", ".mla_attn.mla_attn"])
def test_glm_reference_checks_mla_and_indexer_storage(tmp_path, updates, mla_wrapper):
    """GLM fuses replicated down projections and caches transposed MLA weights."""
    prefix = "model.layers.3.self_attn"
    shapes = {
        "q_a_proj.weight": (6, 4),
        "kv_a_proj_with_mqa.weight": (5, 4),
        "q_b_proj.weight": (24, 6),
        "kv_b_proj.weight": (20, 3),
        "indexer.wq_b.weight": (8, 6),
        "indexer.wk.weight": (4, 4),
        "indexer.weights_proj.weight": (2, 4),
        "indexer.k_norm.weight": (4,),
        "indexer.k_norm.bias": (4,),
    }
    hf = {
        f"{prefix}.{name}": (
            torch.arange(torch.tensor(shape).prod()).reshape(shape) + i * 100
        ).to(torch.bfloat16)
        for i, (name, shape) in enumerate(shapes.items())
    }
    bias_name = "model.layers.3.mlp.gate.e_score_correction_bias"
    hf[bias_name] = torch.tensor([0.10001, 0.20002], dtype=torch.float32)
    # Updates round in Megatron's BF16 parameter storage before vLLM's
    # LayerNorm loader casts to FP32. Multiplying in FP32 would be different.
    for name in ("weight", "bias"):
        hf[f"{prefix}.indexer.k_norm.{name}"] = torch.tensor(
            [0.1, 0.3, 0.7, 1.1], dtype=torch.bfloat16
        )
    save_file(hf, tmp_path / "model.safetensors")
    changed = {name: value.to(torch.bfloat16).clone() for name, value in hf.items()}
    changed[bias_name] = hf[bias_name].clone()
    for name, value in changed.items():
        # Megatron stores the correction bias as a buffer, not a parameter.
        if name != bias_name:
            for _ in range(updates):
                value.mul_(1.015625)
    snapshots = []
    for tp_rank in range(2):
        kv = changed[f"{prefix}.kv_b_proj.weight"].chunk(2, dim=0)[tp_rank]
        per_head = kv.reshape(2, 5, 3)
        local = {
            f"{prefix}.fused_qkv_a_proj.weight": torch.cat(
                [
                    changed[f"{prefix}.q_a_proj.weight"],
                    changed[f"{prefix}.kv_a_proj_with_mqa.weight"],
                ]
            ),
            f"{prefix}.q_b_proj.weight": changed[f"{prefix}.q_b_proj.weight"].chunk(
                2, dim=0
            )[tp_rank],
            f"{prefix}.kv_b_proj.weight": kv,
            f"{prefix}.indexer.wq_b.weight": changed[f"{prefix}.indexer.wq_b.weight"],
            f"{prefix}.indexer.wk_weights_proj.weight": torch.cat(
                [
                    changed[f"{prefix}.indexer.wk.weight"],
                    changed[f"{prefix}.indexer.weights_proj.weight"],
                ]
            ),
            f"{prefix}.indexer.k_norm.weight": changed[
                f"{prefix}.indexer.k_norm.weight"
            ].float(),
            f"{prefix}.indexer.k_norm.bias": changed[
                f"{prefix}.indexer.k_norm.bias"
            ].float(),
            f"{prefix}{mla_wrapper}.W_UK_T": per_head[:, :2, :],
            f"{prefix}{mla_wrapper}.W_UV": per_head[:, 2:, :].transpose(1, 2),
            bias_name: changed[bias_name].float(),
        }
        snapshots.append(
            {
                "rank": tp_rank,
                "plan": {"layer_names": [], "per_layer_params": {}},
                "params": {},
                "tp_rank": tp_rank,
                "tp_size": 2,
                "mla_config": {
                    "kv_lora_rank": 3,
                    "qk_nope_head_dim": 2,
                    "v_head_dim": 3,
                },
                "all_parameters": {
                    name: digest(value) for name, value in local.items()
                },
            }
        )
    result = verify_snapshots(snapshots, str(tmp_path), updates)
    assert result["verified_vllm_parameters"] == 20
    for name in (
        f"{prefix}.fused_qkv_a_proj.weight",
        f"{prefix}{mla_wrapper}.W_UV",
        f"{prefix}.indexer.wk_weights_proj.weight",
        f"{prefix}.indexer.k_norm.weight",
        f"{prefix}.indexer.k_norm.bias",
        bias_name,
    ):
        corrupted = copy.deepcopy(snapshots)
        corrupted[0]["all_parameters"][name]["sha256"] = "stale"
        with pytest.raises(AssertionError):
            verify_snapshots(corrupted, str(tmp_path), updates)

    if updates:
        corrupted = copy.deepcopy(snapshots)
        name = f"{prefix}.indexer.k_norm.weight"
        wrong_rounding = hf[name].float()
        for _ in range(updates):
            wrong_rounding.mul_(1.015625)
        corrupted[0]["all_parameters"][name] = digest(wrong_rounding)
        with pytest.raises(AssertionError):
            verify_snapshots(corrupted, str(tmp_path), updates)


@pytest.mark.parametrize("ep,etp", [(2, 1), (1, 2)])
def test_grouped_expert_reference(tmp_path, ep, etp):
    prefix = "model.layers.0.mlp.experts"
    hf = {}
    for expert in range(12):
        for projection, offset, shape in (
            ("gate_proj", 0, (6, 4)),
            ("up_proj", 1000, (6, 4)),
            ("down_proj", 2000, (4, 6)),
        ):
            hf[f"{prefix}.{expert}.{projection}.weight"] = (
                torch.arange(24).reshape(shape) + offset + 100 * expert
            ).to(torch.bfloat16)
    save_file(hf, tmp_path / "model.safetensors")
    mesh = [[4 + e * etp + t for t in range(etp)] for e in range(ep)]
    plan = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {"model.layers.0": []},
    }
    for projection in ("gate_proj", "up_proj", "down_proj"):
        plan["per_layer_params"]["model.layers.0"].append(
            {
                "name": f"{prefix}.{projection}.weight",
                "grouped_expert_proj": projection,
                "global_shape": [12, 4, 6] if projection == "down_proj" else [12, 6, 4],
                "dst_mesh_info": {"mesh": mesh},
                "dst_placements": [
                    {"dim": 0},
                    {"dim": 2 if projection == "down_proj" else 1},
                ],
            }
        )
    snapshots = []
    for e in range(ep):
        ids = list(range(e * (12 // ep), (e + 1) * (12 // ep)))
        for t in range(etp):
            local = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                pieces = []
                for expert in ids:
                    value = hf[f"{prefix}.{expert}.{projection}.weight"].clone()
                    value.mul_(1.015625)
                    pieces.append(
                        value.chunk(etp, dim=1 if projection == "down_proj" else 0)[t]
                    )
                local[projection] = torch.stack(pieces)
            storage_prefix = prefix + ".routed_experts"
            snapshots.append(
                {
                    "rank": mesh[e][t],
                    "plan": plan,
                    "params": {
                        f"{prefix}.{p}.weight": digest(v) for p, v in local.items()
                    },
                    "tp_rank": t,
                    "tp_size": etp,
                    "expert_ids": {storage_prefix: ids},
                    "etp_rank": t,
                    "etp_size": etp,
                    "all_parameters": {
                        f"{storage_prefix}.w13_weight": digest(
                            torch.cat([local["gate_proj"], local["up_proj"]], dim=1)
                        ),
                        f"{storage_prefix}.w2_weight": digest(local["down_proj"]),
                    },
                }
            )
    snapshots.append({"rank": 8, "plan": plan, "params": {}})
    result = verify_snapshots(snapshots, str(tmp_path), 1, reference_workers=2)
    assert result["verified_local_shards"] == ep * etp * 3
    assert result["verified_vllm_parameters"] == ep * etp * 2

    wrong_expert_order = copy.deepcopy(snapshots)
    wrong_expert_order[0]["expert_ids"][storage_prefix].reverse()
    with pytest.raises(AssertionError):
        verify_snapshots(wrong_expert_order, str(tmp_path), 1)

    stale_update = copy.deepcopy(snapshots)
    stale_update[0]["params"][f"{prefix}.gate_proj.weight"]["sha256"] = "stale"
    with pytest.raises(AssertionError):
        verify_snapshots(stale_update, str(tmp_path), 1)
