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
"""Independent CPU checkpoint oracle for BF16 dense and MoE refits."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


def digest(tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def verify_snapshots(
    snapshots: list[dict[str, Any]],
    model_path: str,
    updates: int,
    *,
    reference_workers: int = 1,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Check a partition of layers and raw-storage ranks against the checkpoint.

    Every shard receives all snapshots so owner and off-stage checks remain
    global. Running every index in ``range(shard_count)`` covers every layer
    and every rank exactly once; the default checks everything locally.
    """
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Expected 0 <= shard_index < shard_count and shard_count > 0")
    file_for_key = {}
    for path in Path(model_path).glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            file_for_key.update({key: path for key in f.keys()})
    plan = snapshots[0]["plan"]

    def load_tensor(name: str) -> torch.Tensor:
        # GLM's routing correction is an FP32 buffer in Megatron. Parameter-only
        # mutations leave it unchanged, and the misc transport preserves FP32.
        correction_bias = name.endswith(".gate.e_score_correction_bias")
        with safe_open(file_for_key[name], framework="pt", device="cpu") as f:
            # Materialize before slicing; strided mmap reads can be very slow.
            value = f.get_tensor(name).to(
                dtype=torch.float32 if correction_bias else torch.bfloat16, copy=True
            )
        for _ in range(0 if correction_bias else updates):
            value.mul_(1.015625)
        return value

    def load_experts(prefix: str, projection: str, ids: list[int]) -> torch.Tensor:
        return torch.stack(
            [load_tensor(f"{prefix}.{expert}.{projection}.weight") for expert in ids]
        )

    def check_layer(layer: str) -> tuple[int, int]:
        count = 0
        logical_bytes = 0
        for param in plan["per_layer_params"][layer]:
            name = param["name"]
            projection = param.get("grouped_expert_proj")
            if projection is not None:
                prefix = name.removesuffix(f".{projection}.weight")
                reference = load_experts(
                    prefix, projection, list(range(param["global_shape"][0]))
                )
            else:
                reference = load_tensor(name)
            mesh = np.array(param["dst_mesh_info"]["mesh"])
            logical_bytes += reference.numel() * reference.element_size()
            owners_seen = set()
            for snap in snapshots:
                position = np.argwhere(mesh == snap["rank"])
                if position.size == 0:
                    assert name not in snap["params"], (
                        name,
                        snap["rank"],
                        "off-stage storage",
                    )
                    continue
                value = reference
                for axis, placement in enumerate(param["dst_placements"]):
                    if "dim" in placement:
                        value = value.chunk(mesh.shape[axis], dim=placement["dim"])[
                            int(position[0, axis])
                        ]
                assert snap["params"][name] == digest(value), (
                    name,
                    snap["rank"],
                    updates,
                    snap["params"][name],
                    digest(value),
                )
                owners_seen.add(snap["rank"])
                count += 1
            assert owners_seen == set(mesh.flatten()), (name, owners_seen, mesh)
        return count, logical_bytes

    count = 0
    logical_bytes = 0
    # Independent checkpoint reads can overlap. Each task owns its tensors and
    # returns only counts, keeping memory bounded by the configured worker count.
    layers = plan["layer_names"][shard_index::shard_count]
    with ThreadPoolExecutor(max_workers=reference_workers) as executor:
        for layer_index, (shards, size) in enumerate(
            executor.map(check_layer, layers), start=1
        ):
            count += shards
            logical_bytes += size
            if layer_index % 16 == 0:
                print(
                    f"REFERENCE_BULK_CHECK update={updates} shard={shard_index}/{shard_count} layers={layer_index}/{len(layers)}",
                    flush=True,
                )

    # Independent checkpoint oracle for vLLM storage, including the misc path and
    # both tied vocabulary aliases. Do not use the production refit mappings here.
    def check_all_parameters(snap: dict[str, Any]) -> int:
        all_count = 0
        for name, actual in snap.get("all_parameters", {}).items():
            name = snap.get("parameter_aliases", {}).get(name, name)
            if name.endswith((".W_UK_T", ".W_UV")):
                # These persistent derived matrices participate in MLA decode.
                # Checking just kv_b_proj would miss a stale post-load cache.
                # vLLM also exposes these directly on self_attn. Strip the
                # parameter name before removing any nested MLA wrapper.
                prefix = name.rsplit(".", 1)[0].split(".mla_attn", 1)[0]
                value = load_tensor(f"{prefix}.kv_b_proj.weight")
                value = value.chunk(snap["tp_size"], dim=0)[snap["tp_rank"]]
                mla = snap["mla_config"]
                value = value.reshape(
                    -1,
                    mla["qk_nope_head_dim"] + mla["v_head_dim"],
                    mla["kv_lora_rank"],
                )
                expected = (
                    value[:, : mla["qk_nope_head_dim"], :]
                    if name.endswith(".W_UK_T")
                    else value[:, mla["qk_nope_head_dim"] :, :].transpose(1, 2)
                )
                assert actual == digest(expected), (name, snap["rank"], updates)
                all_count += 1
                continue
            if name.endswith((".w13_weight", ".w2_weight")):
                storage_prefix = name.rsplit(".", 1)[0]
                prefix = storage_prefix.removesuffix(".routed_experts")
                ids = snap["expert_ids"][storage_prefix]
                projections = (
                    ("gate_proj", "up_proj")
                    if name.endswith(".w13_weight")
                    else ("down_proj",)
                )
                pieces = []
                for projection in projections:
                    value = load_experts(prefix, projection, ids)
                    dim = 2 if projection == "down_proj" else 1
                    pieces.append(
                        value.chunk(snap["etp_size"], dim=dim)[snap["etp_rank"]]
                    )
                expected = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
                assert actual == digest(expected), (name, snap["rank"], updates)
                all_count += 1
                continue
            names = [name]
            shard_dim = None
            if ".fused_qkv_a_proj." in name:
                # MLA down projections are fused but replicated across TP.
                names = [
                    name.replace("fused_qkv_a_proj", p)
                    for p in ("q_a_proj", "kv_a_proj_with_mqa")
                ]
            elif ".indexer.wk_weights_proj." in name:
                names = [
                    name.replace("wk_weights_proj", p) for p in ("wk", "weights_proj")
                ]
            elif name.endswith((".q_b_proj.weight", ".kv_b_proj.weight")):
                shard_dim = 0
            elif ".qkv_proj." in name:
                names = [
                    name.replace("qkv_proj", p) for p in ("q_proj", "k_proj", "v_proj")
                ]
                shard_dim = 0
            elif ".gate_up_proj." in name:
                names = [
                    name.replace("gate_up_proj", p) for p in ("gate_proj", "up_proj")
                ]
                shard_dim = 0
            elif name.endswith(("o_proj.weight", "down_proj.weight")):
                shard_dim = 1
            elif name.endswith("up_proj.weight"):
                shard_dim = 0
            elif name.endswith(("embed_tokens.weight", "lm_head.weight")):
                shard_dim = 0
                if name == "lm_head.weight" and name not in file_for_key:
                    names = ["model.embed_tokens.weight"]
            pieces = []
            for key in names:
                reference = load_tensor(key)
                if shard_dim is not None:
                    reference = reference.chunk(snap["tp_size"], dim=shard_dim)[
                        snap["tp_rank"]
                    ]
                pieces.append(reference)
            expected = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
            if ".indexer.k_norm." in name:
                # vLLM's LayerNorm stores these in FP32. The source parameters
                # and their controlled updates are BF16, so cast after updates.
                expected = expected.float()
            assert actual == digest(expected), (
                name,
                snap["rank"],
                updates,
                actual,
                digest(expected),
            )
            all_count += 1
            if all_count % 128 == 0:
                print(
                    f"REFERENCE_ALL_CHECK update={updates} parameters={all_count}",
                    flush=True,
                )
        return all_count

    with ThreadPoolExecutor(max_workers=reference_workers) as executor:
        all_count = sum(
            executor.map(check_all_parameters, snapshots[shard_index::shard_count])
        )
    return {
        "verified_local_shards": count,
        "logical_bulk_bytes": logical_bytes,
        "verified_vllm_parameters": all_count,
    }
