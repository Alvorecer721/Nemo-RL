# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a Megatron DPO/SFT checkpoint to HF safetensors using an SFT-base overlay.

The stock ``bridge.save_hf_pretrained`` drops every weight tensor because the
Apertus xIELU buffers ``mlp.act_fn.{beta,eps}`` (constants in Megatron, buffers
in HF) are never yielded by the bridge generator and the safetensors save path
discards any shard that's missing tensors.

This converter takes a different route:
1. Load the SFT base HF state_dict into CPU memory (xIELU beta/eps come from
   here — they're constants and never change during training).
2. Stream the Megatron checkpoint via ``stream_weights_megatron_to_hf`` and
   overwrite the matching keys in the SFT-base state_dict.
3. Save the resulting state_dict as standard HF safetensors (sharded), along
   with config / tokenizer / generation_config from the SFT base, plus the
   chat_template / tokenizer files from the canonical tokenizer dir.

The output dir is loadable by vLLM exactly like the SFT base.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# nemo_rl must be imported BEFORE any megatron.* to extend sys.path with
# 3rdparty/Megatron-LM-workspace/Megatron-LM (megatron.training lives there;
# megatron-bridge's training submodule needs it).
import nemo_rl  # noqa: F401

import torch
import yaml
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file


def load_sft_base_state(sft_base: Path) -> dict[str, torch.Tensor]:
    idx_path = sft_base / "model.safetensors.index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"missing {idx_path}")
    with open(idx_path) as f:
        idx = json.load(f)
    files = sorted(set(idx["weight_map"].values()))
    state = {}
    for fn in files:
        print(f"  [overlay] reading {fn}", flush=True)
        state.update(load_file(str(sft_base / fn)))
    print(f"  [overlay] loaded {len(state)} tensors from SFT base", flush=True)
    return state


def stream_megatron_weights(config_yaml: Path, megatron_ckpt: Path):
    """Yield (name, tensor) pairs from the Megatron checkpoint via Megatron-Bridge."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion import model_bridge
    from megatron.bridge.training.model_load_save import temporary_distributed_context
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)
    hf_model_name = cfg["policy"]["model_name"]
    hf_overrides = cfg["policy"].get("hf_overrides", {}) or {}

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **hf_overrides
    )
    with temporary_distributed_context(backend="gloo"):
        model_parallel_cuda_manual_seed(0)
        megatron_model = bridge.load_megatron_model(
            str(megatron_ckpt), skip_temp_dist_context=True
        )
        dispatch_instance = (
            bridge._causal_lm_architecture,
            bridge._get_model_instance(megatron_model),
        )
        for name, tensor in model_bridge.stream_weights_megatron_to_hf(
            dispatch_instance,
            megatron_model,
            bridge.hf_pretrained,
            cpu=True,
            show_progress=True,
            merge_adapter_weights=True,
        ):
            yield name, tensor.contiguous().cpu()

    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="ckpt/config.yaml")
    p.add_argument(
        "--megatron-ckpt-path", type=Path, required=True, help="ckpt/policy/weights/iter_NNNN"
    )
    p.add_argument(
        "--sft-base-hf",
        type=Path,
        default=Path(
            "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"
        ),
        help="HF dir of the SFT base — used as the buffer/template overlay.",
    )
    p.add_argument("--hf-ckpt-path", type=Path, required=True, help="Output dir.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.hf_ckpt_path.exists() and not args.overwrite:
        # If output already has safetensors, skip.
        if any(args.hf_ckpt_path.glob("*.safetensors")):
            print(f"[skip] {args.hf_ckpt_path} already has safetensors")
            return 0
    args.hf_ckpt_path.mkdir(parents=True, exist_ok=True)

    print(f"[stage 1] loading SFT-base state from {args.sft_base_hf}", flush=True)
    state = load_sft_base_state(args.sft_base_hf)
    sft_keys = set(state.keys())

    print(f"[stage 2] streaming Megatron weights from {args.megatron_ckpt_path}", flush=True)
    overwritten = 0
    missing_in_sft = []
    for name, tensor in stream_megatron_weights(args.config, args.megatron_ckpt_path):
        if name in state:
            # Match dtype + shape; raise if shape mismatched.
            if state[name].shape != tensor.shape:
                raise RuntimeError(
                    f"shape mismatch for {name}: sft {state[name].shape} vs megatron {tensor.shape}"
                )
            state[name] = tensor.to(dtype=state[name].dtype)
            overwritten += 1
        else:
            missing_in_sft.append(name)
    print(f"[stage 2] overwrote {overwritten} / {len(sft_keys)} tensors from Megatron stream", flush=True)
    if missing_in_sft:
        print(
            f"[warn] {len(missing_in_sft)} bridge tensors not present in SFT base "
            f"(first 5: {missing_in_sft[:5]})",
            flush=True,
        )

    untouched = [k for k in sft_keys if "act_fn.beta" in k or "act_fn.eps" in k]
    print(f"[stage 2] xIELU buffers preserved from SFT base: {len(untouched)} keys", flush=True)

    print(f"[stage 3] writing sharded safetensors → {args.hf_ckpt_path}", flush=True)
    # Ensure all tensors are contiguous on CPU before sharding
    state = {k: v.contiguous().cpu() for k, v in state.items()}
    plan = split_torch_state_dict_into_shards(state)
    for filename, tensor_names in plan.filename_to_tensors.items():
        shard = {k: state[k] for k in tensor_names}
        out = args.hf_ckpt_path / filename
        print(f"  [shard] {filename}  tensors={len(shard)}", flush=True)
        save_file(shard, str(out), metadata={"format": "pt"})

    if plan.is_sharded:
        index = {"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}
        with open(args.hf_ckpt_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    print("[stage 4] copying config/tokenizer/template artifacts from SFT base", flush=True)
    for fn in (
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "audio_token_mapping.json",
        "vision_token_mapping.json",
    ):
        src = args.sft_base_hf / fn
        if src.exists():
            shutil.copy2(src, args.hf_ckpt_path / fn)

    print("[done]", flush=True)
    print(f"  output dir: {args.hf_ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
