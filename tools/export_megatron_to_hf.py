#!/usr/bin/env python
"""Export a Megatron torch-dist checkpoint (NeMo-RL step_XXXX/policy/weights or
raw Megatron-LM) to a HuggingFace checkpoint via the Megatron-Bridge mapping.

The export direction of the bridge was certified bit-exact (387/387 tensors,
TP1 and TP2) on 2026-06-13; this CLI is the production wrapper around the same
code path. Single GPU, any source parallel geometry (torch-dist re-shards).

Usage (from the NeMo-RL repo root):
  uv run --locked python tools/export_megatron_to_hf.py \
      --hf-base /capstor/.../hf_checkpoints/<base model>  \
      --megatron-ckpt /path/to/step_1921/policy/weights   \
      --out /path/to/output_hf_dir                        \
      [--tokenizer /path/to/tokenizer]   # copied into the output

--hf-base supplies the architecture/config (and default tokenizer files); the
weights come exclusively from --megatron-ckpt.
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-base", required=True, help="HF checkpoint defining the architecture/config")
    p.add_argument("--megatron-ckpt", required=True, help="torch-dist weights dir (contains iter_*/ or is an iter dir)")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=None, help="tokenizer dir to copy into the output (default: from --hf-base)")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)

    import torch

    # load_megatron_model requires initialized distributed + parallel state
    # even single-process (TP1/PP1); torch-dist re-shards any source geometry.
    torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(42)

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.hf_base, trust_remote_code=True)
    models = bridge.load_megatron_model(args.megatron_ckpt, wrap_with_ddp=False)

    # The bridge exports trainable parameters only; non-trainable buffers
    # (Apertus xIELU beta/eps) come from the base checkpoint — the same
    # constants-from-disk rule as the vLLM refit fix. save_hf_pretrained's
    # shard writer refuses incomplete shards, so we assemble and write the
    # full 451-tensor state dict ourselves as a single safetensors file.
    import json
    from safetensors import safe_open
    from safetensors.torch import save_file

    state = {name: t.detach().to("cpu", torch.bfloat16).contiguous()
             for name, t in bridge.export_hf_weights(models, show_progress=True)}
    base = Path(args.hf_base)
    with open(base / "model.safetensors.index.json") as f:
        index = json.load(f)["weight_map"]
    missing = [k for k in index if k not in state]
    for name in missing:
        with safe_open(base / index[name], framework="pt") as f:
            state[name] = f.get_tensor(name)
    extra = [k for k in state if k not in index]
    assert not extra, f"exported keys absent from base index: {extra[:5]}"
    assert set(state) == set(index), "tensor set mismatch vs base checkpoint"
    print(f"exported {len(state) - len(missing)} params + {len(missing)} buffers from base")
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})

    tok_src = Path(args.tokenizer) if args.tokenizer else base
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json", "chat_template.jinja"):
        src = (base if name in ("config.json", "generation_config.json") else tok_src) / name
        if src.exists():
            shutil.copy2(src, out / name)
    print(f"exported -> {out}")


if __name__ == "__main__":
    main()
