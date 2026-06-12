#!/usr/bin/env python
"""Export a Megatron torch-dist checkpoint (NeMo-RL step_XXXX/policy/weights or
raw Megatron-LM) to a HuggingFace checkpoint via the Megatron-Bridge mapping.

The export direction of the bridge was certified bit-exact (387/387 tensors,
TP1 and TP2) on 2026-06-13; this CLI is the production wrapper around the same
code path. Single GPU, any source parallel geometry (torch-dist re-shards).

Usage (from a NeMo-RL runtime dir, e.g. the mlm-restore worktree):
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

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.hf_base, trust_remote_code=True)
    models = bridge.load_megatron_model(args.megatron_ckpt, wrap_with_ddp=False)
    bridge.save_hf_pretrained(models, str(out), show_progress=True)

    tok_src = Path(args.tokenizer) if args.tokenizer else Path(args.hf_base)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json", "chat_template.jinja"):
        src = tok_src / name
        if src.exists():
            shutil.copy2(src, out / name)
    print(f"exported -> {out}")


if __name__ == "__main__":
    main()
