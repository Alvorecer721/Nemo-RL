#!/usr/bin/env python3
"""Compare POST-refit vLLM engine weights against the HF disk checkpoint.

Walks the HF safetensors once, computes sha256 for every tensor in the
vLLM param layout (fusing q/k/v -> qkv_proj rows like QKVParallelLinear
with TP1), and checks them against the sha256 recorded in each engine's
post.*.pt dump. This is a FULL 451-tensor verification, not just the
layer-0/1 subset.

Usage: python compare_post_vs_disk.py <dump_dir> <hf_ckpt_dir>
"""

import glob
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open


def sha(t: torch.Tensor) -> str:
    raw = t.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main():
    dump_dir, ckpt = sys.argv[1], sys.argv[2]

    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    wmap = index["weight_map"]

    # Load every HF tensor (16 GB, CPU) lazily per file, compute vLLM-layout shas.
    handles = {}

    def get(name):
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(ckpt, f), framework="pt")
        return handles[f].get_tensor(name)

    expected: dict[str, str] = {}
    done_qkv = set()
    for name in sorted(wmap):
        if ".self_attn.q_proj.weight" in name:
            base = name.replace("q_proj.weight", "")
            fused = torch.cat(
                [get(base + "q_proj.weight"), get(base + "k_proj.weight"), get(base + "v_proj.weight")],
                dim=0,
            )
            expected[base + "qkv_proj.weight"] = sha(fused)
            done_qkv.add(base)
        elif ".self_attn.k_proj.weight" in name or ".self_attn.v_proj.weight" in name:
            continue
        elif name.endswith(".act_fn.beta") or name.endswith(".act_fn.eps"):
            expected["BUFFER:" + name] = sha(get(name))
        else:
            expected[name] = sha(get(name))

    posts = sorted(glob.glob(os.path.join(dump_dir, "post.*.pt")))
    for post_path in posts:
        uuid = os.path.basename(post_path)[len("post.") : -len(".pt")]
        post = torch.load(post_path, weights_only=True)
        meta = post["meta"]

        n_match = n_mismatch = n_skipped = 0
        mismatches = []
        missing_in_engine = []
        for name, exp_sha in expected.items():
            if name not in meta:
                missing_in_engine.append(name)
                continue
            got_sha = meta[name][2]
            if got_sha == exp_sha:
                n_match += 1
            else:
                n_mismatch += 1
                mismatches.append(name)
        engine_only = [
            n
            for n in meta
            if n not in expected
            and "rotary_emb" not in n
            and "_scale" not in n
        ]

        print(f"\n=== ENGINE {uuid} ===")
        print(f"  disk tensors checked: {len(expected)}")
        print(f"  MATCH: {n_match}   MISMATCH: {n_mismatch}   missing-in-engine: {len(missing_in_engine)}")
        for n in mismatches:
            print(f"    MISMATCH: {n}  engine_sha={meta[n][2][:16]}… disk_sha={expected[n][:16]}…")
        for n in missing_in_engine:
            print(f"    MISSING IN ENGINE: {n}")
        if engine_only:
            print(f"  engine-only tensors (no disk counterpart, excl. rope/scales): {engine_only}")


if __name__ == "__main__":
    main()
