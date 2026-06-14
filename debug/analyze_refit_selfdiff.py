#!/usr/bin/env python3
"""Analyze pre/post refit weight dumps from the vLLM worker self-diff.

Usage: python analyze_refit_selfdiff.py <dump_dir> [hf_ref.pt]

Pairs pre.{uuid}.pt / post.{uuid}.pt, reports every tensor whose sha256
changed, and characterizes the corruption pattern for tensors with full
copies (permutation / shift / zeros / partial rows). Optionally compares
both pre and post against the HF disk reference for layers 0/1.
"""

import glob
import os
import sys

import torch


def load_pairs(dump_dir):
    pres = sorted(glob.glob(os.path.join(dump_dir, "pre.*.pt")))
    pairs = []
    for pre_path in pres:
        uuid = os.path.basename(pre_path)[len("pre.") : -len(".pt")]
        post_path = os.path.join(dump_dir, f"post.{uuid}.pt")
        if os.path.exists(post_path):
            pairs.append((uuid, pre_path, post_path))
        else:
            print(f"!! missing post dump for {uuid}")
    return pairs


def as_u16(t):
    """bf16 -> uint16 view for exact multiset comparison."""
    return t.contiguous().view(torch.uint8).view(-1)


def characterize(name, pre, post):
    out = []
    if pre.shape != post.shape or pre.dtype != post.dtype:
        out.append(f"  SHAPE/DTYPE CHANGED: {pre.shape}/{pre.dtype} -> {post.shape}/{post.dtype}")
        return out

    pf = pre.float()
    qf = post.float()
    neq = (pre.view(torch.uint8) != post.view(torch.uint8)).any(dim=-1) if False else None

    diff_mask = ~torch.eq(pf, qf) | (torch.isnan(pf) ^ torch.isnan(qf))
    n_diff = int(diff_mask.sum())
    n_tot = pf.numel()
    out.append(f"  changed elements: {n_diff}/{n_tot} ({100.0 * n_diff / n_tot:.2f}%)")
    out.append(
        f"  pre  stats: min={pf.min():.6g} max={pf.max():.6g} mean={pf.mean():.6g} nan={int(torch.isnan(pf).sum())}"
    )
    out.append(
        f"  post stats: min={qf.min():.6g} max={qf.max():.6g} mean={qf.mean():.6g} nan={int(torch.isnan(qf).sum())}"
    )
    out.append(f"  max |delta| over changed: {(pf - qf).abs().max():.6g}")

    # zeros / garbage check
    if int((qf == 0).sum()) > 0.9 * n_tot:
        out.append("  PATTERN: post is (mostly) ZEROS -> wrong/absent source write")

    # multiset (permutation) check on raw bf16 bit patterns
    a = as_u16(pre)
    b = as_u16(post)
    perm = torch.equal(a.sort().values, b.sort().values)
    if perm and n_diff > 0:
        out.append("  PATTERN: EQUAL MULTISET of raw values -> PERMUTATION")

    # row structure for 2D
    if pre.dim() == 2 and n_diff > 0:
        row_changed = diff_mask.any(dim=1)
        rows = torch.nonzero(row_changed).flatten().tolist()
        # contiguous ranges
        ranges = []
        for r in rows:
            if ranges and r == ranges[-1][1] + 1:
                ranges[-1][1] = r
            else:
                ranges.append([r, r])
        out.append(
            f"  changed rows: {int(row_changed.sum())}/{pre.shape[0]}; ranges (first 12): "
            + ", ".join(f"[{a0}:{b0 + 1}]" for a0, b0 in ranges[:12])
        )
        col_changed = diff_mask.any(dim=0)
        out.append(f"  changed cols: {int(col_changed.sum())}/{pre.shape[1]}")

        # per-row permutation check on first changed row
        if rows:
            r0 = rows[0]
            ra, rb = as_u16(pre[r0]), as_u16(post[r0])
            if torch.equal(ra.sort().values, rb.sort().values):
                out.append(f"  row {r0}: equal multiset -> intra-row permutation")
            # row-swap check: does post row r0 equal some other pre row?
            cand = torch.nonzero((pre == post[r0]).all(dim=1)).flatten().tolist()
            if cand:
                out.append(f"  post row {r0} == pre row(s) {cand[:8]} -> ROW PERMUTATION")

    # shift detection on flat element view
    if n_diff > 0:
        fa = pf.flatten()
        fb = qf.flatten()
        i0 = int(torch.nonzero(fa != fb).flatten()[0])
        i1 = int(torch.nonzero(fa != fb).flatten()[-1])
        out.append(f"  first/last differing flat index: {i0} / {i1}")
        # try element shifts up to 4096
        probe = fa[i0 : i0 + 256]
        win_lo = max(0, i0 - 4096)
        win = fb[win_lo : i0 + 4096 + 256]
        for k in range(win.numel() - 256 + 1):
            if torch.equal(win[k : k + 256], probe):
                out.append(
                    f"  PATTERN: pre[{i0}:{i0}+256] found in post at flat offset {win_lo + k} "
                    f"(shift {win_lo + k - i0} elements) -> STREAM OFFSET MISALIGNMENT"
                )
                break
    return out


def main():
    dump_dir = sys.argv[1]
    ref = torch.load(sys.argv[2], weights_only=True) if len(sys.argv) > 2 else None

    for uuid, pre_path, post_path in load_pairs(dump_dir):
        print(f"\n{'=' * 80}\nENGINE {uuid}\n{'=' * 80}")
        pre = torch.load(pre_path, weights_only=True)
        post = torch.load(post_path, weights_only=True)

        names_pre = set(pre["meta"])
        names_post = set(post["meta"])
        for n in sorted(names_pre - names_post):
            print(f"  only in pre: {n}")
        for n in sorted(names_post - names_pre):
            print(f"  only in post: {n}")

        changed = []
        for n in sorted(names_pre & names_post):
            if pre["meta"][n] != post["meta"][n]:
                changed.append(n)

        print(f"\ntensors total: {len(names_pre)}, CHANGED: {len(changed)}")
        for n in changed:
            sp, dp, hp = pre["meta"][n]
            sq, dq, hq = post["meta"][n]
            print(f"\nCHANGED {n}  shape={sp} dtype={dp}")
            print(f"  sha pre ={hp[:16]}…  sha post={hq[:16]}…")
            if n in pre["full"] and n in post["full"]:
                for line in characterize(n, pre["full"][n], post["full"][n]):
                    print(line)

        # disk-reference comparison for full-copy tensors
        if ref is not None:
            print(f"\n--- disk-reference check (engine {uuid[:16]}…) ---")
            for n in sorted(pre["full"]):
                hf = vllm_name_to_hf_ref(n, ref)
                if hf is None:
                    print(f"  (no HF mapping) {n}")
                    continue
                pr = pre["full"][n]
                po = post["full"][n]
                ok_pre = pr.shape == hf.shape and torch.equal(pr, hf.to(pr.dtype))
                ok_post = po.shape == hf.shape and torch.equal(po, hf.to(po.dtype))
                flag = "" if (ok_pre and ok_post) else "   <-- MISMATCH"
                print(f"  {n}: pre==disk {ok_pre}, post==disk {ok_post}{flag}")


def vllm_name_to_hf_ref(name, ref):
    """Map a vLLM param name to the equivalent HF-checkpoint tensor."""
    name = name.removeprefix("BUFFER:")
    if name.endswith("self_attn.qkv_proj.weight"):
        base = name.replace("qkv_proj.weight", "")
        try:
            q = ref[base + "q_proj.weight"]
            k = ref[base + "k_proj.weight"]
            v = ref[base + "v_proj.weight"]
        except KeyError:
            return None
        return torch.cat([q, k, v], dim=0)
    if name == "model.embed_tokens.weight[:1024]":
        return ref.get("model.embed_tokens.weight")
    if name == "lm_head.weight[:1024]":
        return ref.get("lm_head.weight")
    if name in ref:
        return ref[name]
    # buffers saved as 0-dim in HF, 0-dim in vllm too
    return ref.get(name)


if __name__ == "__main__":
    main()
