# Timeline

## 2026-08-23 14:01 CEST

- Created a clean current-main GLM production worktree.
- Re-read historical job `3148504`: 272/288 shards landed in under a minute, 16 ranks never completed, no metadata was written, and completed ranks waited in the post-save barrier until manual cancellation.
- Verified NVRx v0.6 persistent finalization blocks without checking writer-process liveness. The exact child failure is not preserved.
- Added explicit `fully_parallel_save` and load-integrity passthrough, plus an optional per-node Ray object-store cap.
- Staged matched one-node synchronous and NVRx asynchronous save / fresh-allocation resume controls using the certified TOML and image-owned `/opt/ray_venvs`.

## 2026-08-23 14:10 CEST

- Jobs `3161932` and `3161933` failed in the harness preflight before Ray or model startup: the driver Python was incorrectly given raw Bridge/MCore source paths and therefore lacked worker-only Transformer Engine dependencies.
- Cancelled dependent resume jobs `3161934` and `3161935`. The correction keeps the driver on the NeMo-RL source overlay and runs Megatron-specific preflight through the baked Megatron worker Python.

## 2026-08-23 15:48 CEST

- Current-source controls closed the small-topology question: jobs `3162831` and `3162855` each wrote two complete 4/4-rank optimizer checkpoints, about 103 GB per checkpoint, with synchronous and NVRx persistence respectively.
- Fresh-allocation jobs `3162832` and `3162856` both loaded `step_1` and completed the next training step. Their Slurm wrappers exited red only because the test harness attempted a permission-preserving cross-filesystem `mv` and later expected Megatron iteration 1 although the outer NeMo-RL step cursor was 2.
- Recovered the historical GLM contract and replaced its custom dataset adapter, raw squashfs launch, split Slurm steps and nested actor venv with the built-in `DAPOMath17K` loader, `docker/nemo_rl_vllm0251.toml`, single-step Ray and image-owned `/opt/ray_venvs`.
- Added the real 80-node Phase-A/Phase-B harness. It preserves the proven 288-rank TP1/PP18/EP16 training plus TP32/EP32 rollout topology, caps Ray's object store at 64 GiB per node, requests 850000M host memory, disables fully-parallel save, and fails after 20 minutes without shard-count progress while retaining diagnostics.
- Focused config tests passed 3/3 with the repository-wide Ray autouse fixture intentionally bypassed; the ordinary unit harness failed before test collection because Ray could not resolve the current container hostname.
