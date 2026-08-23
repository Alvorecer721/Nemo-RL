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

## 2026-08-23 16:37 CEST

- Real-topology retry `3163157` failed after 4:08 before model initialization. The source overlay changed each actor's uv command root, so the worker builder correctly rejected the baked fingerprint and attempted an in-place update of the image-owned venv. On several nodes uv collided with lower-layer package directories (`flash_attn/layers`, `transformers/quantizers`, and `deep_ep/backend`: `File exists`). This is a launch-layer overlay failure, not a recurrence of the checkpoint failure.
- Historical job `3148504` is now correlated more tightly. The missing global ranks were `9,39,57,73,83,117,162,211,215,219,237,273,277,282,284,285`, spread across fourteen nodes; therefore one failed Slurm node cannot explain the loss. Five of sixteen final-PP-stage writers failed versus eleven of 272 other ranks.
- Slurm accounting reports the historical head step `OUT_OF_MEMORY`. The worker step's maximum RSS was 261.33 GiB on `nid006625` (`172.28.32.212`), the node whose rank 273 writer is missing. That node had already written roughly 162 GB of its four final-stage rank files; its missing fourth file was about 52 GB. Combined with the old Ray object-store allocation of about 135 GiB per node and other process overhead, the 450-GiB node cgroup had insufficient checkpoint-staging headroom.
- The exact hang mechanism is source-confirmed: NVRx persistent save performs D2H staging inside its child process while the training parent blocks on `preload_q.join()`. If that child is OOM-killed, it never calls `task_done()` and the parent waits forever. The historical driver never progressed beyond `policy.save_checkpoint`, matching the 16 absent files and lack of a Python traceback.
- The rerun now uses an empty source-head-scoped actor venv, retaining the warm image uv cache without mutating baked packages. It captures per-node `memory.events`, `memory.peak`, cgroup identity and process state once checkpointing begins and again on a bounded stall, so the next run can directly attribute any lost writer instead of inferring it from aggregate accounting.
