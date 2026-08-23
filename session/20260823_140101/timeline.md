# Timeline

## 2026-08-23 14:01 CEST

- Created a clean current-main GLM production worktree.
- Re-read historical job `3148504`: 272/288 shards landed in under a minute, 16 ranks never completed, no metadata was written, and completed ranks waited in the post-save barrier until manual cancellation.
- Verified NVRx v0.6 persistent finalization blocks without checking writer-process liveness. The exact child failure is not preserved.
- Added explicit `fully_parallel_save` and load-integrity passthrough, plus an optional per-node Ray object-store cap.
- Staged matched one-node synchronous and NVRx asynchronous save / fresh-allocation resume controls using the certified TOML and image-owned `/opt/ray_venvs`.
