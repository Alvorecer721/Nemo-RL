# Session State

- Goal: turn the GLM-5.1 scale prototype into production evidence for optimizer checkpoint completion, fresh-allocation resume, reference KL, and endurance.
- Branch: `autoresearch/2026-08-23-glm51-production/sync-checkpoint`
- Base: `b23a0d582`
- Certified image: `/capstor/store/cscs/swissai/infra01/MLLM/containers/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.aarch64.sqsh`
- Reservation: `SD-69241-apertus-1-5-0`, expires 2026-08-31 12:00 CEST. Do not cancel it.
- Historical incomplete checkpoint and the 1.488-TB conversion cache are preservation boundaries.

## Current diagnosis

The initiating fault was a strict NUMA memory-policy bug. Instrumented retry `3163625` failed at checkpoint with Slurm reporting `nid006944: task 20: Out Of Memory`. Ranks 200, 201 and 203 on that node wrote roughly 29.52-GB shards; rank 202, hard-bound to the roughly 120-GB CPU NUMA node 2, wrote none. The node's 891.29-GB job cgroup peaked at only 334.78 GB, so aggregate node memory was not exhausted. MCore's async preload creates a full CPU copy of each rank's tensors; strict `numa_set_membind` forbids fallback to free memory on other nodes. NVRx v0.6 can additionally turn a preload-child death into an infinite parent wait because it blocks on `preload_q.join()` without checking child liveness. Preferred-local placement with fallback passed the exact 80-node save in job `3164148`: 288/288 shards, 8.926 TB and complete metadata with no OOM. Fresh-allocation optimizer/replay restoration remains the final checkpoint gate.

## Plan

- [x] Rebase the investigation onto current certified main.
- [x] Expose Bridge checkpoint save/load controls and a Ray object-store cap.
- [x] Pass current-source sync and NVRx save plus fresh-allocation next-step controls; both red Slurm exits were post-proof harness assertions, not checkpoint failures.
- [x] Port the GLM harness without a custom dataset adapter. Source-overlay diagnostics use an empty head-scoped actor venv because mutating baked packages in place fails on Enroot's writable overlay; production remains image-owned.
- [x] Runtime-attribute the checkpoint loss to strict per-worker NUMA memory binding and replace it with preferred-local placement that allows fallback.
- [ ] Complete a 288-rank GLM optimizer save and fresh-allocation resume. Save is green in `3164148`; resume is pending.
- [ ] Runtime-prove nonzero reference KL.
- [ ] Run representative endurance and publish only runtime-proven changes.
