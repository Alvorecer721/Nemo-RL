# Session State

- Goal: turn the GLM-5.1 scale prototype into production evidence for optimizer checkpoint completion, fresh-allocation resume, reference KL, and endurance.
- Branch: `autoresearch/2026-08-23-glm51-production/sync-checkpoint`
- Base: `b23a0d582`
- Certified image: `/capstor/store/cscs/swissai/infra01/MLLM/containers/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.aarch64.sqsh`
- Reservation: `SD-69241-apertus-1-5-0`, expires 2026-08-31 12:00 CEST. Do not cancel it.
- Historical incomplete checkpoint and the 1.488-TB conversion cache are preservation boundaries.

## Current diagnosis

The 272/288 save stalled because 16 rank-local persistent NVRx children never completed D2H staging. Host-memory exhaustion is the evidence-backed root cause: the historical head step is recorded `OUT_OF_MEMORY`; the worker step's 261.33-GiB MaxRSS node is also a missing-writer node; missing writers are disproportionately concentrated on the 52-GB final pipeline stage; and the old 450-GiB node budget also carried a roughly 135-GiB Ray object store. NVRx then converted the child loss into an infinite wait because the parent blocks on `preload_q.join()` without checking child liveness. The next real-topology run captures cgroup OOM counters and writer processes on all nodes to close the remaining runtime-attribution gap.

## Plan

- [x] Rebase the investigation onto current certified main.
- [x] Expose Bridge checkpoint save/load controls and a Ray object-store cap.
- [x] Pass current-source sync and NVRx save plus fresh-allocation next-step controls; both red Slurm exits were post-proof harness assertions, not checkpoint failures.
- [x] Port the GLM harness without a custom dataset adapter. Source-overlay diagnostics use an empty head-scoped actor venv because mutating baked packages in place fails on Enroot's writable overlay; production remains image-owned.
- [ ] Complete a 288-rank GLM optimizer save and fresh-allocation resume.
- [ ] Runtime-prove nonzero reference KL.
- [ ] Run representative endurance and publish only runtime-proven changes.
