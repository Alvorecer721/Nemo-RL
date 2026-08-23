# Session State

- Goal: turn the GLM-5.1 scale prototype into production evidence for optimizer checkpoint completion, fresh-allocation resume, reference KL, and endurance.
- Branch: `autoresearch/2026-08-23-glm51-production/sync-checkpoint`
- Base: `b23a0d582`
- Certified image: `/capstor/store/cscs/swissai/infra01/MLLM/containers/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.aarch64.sqsh`
- Reservation: `SD-69241-apertus-1-5-0`, expires 2026-08-31 12:00 CEST. Do not cancel it.
- Historical incomplete checkpoint and the 1.488-TB conversion cache are preservation boundaries.

## Current hypothesis

The 272/288 save stalled after 16 rank-local persistent NVRx writers failed to report completion. The historical logs do not retain their tracebacks. Ray also reserved about 145 GB of object-store memory per node, leaving little host-memory headroom for 30-52 GB checkpoint shards. Test a bounded synchronous save with a smaller configurable Ray object store before retrying async persistence.

## Plan

- [x] Rebase the investigation onto current certified main.
- [x] Expose Bridge checkpoint save/load controls and a Ray object-store cap.
- [ ] Pass matched current-source, exact-image sync and async save plus fresh-allocation resume controls.
- [ ] Port the GLM harness without a custom nested actor-venv directory.
- [ ] Complete a 288-rank GLM optimizer save and fresh-allocation resume.
- [ ] Runtime-prove nonzero reference KL.
- [ ] Run representative endurance and publish only runtime-proven changes.
