# Session State

- Goal: turn the GLM-5.1 scale prototype into production evidence for optimizer checkpoint completion, fresh-allocation resume, reference KL, and endurance.
- Branch: `autoresearch/2026-08-23-glm51-production/sync-checkpoint`
- Base: `b23a0d582`
- Certified image: `/capstor/store/cscs/swissai/infra01/MLLM/containers/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.aarch64.sqsh`
- Reservation: `SD-69241-apertus-1-5-0`, expires 2026-08-31 12:00 CEST. Do not cancel it.
- Historical incomplete checkpoint and the 1.488-TB conversion cache are preservation boundaries.

## Current diagnosis

The initiating fault was a strict NUMA memory-policy bug. Instrumented retry `3163625` failed at checkpoint with Slurm reporting `nid006944: task 20: Out Of Memory`. Ranks 200, 201 and 203 on that node wrote roughly 29.52-GB shards; rank 202, hard-bound to the roughly 120-GB CPU NUMA node 2, wrote none. The node's 891.29-GB job cgroup peaked at only 334.78 GB, so aggregate node memory was not exhausted. MCore's async preload creates a full CPU copy of each rank's tensors; strict `numa_set_membind` forbids fallback to free memory on other nodes. NVRx v0.6 can additionally turn a preload-child death into an infinite parent wait because it blocks on `preload_q.join()` without checking child liveness. Preferred-local placement with fallback passed the exact 80-node save in job `3164148`: 288/288 shards, 8.926 TB and complete metadata with no OOM. Fresh-allocation job `3165089` then loaded as far as Transformer Engine optimizer-state initialization but CUDA-OOMed only on ranks 272-287, the 16 DP replicas of the heavy final PP stage. This is fail-loud capacity pressure, not checkpoint corruption. A 152-node Phase B keeps checkpoint-compatible TP1/PP18 while increasing DP16 to DP32 and is the next capacity-only test.

## Plan

- [x] Rebase the investigation onto current certified main.
- [x] Expose Bridge checkpoint save/load controls and a Ray object-store cap.
- [x] Pass current-source sync and NVRx save plus fresh-allocation next-step controls; both red Slurm exits were post-proof harness assertions, not checkpoint failures.
- [x] Port the GLM harness without a custom dataset adapter. Source-overlay diagnostics use an empty head-scoped actor venv because mutating baked packages in place fails on Enroot's writable overlay; production remains image-owned.
- [x] Runtime-attribute the checkpoint loss to strict per-worker NUMA memory binding and replace it with preferred-local placement that allows fallback.
- [ ] Complete a 288-rank GLM optimizer save and fresh-allocation resume. Save is green in `3164148`; the 80-node resume failed on final-stage GPU capacity, and a 152-node DP32 restore is staged.
- [ ] Runtime-prove nonzero reference KL.
- [ ] Run representative endurance and publish only runtime-proven changes.

## Current subtask (2026-08-23 23:45 CEST)

- Test whether training TP2 avoids the unpatched Transformer Engine optimizer-restore CUDA peak.
- Keep the Megatron-LM CPU-placeholder fix absent from this experiment.
- Use a fresh TP2 Phase-A checkpoint and same-topology Phase-B restore rather than interpreting a TP1 optimizer checkpoint as TP2.
- Preserve reservation `SD-69241-apertus-1-5-0`; the user authorized removing the complete TP1 training checkpoint once the TP2 launch is staged. Preserve the reusable 1.488-TB model-conversion cache.

## Loaded skills

- `nemo-rl-auto-research` - one-variable experiment, committed hypothesis, terminal artifacts, and TSV ledger.
- `nemo-rl-session-memory` - checkpoint state before edits and long-running launches.
- `config-conventions` and `testing` - recipe inheritance, naming, and focused harness coverage.

## TP2 experiment plan

- [x] Add `TP2/PP18/ETP1/EP16` save/resume recipes on the existing 80-node layout. Dense DP is 8; expert DP remains 1.
- [x] Generalize only the harness topology preflight and pass focused tests/lint (9/9 tests; Ruff clean; `bash -n` clean).
- [ ] Commit the hypothesis before launch.
- [ ] Remove only the superseded complete TP1 optimizer checkpoint, then submit Phase A using the existing reservation.
- [ ] After a green Phase A, submit Phase B in a fresh allocation and require optimizer restoration plus the next training step.

The checked-out MCore remains unpatched: `distrib_optimizer.py` still allocates the resume placeholder with `device=torch.cuda.current_device()`.
