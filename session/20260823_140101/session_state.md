# Session State

- Goal: turn the GLM-5.1 scale prototype into production evidence for optimizer checkpoint completion, fresh-allocation resume, reference KL, and endurance.
- Branch: `autoresearch/2026-08-24-glm51-tp2-cpu-placeholder`
- Base: `b23a0d582`
- Certified image: `/capstor/store/cscs/swissai/infra01/MLLM/containers/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.aarch64.sqsh`
- Reservation: `SD-69241-apertus-1-5-0`, expires 2026-08-31 12:00 CEST. Do not cancel it.
- Historical incomplete checkpoint and the 1.488-TB conversion cache are preservation boundaries.

## Current diagnosis

Two independent memory faults are now isolated and runtime-proven. Save-side CPU pressure came from strict per-worker NUMA binding; preferred-local placement with fallback completed all 288 shards in `3164148`. Restore-side GPU pressure came from allocating a full optimizer load skeleton and then letting `torch.optim.Optimizer.load_state_dict` cast it to CUDA before Transformer Engine allocated its final scaled state. TP2 did not hide that duplication: unpatched fresh-allocation job `3168898` still CUDA-OOMed on final-stage ranks 272-287 at about 92.6-93.0 GiB allocated with only 12-277 MiB free. The clean MCore candidate `23ae88370` initializes TE's final optimizer representation as the DCP target and restores group metadata without the casting loader. Job `3169314` restored all 288 ranks in a fresh 80-node allocation, including ranks 272-287, at about 63.0 GiB allocated with about 26.5 GiB free; it then completed refit and training step 2 with loss `0.0295`, reward range `0-1`, generation KL `0.0025`, and exit code 0. This closes checkpoint save and cross-allocation optimizer recovery at GLM-5.1 scale. Multi-hour endurance remains a separate production gate.

## Plan

- [x] Rebase the investigation onto current certified main.
- [x] Expose Bridge checkpoint save/load controls and a Ray object-store cap.
- [x] Pass current-source sync and NVRx save plus fresh-allocation next-step controls; both red Slurm exits were post-proof harness assertions, not checkpoint failures.
- [x] Port the GLM harness without a custom dataset adapter. Source-overlay diagnostics use an empty head-scoped actor venv because mutating baked packages in place fails on Enroot's writable overlay; production remains image-owned.
- [x] Runtime-attribute the checkpoint loss to strict per-worker NUMA memory binding and replace it with preferred-local placement that allows fallback.
- [x] Complete a 288-rank GLM optimizer save and fresh-allocation resume. TP2 Phase A `3168499` wrote a complete 8.946-TB DCP checkpoint; patched Phase B `3169314` restored it and trained the next step.
- [x] Runtime-prove nonzero reference KL (`0.0025` after restore/refit in `3169314`).
- [ ] Run representative endurance and publish only runtime-proven changes.

## Current subtask (2026-08-24 02:47 CEST)

- MCore PR #1 is merged at `9c82d4cad8c2aba345903cefee56989ba46f7013`; Bridge PR #5 is merged at `6b24b9e7944300a2a908c4e710841a679d435b95`, and this NeMo branch pins that Bridge commit.
- Keep the valid TP2 checkpoint until PR evidence and integration pins are safely recorded.
- Preserve reservation `SD-69241-apertus-1-5-0` and the reusable 1.488-TB model-conversion cache.
- Publish the NeMo-RL branch and run its CI; the source overlay remains validation machinery, not the production install contract.

## Loaded skills

- `nemo-rl-auto-research` - one-variable experiment, committed hypothesis, terminal artifacts, and TSV ledger.
- `nemo-rl-session-memory` - checkpoint state before edits and long-running launches.
- `config-conventions` and `testing` - recipe inheritance, naming, and focused harness coverage.

## TP2 experiment plan

- [x] Add `TP2/PP18/ETP1/EP16` save/resume recipes on the existing 80-node layout. Dense DP is 8; expert DP remains 1.
- [x] Generalize only the harness topology preflight and pass focused tests/lint (9/9 tests; Ruff clean; `bash -n` clean).
- [x] Commit the hypothesis before launch.
- [x] Submit Phase A using the existing reservation and write a complete TP2 checkpoint (`3168499`).
- [x] Prove unpatched TP2 still fails during optimizer restoration (`3168898`).
- [x] Prove the final MCore patch restores in a fresh allocation and completes the next training step (`3169314`).
- [x] Add a launcher preflight so uninitialized or mismatched recursive submodules fail before node allocation (`558dd58b5` after DCO replay; focused tests 10/10, Ruff and `bash -n` clean).
- [x] Merge the clean MCore fix through fork PR #1, merge its Bridge gitlink through fork PR #5, and pin the merged Bridge SHA in NeMo-RL.
- [x] Relock with image-owned uv 0.11.28 in job `3169750`; 549 packages resolved in 1.04 seconds and `uv.lock` remained byte-identical.
- [ ] Publish the NeMo-RL branch, trigger CI with its full commit SHA, and address only substantive failures.
