# Session State

- Session: 20260820_003344
- Repo: /capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl/v0.7.0/.tmp/nemo-bridge-sync
- Branch: codex/sync-megatron-bridge-d9212902
- Started: 2026-08-20 00:33:44 CEST
- Updated: 2026-08-20 00:40 CEST

## Goal

Upgrade the Apertus NeMo-RL stack to Megatron-Bridge `d9212902`, deliver a reproducible GH200 image, and validate the baked image before publishing the PR.

## Current Subtask

Run the baked-image provider, fused-XIELU, checkpoint/refit, DPO, and GRPO probes.

## Loaded Skills

- `build-and-dependency` - use uv and preserve the hermetic image boundary.
- `testing` - validate the real Ray/GPU paths, not only imports.
- `nemo-rl-session-memory` - preserve the long-running build and probe state.

## Current Status

Image build job 3127636 completed. Artifact:
`/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-3b116bb38113-723462d5ac40.sqsh`
(SHA-256 `41e3c8afd68f40bc47f2b2ff2b0ed28525a9642f71f70abcc8d5e080dc3c66ed`).

Operational invariant: on 334 GiB nodes, a dependency-changing build must not carry the hermetic build graph into release assembly. The launcher now maps `HERMETIC_CACHE_TAG=rebuild` to `--target=hermetic`, publishes the hermetic image under its dependency fingerprint, prints the exact cache pin and digests, and exits. Release assembly must resume in a fresh allocation-local Podman store.

## Plan

- [x] Run one-GPU provider/image smoke against the exact SquashFS (job 3128457).
- [x] Run four-GPU DPO with fused XIELU (job 3128458).
- [ ] Run sync and async GRPO refit probes.
- [ ] Record results and address any real failure.

## Assumptions

- The existing XIELU extension remains compatible because Python 3.13, Torch 2.11+cu130, CUDA ABI, and AArch64/GH200 are unchanged; verify with CUDA forward/backward.

## Blockers

- Slurm controller `clariden-slurmctl` is currently unreachable from the login node (`scontrol ping` reports `DOWN`), so the sync and async GRPO jobs have not received job IDs.
