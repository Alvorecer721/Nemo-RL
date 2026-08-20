# Session State

- Session: 20260820_003344
- Repo: /capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl/v0.7.0/.tmp/nemo-bridge-sync
- Branch: codex/sync-megatron-bridge-d9212902
- Started: 2026-08-20 00:33:44 CEST
- Updated: 2026-08-20 11:00 CEST

## Goal

Upgrade the Apertus NeMo-RL stack to Megatron-Bridge `d9212902`, deliver a reproducible GH200 image, and validate the baked image before publishing the PR.

## Current Subtask

Finish publication and production validation: multi-node GRPO, a true
cross-allocation restore, async endurance, and text-only SGLang.

## Loaded Skills

- `build-and-dependency` - use uv and preserve the hermetic image boundary.
- `testing` - validate the real Ray/GPU paths, not only imports.
- `nemo-rl-session-memory` - preserve the long-running build and probe state.

## Current Status

Image build job 3127636 completed. Artifact:
`/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-3b116bb38113-723462d5ac40.sqsh`
(SHA-256 `41e3c8afd68f40bc47f2b2ff2b0ed28525a9642f71f70abcc8d5e080dc3c66ed`).

Operational invariant: on 334 GiB nodes, a dependency-changing build must not carry the hermetic build graph into release assembly. The launcher now maps `HERMETIC_CACHE_TAG=rebuild` to `--target=hermetic`, publishes the hermetic image under its dependency fingerprint, prints the exact cache pin and digests, and exits. Release assembly must resume in a fresh allocation-local Podman store.

Bridge lint follow-up: mandatory full tracked-file Ruff checks found mechanical
import-order/format failures in the merged Bridge fork. Signed Bridge commit
`535b7aa7` fixes only formatting and passes Ruff check and format-check across
all 1,706 tracked Python files. Bridge PR #3 merged it into the fork's `main`,
so the NeMo-RL gitlink is now remotely resolvable.

The exact candidate image has now passed all bounded release probes:

- sync GRPO refit on four GH200 GPUs, job `3128582`;
- two-step async GRPO with CUDA graphs and repeated refit, job `3128583`;
- async DPO checkpoint save followed by a fresh-process resume to step 2, job `3128587`;
- ten-step, two-node/eight-GPU Apertus DPO with TP2 Megatron workers spanning
  both hosts, job `3130139` (`train/loss` step 10 = `0.6937193870544434`).

The multi-node launch attempt `3130080` failed before NeMo-RL started because
the Ray head raylet did not register after an over-constrained static port
layout. The replacement retained only the established low worker-port range,
increased Ray's registration window, registered two nodes/eight GPUs, and
completed with `baked_dpo_multinode_training=OK`.

Job `3130089` first passed the corrected multi-node launcher for two optimizer
steps. The final bounded gate `3130139` then repeated the test in a new Slurm
allocation and completed ten optimizer steps. All ten losses were finite
(`0.601646` to `0.763633`), every step reported eight valid samples and sixteen
global valid sequences, and mean step time was `2.3791` seconds.

The async crash/EOF/checkpoint-order fixes landed on `main` as `cc904664e`, are
ancestors of the candidate-image source commit, and passed 175 focused tests.
The earlier handoff incorrectly searched only for the pre-merge SHA
`d0d97df4c`.

Publication status: Gym PR #22 merged as `53cf1c038`. The six unpublished
Bridge/build/probe commits were cleanly rebased onto that new `main`.
Multi-node GRPO job `3130301` completed ten steps across two nodes/eight GPUs;
all ten generation-KL checks were `0.0000` and the final refit gate passed.

Cross-allocation save `3130280` completed on `nid007583`; restore `3130309`
completed on `nid007628`, but revealed that NeMo-RL treated the embedded
PyTorch-DCP optimizer as absent and resumed weights only. The current Bridge
format stores optimizer entries in `.metadata`, while the fork recognized only
the old `common.pt`. A narrow detector and regression tests now pass locally
(54 checkpoint tests plus the real saved metadata). Baked-image preflight
`3130348` correctly reproduced the old behavior because the candidate image
predates this source fix. An exact-head source-only image rebuild is therefore
required before the corrected restore can be called production evidence.

Megatron-backed Apertus SGLang job `3130299` loaded four engines but reached
the explicit unsupported boundary
`MegatronPolicyWorker.set_rollout_num_gpus_per_engine`. Supported DTensor-v2
Apertus SGLang job `3130467` is the replacement compatibility probe. Async
endurance `3130279` remains healthy beyond step 100.

## Plan

- [x] Run one-GPU provider/image smoke against the exact SquashFS (job 3128457).
- [x] Run four-GPU DPO with fused XIELU (job 3128458).
- [x] Run sync and async GRPO refit probes (jobs 3128582 and 3128583).
- [x] Run async checkpoint/save fresh-process resume (job 3128587).
- [x] Run ten-step two-node/eight-GPU Apertus Megatron training (job 3130139).
- [x] Record results and address any real failure.
- [x] Run full tracked-file NeMo-RL and Bridge Ruff validation.
- [x] Complete ten-step two-node GRPO/refit (job 3130301).
- [ ] Rebuild exact-head image and repeat optimizer-aware cross-allocation restore.
- [ ] Complete bounded 500-step async-GRPO endurance (job 3130279).
- [ ] Complete supported DTensor-v2 text-only Apertus SGLang GRPO (job 3130467).
- [ ] Commit/push final probe evidence and open the Bridge-upgrade NeMo-RL PR.

## Assumptions

- The existing XIELU extension remains compatible because Python 3.13, Torch 2.11+cu130, CUDA ABI, and AArch64/GH200 are unchanged; verify with CUDA forward/backward.

## Blockers

- Exact-head image rebuild and active production-evidence jobs must finish
  before the final evidence commit.

The earlier Slurm-controller diagnosis was a sandbox-network false positive.
Escalated Slurm calls reached the healthy controller and all jobs above were
scheduled normally.

## Deferred Validation

- multi-day endurance remains beyond the bounded 500-step gate.
