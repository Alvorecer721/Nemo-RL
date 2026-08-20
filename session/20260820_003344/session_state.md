# Session State

- Session: 20260820_003344
- Repo: /capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl/v0.7.0/.tmp/nemo-bridge-sync
- Branch: codex/sync-megatron-bridge-d9212902
- Started: 2026-08-20 00:33:44 CEST
- Updated: 2026-08-20 15:00 CEST

## Goal

Upgrade the Apertus NeMo-RL stack to Megatron-Bridge `d9212902`, deliver a reproducible GH200 image, and validate the baked image before publishing the PR.

## Current Subtask

Monitor the published Bridge-upgrade NeMo-RL PR.

## Loaded Skills

- `build-and-dependency` - use uv and preserve the hermetic image boundary.
- `testing` - validate the real Ray/GPU paths, not only imports.
- `nemo-rl-session-memory` - preserve the long-running build and probe state.

## Current Status

The exact runtime-source image completed in release job `3131131`. Artifact:
`/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-0e73bdb8367e-f8a605afd716.sqsh`
(SHA-256 `4f4602919f6b982df135abf8dd5c0ae9bc69509e83b90a857b190ce6fb725e3d`).
Hermetic job `3130524` published cache fingerprint
`508e7c3083af7ce63ece6885650b95dd70e66cc10f576b3ed8b41de6b7727d26`;
release assembly then ran on fresh node `nid007066` and completed all 47 steps.

Operational invariant: on 334 GiB nodes, a dependency-changing build must not carry the hermetic build graph into release assembly. The launcher now maps `HERMETIC_CACHE_TAG=rebuild` to `--target=hermetic`, publishes the hermetic image under its dependency fingerprint, prints the exact cache pin and digests, and exits. Release assembly must resume in a fresh allocation-local Podman store.

Bridge lint follow-up: mandatory full tracked-file Ruff checks found mechanical
import-order/format failures in the merged Bridge fork. Signed Bridge commit
`535b7aa7` fixes only formatting and passes Ruff check and format-check across
all 1,706 tracked Python files. Bridge PR #3 merged it into the fork's `main`,
so the NeMo-RL gitlink is now remotely resolvable.

The exact runtime-source image has now passed these bounded release probes:

- sync GRPO refit on four GH200 GPUs, job `3128582`;
- two-step async GRPO with CUDA graphs and repeated refit, job `3128583`;
- async DPO checkpoint save followed by a fresh-process resume to step 2, job `3128587`;
- ten-step, two-node/eight-GPU Apertus GRPO with repeated vLLM refits and real
  learning signal, job `3132023`;
- optimizer-aware DPO save/restore across different allocations and nodes,
  jobs `3132149` and `3132165`;
- five-step DTensor-v2 Qwen2.5-Math-1.5B SGLang GRPO/refit, job `3132159`.

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

Publication status: Gym PR #22 merged as `53cf1c038`. The Bridge/build/probe
commits were rebased onto that mainline. The old multi-node GRPO and endurance
jobs `3130301` and `3130279` completed operationally but used an incorrect
absolute `0.02` GRPO threshold that masked every valid nonzero batch; they are
not learning-signal evidence. The corrected multiplicative threshold is
validated by unit coverage and the replacement jobs.

Cross-allocation save `3132149` completed on `nid005395` with a two-step
scheduler horizon. Step 2 was moved aside, and restore `3132165` resumed step 1
on fresh node `nid006056`. The exact image detected optimizer entries in
PyTorch-DCP `.metadata`, restored optimizer/scheduler state, reproduced the
original step-2 loss exactly (`0.7125550508499146`), and emitted
`baked_dpo_cross_allocation_resume=OK`.

Megatron-backed Apertus SGLang remains explicitly unsupported. The supported
DTensor-v2 text-only probe used Qwen2.5-Math-1.5B. Job `3132159` completed five
steps with nine positive-reward samples, 2,518 nonzero advantage entries,
9,932 valid token-mask entries, final loss `0.0021`, Generation-KL
`0.0001`-`0.0002`, and repeated successful SGLang weight-update/cache-flush
responses. A separate Automodel lexicographic Torch-version check incorrectly
disables DTensor async checkpointing under Torch 2.11; checkpointing was
disabled for this probe and the defect is deferred from the Bridge PR.

Corrected async Apertus/vLLM job `3132025` completed 500 optimizer steps in
29:41 on `nid007400` with Slurm exit `0:0`. Its 500 step files contain 4,000
trajectories, 167 positive rewards, 95,952 nonzero advantage entries (maximum
absolute value `1.0`), and 1,000,157 valid token-mask entries out of 1,923,632.
Mean Generation-KL was `0.0003124` (maximum `0.0006`). The run emitted
`baked_async_grpo_tp2=OK` without a traceback, OOM, engine-health failure, or
collector-loop failure.

## Plan

- [x] Run one-GPU provider/image smoke against the exact SquashFS (job 3128457).
- [x] Run four-GPU DPO with fused XIELU (job 3128458).
- [x] Run sync and async GRPO refit probes (jobs 3128582 and 3128583).
- [x] Run async checkpoint/save fresh-process resume (job 3128587).
- [x] Run ten-step two-node/eight-GPU Apertus Megatron training (job 3130139).
- [x] Record results and address any real failure.
- [x] Run full tracked-file NeMo-RL and Bridge Ruff validation.
- [x] Rebuild exact runtime-source image (jobs 3130524 and 3131131).
- [x] Repeat optimizer-aware cross-allocation restore (jobs 3132149 and 3132165).
- [x] Complete supported DTensor-v2 text-only SGLang GRPO (job 3132159).
- [x] Complete corrected ten-step two-node GRPO/refit (job 3132023).
- [x] Complete corrected bounded 500-step async-GRPO endurance (job 3132025).
- [x] Commit/push final probe evidence and open Bridge-upgrade NeMo-RL PR #23.

## Assumptions

- The existing XIELU extension remains compatible because Python 3.13, Torch 2.11+cu130, CUDA ABI, and AArch64/GH200 are unchanged; verify with CUDA forward/backward.

## Blockers

- None.

## Publication

- PR: `https://github.com/Alvorecer721/Nemo-RL/pull/23`
- Required label: exactly `CI:L1`
- CI trigger: PR #23's `/ok to test <full-final-sha>` comment identifies the
  final publication-record commit.

The earlier Slurm-controller diagnosis was a sandbox-network false positive.
Escalated Slurm calls reached the healthy controller and all jobs above were
scheduled normally.

## Deferred Validation

- multi-day endurance remains beyond the bounded 500-step gate.
