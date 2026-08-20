# Timeline

## 2026-08-20 00:33 CEST

- Build 3126533 completed the hermetic stage but exhausted the 334 GiB node-local Podman workspace while committing release step 34.
- The completed hermetic cache was pinned by signed commit `3b116bb38`.
- Fresh-node build 3127636 resumed from cache `ccab8a76...`, crossed step 34, completed all 47 release steps, exported the SquashFS, and passed the baked vLLM import check.
- Decision: never treat pre-build pruning as sufficient for a dependency-changing build; enforce a clean storage boundary between hermetic and release phases.

## 2026-08-20 00:40 CEST

- Added the enforced two-job boundary to `build_nemo_rl_image.slurm` and documented it in the CSCS README.
- Job 3128457 passed the exact-SquashFS GH200 vLLM smoke: model load, generation, and Apertus tool parser all reported `OK`.
- Job 3128458 passed the shared XIELU CUDA forward/backward check and a four-GPU one-step Megatron DPO run; `train/loss=0.6931471824645996` passed the metric gate.
- Sync and async GRPO submissions did not receive job IDs because `clariden-slurmctl` became unreachable; `scontrol ping` reported the primary controller `DOWN`.

## 2026-08-20 01:12 CEST

- Full Ruff check and format-check passed for all 528 Python files tracked directly by NeMo-RL.
- Bridge's own mandatory tracked-file lint exposed eight import-order and seven formatting failures in the merged fork.
- Applied only Ruff's mechanical fixes and created signed local Bridge commit `535b7aa7`; full Bridge Ruff now passes across 1,706 tracked Python files and the changed files compile under Python 3.12.
- The existing candidate image remains valid functional evidence because `535b7aa7` changes formatting/import ordering only; exact-SHA provenance would require publishing the Bridge commit and producing a new image.

## 2026-08-20 10:14 CEST

- Corrected the earlier controller diagnosis: sandboxed Slurm calls could not
  create controller sockets, while escalated calls showed the controller was
  healthy.
- Candidate-image sync GRPO job `3128582` passed one real rollout, Megatron
  update, generation-KL check, and vLLM refit (`baked_grpo_refit=OK`).
- Candidate-image async GRPO job `3128583` passed two steps with CUDA graphs,
  async scheduling, two generation-KL checks, and repeated refit
  (`baked_async_grpo_tp2=OK`).
- DPO job `3128587` asynchronously saved step 1, started a fresh Python
  process, resumed to step 2, and verified `training_info.total_steps == 2`
  (`baked_dpo_async_checkpoint_resume=OK`).
- Added a two-node/eight-GPU Apertus DPO probe. Attempt `3130080` failed in Ray
  startup before NeMo-RL after an over-constrained static port layout. The
  corrected attempt `3130089` registered two fresh GH200 nodes/eight GPUs,
  initialized eight Megatron workers across both hosts, completed two TP2 DPO
  optimizer steps, passed the finite-loss gate at `0.7125550508499146`, and
  emitted `baked_dpo_multinode_training=OK` with Slurm exit `0:0`.
- Verified that async failure-path hardening commit `d0d97df4c` is on
  `origin/fix/async-grpo-reliability`, not in the Bridge branch/image.

## 2026-08-20 10:21 CEST

- Promoted the multi-node probe from a two-step startup smoke to a configurable
  bounded gate and launched job `3130139` with `MAX_STEPS=10` in a new Slurm
  allocation.
- The job again registered two nodes/eight GPUs and placed four TP2 data
  replicas across eight Megatron workers. It completed all ten DPO optimizer
  steps with finite loss (`0.601646` to `0.763633`), eight valid samples and
  sixteen global valid sequences at every step, `2.3791` seconds mean step
  time, and final loss `0.6937193870544434`.
- The step-10 metric gate passed and emitted
  `baked_dpo_multinode_training=OK`.
