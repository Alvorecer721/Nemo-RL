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
