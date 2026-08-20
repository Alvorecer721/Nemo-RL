# Files

## Inspected

- `infra/slurm/cscs/build_nemo_rl_image.slurm` - hermetic-cache and release control flow.
- `infra/slurm/cscs/README.md` - documented two-allocation build contract.
- `logs/nrl-vllm0251-image_3126533.*` - first-job storage failure evidence.
- `logs/nrl-vllm0251-image_3127636.*` - successful fresh-node build evidence.

## Changed

- `infra/slurm/cscs/build_nemo_rl_image.slurm` - pinned the Bridge dependency cache, fingerprint, and digests in commit `3b116bb38`.
- `infra/slurm/cscs/build_nemo_rl_image.slurm` - enforce hermetic-only publication for `HERMETIC_CACHE_TAG=rebuild`.
- `infra/slurm/cscs/README.md` - document the enforced two-job dependency rebuild contract and failure evidence.
- `infra/slurm/cscs/probe_nemo_rl_dpo_vllm0251_image.slurm` - optional async
  checkpoint/save and fresh-process resume gate.
- `infra/slurm/cscs/probe_nemo_rl_dpo_vllm0251_multinode_image.slurm` - new
  two-node/eight-GPU baked-image Apertus Megatron DPO gate.
- `infra/slurm/cscs/probe_nemo_rl_dpo_vllm0251_image.slurm` - explicit
  `save`, `resume`, and backward-compatible same-allocation checkpoint modes.
- `infra/slurm/cscs/probe_nemo_rl_grpo_vllm0251_image.slurm` - configurable
  duration and generation actor for vLLM/SGLang compatibility gates.
- `infra/slurm/cscs/probe_nemo_rl_grpo_async_vllm0251_image.slurm` -
  configurable duration and rollout size for endurance testing.
- `infra/slurm/cscs/probe_nemo_rl_grpo_vllm0251_multinode_image.slurm` - new
  two-node/eight-GPU GRPO, generation-KL, and refit gate.
- `nemo_rl/utils/checkpoint.py` - recognize embedded optimizer state in current
  Megatron PyTorch-DCP `.metadata` checkpoints.
- `tests/unit/utils/test_checkpoint.py` - DCP optimizer and weights-only resume
  regression coverage.

## Generated

- `.tmp/nemo_rl_bridge_3b116bb.toml` - test-only EDF pointing at the new SquashFS.
- `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-3b116bb38113-723462d5ac40.sqsh` - candidate image.
- `.tmp/nemo_rl_bridge_0e73bdb.toml` - exact runtime-source image EDF.
- `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-0e73bdb8367e-f8a605afd716.sqsh` - exact runtime-source image used by the final probe ladder.

## Evidence

- `logs/nrl_vllm0251_smoke_3128457.out` - exact-image GH200 vLLM generation smoke passed.
- `logs/nrl_vllm0251_dpo_3128458.out` - XIELU CUDA forward/backward and four-GPU one-step DPO passed.
- Bridge commit `535b7aa7` - mechanical Ruff fixes; full Bridge tracked-file check and format-check pass.
- NeMo-RL tracked-file Ruff check and format-check - 528 Python files pass.
- `logs/nrl_vllm0251_grpo_3128582.out` - sync GRPO and refit gate passed.
- `logs/nrl_vllm0251_grpo_async_3128583.out` - two-step async GRPO gate passed.
- `logs/nrl_vllm0251_dpo_3128587.out` - async checkpoint and fresh-process
  resume gate passed.
- `logs/nrl_vllm0251_dpo_mn_3130089.out` - two-node/eight-GPU Apertus
  Megatron DPO launcher smoke passed; `train/loss[2]=0.7125550508499146`.
- `logs/nrl_vllm0251_dpo_mn_3130139.out` - final ten-step,
  two-node/eight-GPU Apertus Megatron DPO gate passed;
  `train/loss[10]=0.6937193870544434`.
- Slurm accounting: jobs `3128582`, `3128583`, `3128587`, `3130089`, and `3130139`
  completed with exit `0:0`.
- `logs/nrl_vllm0251_grpo_mn_3130301.out` - ten-step two-node/eight-GPU GRPO
  and refit gate passed with ten zero generation-KL checks.
- `logs/nrl_vllm0251_dpo_3130280.out` and
  `logs/nrl_vllm0251_dpo_3130309.out` - cross-allocation save/restore exposed
  weights-only resume caused by stale DCP optimizer detection.
- `logs/nrl-vllm0251-image_3130524.*` - exact-source hermetic phase completed
  and published fingerprinted cache metadata.
- `logs/nrl-vllm0251-image_3131131.*` - fresh-allocation release phase
  completed all 47 build steps and baked import checks.
- `logs/nrl_bridge_exact_tests_3132043.*` - exact-image checkpoint and GRPO
  focused tests passed.
- `logs/nrl_vllm0251_grpo_mn_3132023.out` - corrected ten-step two-node/eight-GPU Apertus GRPO and refit gate passed with real signal.
- `logs/nrl_vllm0251_dpo_3132149.out` and
  `logs/nrl_vllm0251_dpo_3132165.out` - exact-image optimizer-aware
  cross-allocation save/restore passed on different GH200 nodes.
- `logs/nrl_vllm0251_grpo_3132159.out` - DTensor-v2 text-only SGLang GRPO and
  repeated refit gate passed on four GH200 GPUs.
- `logs/nrl_vllm0251_grpo_async_3132025.out` - corrected 500-step async
  Apertus/vLLM endurance passed with Slurm exit `0:0` and real learning signal.
