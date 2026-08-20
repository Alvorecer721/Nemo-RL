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

## Generated

- `.tmp/nemo_rl_bridge_3b116bb.toml` - test-only EDF pointing at the new SquashFS.
- `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-3b116bb38113-723462d5ac40.sqsh` - candidate image.

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
