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

## Generated

- `.tmp/nemo_rl_bridge_3b116bb.toml` - test-only EDF pointing at the new SquashFS.
- `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-3b116bb38113-723462d5ac40.sqsh` - candidate image.

## Evidence

- `logs/nrl_vllm0251_smoke_3128457.out` - exact-image GH200 vLLM generation smoke passed.
- `logs/nrl_vllm0251_dpo_3128458.out` - XIELU CUDA forward/backward and four-GPU one-step DPO passed.
