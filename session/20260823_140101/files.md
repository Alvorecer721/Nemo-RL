# Files

- `nemo_rl/models/policy/__init__.py`: checkpoint configuration surface.
- `nemo_rl/models/megatron/setup.py`: Bridge passthrough.
- `tests/unit/models/megatron/test_megatron_setup.py`: passthrough regression.
- `ray.sub`: optional Ray object-store memory cap.
- `infra/slurm/cscs/autoresearch/probe_glm51_checkpoint_sync_overlay.slurm`: exact-image save/resume control.
- `examples/configs/recipes/llm/autoresearch/grpo-glm5.1-80n4g-megatron-async-vllm-tp32-checkpoint-{save,resume}.yaml`: real GLM Phase A/B recipes using the upstream DAPO loader.
- `infra/slurm/cscs/autoresearch/run_glm51_cross_allocation_checkpoint.sh`: bounded 288-rank save/resume driver and shard-progress diagnostics.
- `infra/slurm/cscs/autoresearch/submit_glm51_cross_allocation_checkpoint.sh`: TOML, single-step Ray, 64-GiB object-store and 850-GB host-memory submission contract.
- `tests/unit/infra/test_glm51_cross_allocation_checkpoint.py`: resolved topology and cluster-control regression tests.
