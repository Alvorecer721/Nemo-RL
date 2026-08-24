# Files

- `nemo_rl/models/policy/__init__.py`: checkpoint configuration surface.
- `nemo_rl/models/megatron/setup.py`: Bridge passthrough.
- `tests/unit/models/megatron/test_megatron_setup.py`: passthrough regression.
- `ray.sub`: optional Ray object-store memory cap and allocation-ID preservation across the deliberate `SLURM_*` environment clear.
- `infra/slurm/cscs/autoresearch/probe_glm51_checkpoint_sync_overlay.slurm`: exact-image save/resume control.
- `examples/configs/recipes/llm/autoresearch/grpo-glm5.1-80n4g-megatron-async-vllm-tp32-checkpoint-{save,resume}.yaml`: real GLM Phase A/B recipes using the upstream DAPO loader.
- `infra/slurm/cscs/autoresearch/run_glm51_cross_allocation_checkpoint.sh`: bounded 288-rank save/resume driver, shard-progress diagnostics and terminal artifact keyed by the preserved allocation ID.
- `infra/slurm/cscs/autoresearch/collect_ray_node_diagnostics.py`: node-affine Ray sampler for cgroup OOM counters, memory peaks and checkpoint-writer process state.
- `infra/slurm/cscs/autoresearch/submit_glm51_cross_allocation_checkpoint.sh`: TOML, single-step Ray, 64-GiB object-store and 850-GB host-memory submission contract.
- `tests/unit/infra/test_glm51_cross_allocation_checkpoint.py`: resolved topology and cluster-control regression tests.
- `nemo_rl/distributed/numa_utils.py`: prefer GPU-local CPU memory while permitting fallback instead of hard-capping each worker at one NUMA node.
- `tests/unit/distributed/test_numa_utils.py`: preferred-memory-policy regression coverage.
- `examples/configs/recipes/llm/autoresearch/grpo-glm5.1-152n4g-megatron-async-vllm-tp32-checkpoint-resume.yaml`: capacity-only DP32 restore recipe; TP1/PP18 and the eight-node rollout pool remain unchanged.
- `infra/slurm/cscs/autoresearch/{submit,run}_glm51_cross_allocation_checkpoint.sh`: now accept an explicit node count, recipe, Phase-A terminal artifact and optional reservation without changing the certified 80-node defaults.

## TP2 no-fix diagnostic

- `examples/configs/recipes/llm/autoresearch/grpo-glm5.1-80n4g-megatron-tp2pp18ep16-async-vllm-tp32-checkpoint-{save,resume}.yaml`: fresh TP2/PP18/ETP1/EP16 Phase A/B pair on the 80-node reservation; dense DP8 and expert DP1.
- `infra/slurm/cscs/autoresearch/run_glm51_cross_allocation_checkpoint.sh`: validates both certified TP1 and diagnostic TP2 grids and prints dense/expert DP explicitly.
- `tests/unit/infra/test_glm51_cross_allocation_checkpoint.py`: covers both TP2 phases plus existing TP1/DP32 recipes; 9/9 focused tests pass.

## Restore-memory fix evidence

- `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/optimizer/__init__.py`: passes `store_param_remainders` to the TE 2.15 optimizer-state initializer.
- `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py`: uses TE's final float32 scaled optimizer state as the DCP target for the alias-safe precision-aware path, avoiding a second CUDA-resident load skeleton and preserving the existing fallback for other optimizers/dtypes.
- `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/optimizer/test_distrib_optimizer_load_state.py`: covers CPU fallback placeholders, the TE 2.15 initializer signature, tensor-identity preservation, repeat loads, and param-group metadata restoration; 3/3 focused tests pass.
- `infra/slurm/cscs/autoresearch/submit_glm51_cross_allocation_checkpoint.sh`: rejects uninitialized or gitlink-mismatched recursive submodules before allocating nodes.
- `tests/unit/infra/test_glm51_cross_allocation_checkpoint.py`: launcher preflight plus topology/cluster contracts; 10/10 focused tests pass.
- `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`: NeMo gitlink at merged Bridge SHA `6b24b9e7`, which pins merged MCore SHA `9c82d4ca`.
