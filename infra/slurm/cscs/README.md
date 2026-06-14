# CSCS Slurm Probes

This directory contains the Clariden/GH200 Slurm wrappers used to validate the
NeMo-RL `nvcr.io/nvidia/nemo-rl:v0.6.0` container on Slingshot.

The default container environment is `docker/nemo_rl.toml` in this checkout. The
wrappers set `CUDA_CACHE_PATH` and Hugging Face cache paths in shell code because
TOML values are not shell-expanded by Pyxis/EDF.

Run from the repository root after creating the log directory:

```bash
mkdir -p logs
sbatch infra/slurm/cscs/probe_nemo_rl_env.slurm
sbatch infra/slurm/cscs/probe_nemo_rl_nccl_2n_4r.slurm
sbatch infra/slurm/cscs/probe_nemo_rl_dpo_megatron_2n.slurm
```

Useful overrides:

```bash
CONTAINER_ENV=/users/xyixuan/.edf/nemo_rl.toml sbatch infra/slurm/cscs/probe_nemo_rl_env.slurm
GPUS_PER_NODE=4 TRAIN_GLOBAL_BATCH_SIZE=16 sbatch infra/slurm/cscs/probe_nemo_rl_dpo_megatron_2n.slurm
```

