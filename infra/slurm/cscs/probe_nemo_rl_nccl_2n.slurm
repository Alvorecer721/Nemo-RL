#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=nemo-rl-nccl-2n
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/nemo_rl_nccl_2n_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/nemo_rl_nccl_2n_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
MASTER_PORT=${MASTER_PORT:-29593}

echo "batch_host=$(hostname)"
echo "container_env=$CONTAINER_ENV"
echo "master_addr=$MASTER_ADDR"
echo "master_port=$MASTER_PORT"
echo "nodes=$SLURM_JOB_NODELIST"

unset LD_PRELOAD

srun \
  --mpi=pmix \
  --environment="$CONTAINER_ENV" \
  --ntasks="$SLURM_NTASKS" \
  --gpus-per-task=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export CUDA_CACHE_DISABLE=${CUDA_CACHE_DISABLE:-0}
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/cuda/compute-cache}
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export RANK=${SLURM_PROCID}
    export WORLD_SIZE=${SLURM_NTASKS}
    export LOCAL_RANK=${SLURM_LOCALID}
    mkdir -p "$CUDA_CACHE_PATH"

    cd /opt/nemo-rl
    python - <<'"'"'PY'"'"'
import os
import socket

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

print(
    f"rank={rank} host={socket.gethostname()} world={world_size} "
    f"local_rank={local_rank} cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(local_rank)}",
    flush=True,
)

dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
t = torch.tensor([float(rank + 1)], device="cuda")
dist.all_reduce(t)
expected = world_size * (world_size + 1) / 2
print(f"rank={rank} all_reduce={t.item()} expected={expected}", flush=True)
assert t.item() == expected
dist.destroy_process_group()
PY
  '
