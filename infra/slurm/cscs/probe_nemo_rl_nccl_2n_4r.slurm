#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=net-2n4r
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/nemo_rl_nccl_2n_4r_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/nemo_rl_nccl_2n_4r_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${MASTER_ADDR:-${NODES[0]}}
MASTER_PORT=${MASTER_PORT:-29613}

echo "batch_host=$(hostname)"
echo "container_env=$CONTAINER_ENV"
echo "master_addr=$MASTER_ADDR"
echo "master_port=$MASTER_PORT"
echo "nodes=$SLURM_JOB_NODELIST"
echo "ntasks=$SLURM_NTASKS"
echo "NCCL_NET_OVERRIDE=${NCCL_NET_OVERRIDE:-}"
echo "NCCL_CROSS_NIC_OVERRIDE=${NCCL_CROSS_NIC_OVERRIDE:-}"
echo "NCCL_NET_GDR_LEVEL_OVERRIDE=${NCCL_NET_GDR_LEVEL_OVERRIDE:-}"
echo "FI_CXI_RX_MATCH_MODE_OVERRIDE=${FI_CXI_RX_MATCH_MODE_OVERRIDE:-}"
echo "FI_CXI_DEFAULT_TX_SIZE_OVERRIDE=${FI_CXI_DEFAULT_TX_SIZE_OVERRIDE:-}"
echo "FI_CXI_RDZV_GET_MIN_OVERRIDE=${FI_CXI_RDZV_GET_MIN_OVERRIDE:-}"
echo "NCCL_NET_FORCE_FLUSH_OVERRIDE=${NCCL_NET_FORCE_FLUSH_OVERRIDE:-}"
echo "NCCL_DEBUG_OVERRIDE=${NCCL_DEBUG_OVERRIDE:-}"

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
    export NCCL_NET=${NCCL_NET_OVERRIDE:-${NCCL_NET:-AWS Libfabric}}
    export NCCL_CROSS_NIC=${NCCL_CROSS_NIC_OVERRIDE:-${NCCL_CROSS_NIC:-1}}
    export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL_OVERRIDE:-${NCCL_NET_GDR_LEVEL:-PHB}}
    export FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE_OVERRIDE:-${FI_CXI_RX_MATCH_MODE:-software}}
    if [[ -n "${FI_CXI_DEFAULT_TX_SIZE_OVERRIDE:-}" ]]; then
      export FI_CXI_DEFAULT_TX_SIZE="$FI_CXI_DEFAULT_TX_SIZE_OVERRIDE"
    fi
    if [[ -n "${FI_CXI_RDZV_GET_MIN_OVERRIDE:-}" ]]; then
      export FI_CXI_RDZV_GET_MIN="$FI_CXI_RDZV_GET_MIN_OVERRIDE"
    fi
    if [[ -n "${NCCL_NET_FORCE_FLUSH_OVERRIDE:-}" ]]; then
      export NCCL_NET_FORCE_FLUSH="$NCCL_NET_FORCE_FLUSH_OVERRIDE"
    fi
    export NCCL_DEBUG=${NCCL_DEBUG_OVERRIDE:-${NCCL_DEBUG:-WARN}}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
    export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
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
import time

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
device_count = torch.cuda.device_count()
device = 0 if device_count == 1 else local_rank
nccl_net = os.environ.get("NCCL_NET")
nccl_cross_nic = os.environ.get("NCCL_CROSS_NIC")
fi_cxi_rx_match_mode = os.environ.get("FI_CXI_RX_MATCH_MODE")
fi_cxi_default_tx_size = os.environ.get("FI_CXI_DEFAULT_TX_SIZE")
fi_cxi_rdzv_get_min = os.environ.get("FI_CXI_RDZV_GET_MIN")
nccl_net_force_flush = os.environ.get("NCCL_NET_FORCE_FLUSH")

torch.cuda.set_device(device)

print(
    f"rank={rank} host={socket.gethostname()} world={world_size} "
    f"local_rank={local_rank} cuda_visible={visible} device_count={device_count} "
    f"device={device} gpu={torch.cuda.get_device_name(device)} "
    f"NCCL_NET={nccl_net} NCCL_CROSS_NIC={nccl_cross_nic} "
    f"FI_CXI_RX_MATCH_MODE={fi_cxi_rx_match_mode} "
    f"FI_CXI_DEFAULT_TX_SIZE={fi_cxi_default_tx_size} "
    f"FI_CXI_RDZV_GET_MIN={fi_cxi_rdzv_get_min} "
    f"NCCL_NET_FORCE_FLUSH={nccl_net_force_flush}",
    flush=True,
)

dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)

for i in range(20):
    t = torch.tensor([float(rank + 1 + i)], device="cuda")
    dist.all_reduce(t)
    expected = sum(float(r + 1 + i) for r in range(world_size))
    if t.item() != expected:
        raise AssertionError(f"rank={rank} iter={i} got={t.item()} expected={expected}")

dist.barrier()

src = torch.tensor([1234.0], device="cuda")
if rank != 0:
    src.zero_()
dist.broadcast(src, src=0)
if src.item() != 1234.0:
    raise AssertionError(f"rank={rank} broadcast got={src.item()}")

payload = torch.full((1024 * 1024,), float(rank), device="cuda", dtype=torch.float32)
gathered = [torch.empty_like(payload) for _ in range(world_size)]
dist.all_gather(gathered, payload)
torch.cuda.synchronize()

print(f"rank={rank} PASS all_reduce_loop broadcast all_gather t={time.time()}", flush=True)
dist.destroy_process_group()
PY
  '
