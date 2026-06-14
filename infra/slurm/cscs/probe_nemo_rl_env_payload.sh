#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_CACHE_DISABLE=${CUDA_CACHE_DISABLE:-0}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/cuda/compute-cache}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
mkdir -p "$CUDA_CACHE_PATH"

cd "${NEMO_RL_PROBE_DIR:-/opt/nemo-rl}"

section() {
  printf '\n===== %s =====\n' "$1"
}

section "node"
date
hostname
uname -a
arch
pwd
id

section "gpu"
nvidia-smi || true

section "tools"
which python3 || true
python3 -V || true
which uv || true
uv --version || true

section "selected environment"
env | sort | grep -E '^(SLURM|CUDA|NVIDIA|NCCL|FI_|CXI|PMIX|UCX|RAY|PYXIS|ENROOT|MASTER|HF_)' || true

section "network interfaces"
if command -v ip >/dev/null 2>&1; then
  ip -brief addr || true
else
  hostname -I || true
  cat /proc/net/dev || true
fi

section "communication libraries"
ldconfig -p 2>/dev/null | grep -Ei 'nccl|libfabric|cxi|ofi' || true

section "base uv/python import and single-rank nccl"
uv run python - <<'PY'
import importlib.util
import os
import platform
import socket

print("python", platform.python_version())
print("platform", platform.platform())
print("machine", platform.machine())
print("hostname", socket.gethostname())

for name in ["torch", "ray", "nemo_rl"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'found' if spec else 'missing'}")

import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count", torch.cuda.device_count())
    print("gpu0", torch.cuda.get_device_name(0))
    x = torch.tensor([1.0], device="cuda")
    print("cuda_tensor", x.item())

    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    dist.init_process_group("nccl", rank=0, world_size=1)
    dist.all_reduce(x)
    print("nccl_single_rank_all_reduce", x.item())
    dist.destroy_process_group()
PY

if [[ "${RUN_HEAVY_IMPORTS:-0}" == "1" ]]; then
  section "optional backend imports"
  uv run --extra mcore python -c 'import megatron.core, transformer_engine.pytorch; print("mcore ok")'
  uv run --extra vllm python -c 'import vllm; print("vllm", getattr(vllm, "__version__", "unknown"))'
  uv run --extra sglang python -c 'import sglang; print("sglang ok")'
else
  section "optional backend imports skipped"
  echo "Set RUN_HEAVY_IMPORTS=1 when submitting to test mcore/vllm/sglang extras."
fi
