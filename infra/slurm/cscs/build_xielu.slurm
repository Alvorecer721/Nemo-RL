#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=build-xielu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
set -euo pipefail

# Build the fused CUDA xIELU kernel against whatever torch/CUDA the container ships.
# The .so links libtorch's unstable C++ ABI and a specific libcudart SONAME, so one
# build serves exactly one (python, torch, cuda) triple — hence the ABI-tagged output.
REPO_DIR=${REPO_DIR:-$SLURM_SUBMIT_DIR}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_v070.toml}
SRC=${SRC:-$REPO_DIR/3rdparty/kernels/xielu}
OUTROOT=${OUTROOT:-/capstor/store/cscs/swissai/infra01/users/xyixuan/xielu-builds}
ARCHES=${ARCHES:-9.0}

unset LD_PRELOAD
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/sbatchlibs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
for v in "${!SLURM_SPANK_@}"; do unset "$v"; done

mkdir -p "$OUTROOT"
[[ -f "$SRC/setup.py" ]] || { echo "xielu source not found at $SRC (git submodule update --init 3rdparty/kernels)"; exit 1; }

srun --environment="$CONTAINER_ENV" --nodes=1 --ntasks=1 bash -lc '
set -euo pipefail
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST="'"$ARCHES"'"
export MAX_JOBS=${MAX_JOBS:-8}

echo "=== toolchain in the container ==="
python -c "import sys, torch; print(f\"python {sys.version_info.major}.{sys.version_info.minor} | torch {torch.__version__} | cuda {torch.version.cuda}\")"
nvcc --version | tail -2 || true

TAG=$(python -c "
import sys, torch
py = f\"cp{sys.version_info.major}{sys.version_info.minor}\"
tv = torch.__version__.split(\"+\")[0]
cu = (torch.version.cuda or \"cpu\").replace(\".\", \"\")
print(f\"{py}-torch{tv}-cu{cu}\")
")
STAGE="'"$OUTROOT"'/xielu-site-${TAG}"
echo "=== building -> $STAGE ==="
rm -rf "$STAGE"; mkdir -p "$STAGE"
# --no-build-isolation is required so the extension links the CONTAINER torch;
# an isolated build env would pull a different torch from PyPI and bind the wrong ABI.
# --python pins uv to the container venv: with VIRTUAL_ENV unset it otherwise picks its
# own managed CPython, which has neither torch nor setuptools.
PYBIN=$(which python)
uv pip install --python "$PYBIN" --no-build-isolation --target "$STAGE" "'"$SRC"'"

echo "=== verify (import + registered op) ==="
PYTHONPATH="$STAGE" python -c "
import torch, xielu
assert hasattr(torch.ops.xielu, \"xielu\"), \"torch.ops.xielu.xielu not registered\"
print(\"xielu module:\", xielu.__file__)
x = torch.randn(256, 128, dtype=torch.bfloat16, device=\"cuda\")
ap = torch.tensor([0.8], dtype=torch.bfloat16, device=\"cuda\")
an = torch.tensor([0.6], dtype=torch.bfloat16, device=\"cuda\")
y = torch.ops.xielu.xielu(x, ap, an, 0.5, -1e-6)
assert y.shape == x.shape and torch.isfinite(y.float()).all(), \"bad kernel output\"
print(\"forward OK:\", tuple(y.shape), y.dtype)
"
echo "=== ABI of the new .so ==="
readelf -d "$STAGE"/_xielu*.so | grep NEEDED | grep -E "cudart|torch|c10"
echo ""
echo "BUILT: $STAGE"
'
echo "END: $(date)"
