#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-overlay2hf
#SBATCH --output=logs/convert_dpo_overlay_to_hf_%j.out
#SBATCH --error=logs/convert_dpo_overlay_to_hf_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=debug
#SBATCH --no-requeue
# Convert a Megatron DCP checkpoint to HF using the SFT-base-overlay approach
# (preserves xIELU beta/eps buffers from the SFT base, overwrites trainable
# params from the Megatron stream). Env: CKPT_DIR, HF_OUT_DIR.

set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}

CKPT_DIR=${CKPT_DIR:?CKPT_DIR must be set (e.g. results/dpo-apertus-pilot/step_100)}
HF_OUT_DIR=${HF_OUT_DIR:?HF_OUT_DIR must be set}

if [ ! -s "$CKPT_DIR/config.yaml" ]; then
  echo "ERROR: $CKPT_DIR/config.yaml missing." >&2
  exit 1
fi

ITER_DIR=$(ls -d "$CKPT_DIR"/policy/weights/iter_* 2>/dev/null | tail -n 1)
if [ -z "$ITER_DIR" ]; then
  echo "ERROR: no iter_* dir under $CKPT_DIR/policy/weights/" >&2
  exit 1
fi

# If the output already has safetensors, skip.
if compgen -G "$HF_OUT_DIR/*.safetensors" > /dev/null; then
  echo "[skip] $HF_OUT_DIR already has safetensors"
  exit 0
fi

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized." >&2
    exit 1
  fi
done

mkdir -p "$REPO_DIR/logs" "$(dirname "$HF_OUT_DIR")"

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}
    cd $REPO_DIR
    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'
    echo '[convert] CKPT_DIR=$CKPT_DIR'
    echo '[convert] ITER_DIR=$ITER_DIR'
    echo '[convert] HF_OUT_DIR=$HF_OUT_DIR'
    uv run --locked --extra mcore python -m nemo_rl_apertus.convert_dpo_to_hf \
      --config $CKPT_DIR/config.yaml \
      --megatron-ckpt-path $ITER_DIR \
      --hf-ckpt-path $HF_OUT_DIR \
      --overwrite
    echo '[convert] sanity check:'
    ls -la $HF_OUT_DIR | head -20
    [ -f $HF_OUT_DIR/model.safetensors.index.json ] || { echo 'FATAL: index missing'; exit 1; }
  "

echo "END TIME: $(date)"
