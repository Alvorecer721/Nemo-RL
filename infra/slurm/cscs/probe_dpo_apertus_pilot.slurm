#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-dpopilot
#SBATCH --output=logs/dpo_apertus_pilot_%j.out
#SBATCH --error=logs/dpo_apertus_pilot_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --no-requeue

# Pilot DPO on the 39,815-pair dataset from the pass@8 + Qwen teacher pipeline.
# 311 steps × ~15s ≈ 80 min compute, plus ~10 min cold init. 2.5h budget gives slack.
set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/dpo-apertus1p5-8b-pilot-megatron.yaml}

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized." >&2
    exit 1
  fi
done

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}
    cd $REPO_DIR
    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'
    uv run --locked --extra mcore python examples/run_dpo.py --config $RECIPE
  "
echo "END TIME: $(date)"
