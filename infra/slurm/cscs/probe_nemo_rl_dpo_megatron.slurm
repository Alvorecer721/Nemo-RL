#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-meg
#SBATCH --output=logs/nemo_rl_dpo_megatron_%j.out
#SBATCH --error=logs/nemo_rl_dpo_megatron_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
REPO_LOG_ROOT=${REPO_LOG_ROOT:-$REPO_DIR/logs}
RUN_DIR=${RUN_DIR:-$REPO_LOG_ROOT/nemo_rl_dpo_megatron_${SLURM_JOB_ID}}

mkdir -p "$RUN_DIR"

echo "batch_host=$(hostname)"
echo "container_env=$CONTAINER_ENV"
echo "run_dir=$RUN_DIR"

unset LD_PRELOAD

srun \
  --mpi=pmix \
  --environment="$CONTAINER_ENV" \
  --ntasks=1 \
  --gpus-per-task=2 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export CUDA_CACHE_DISABLE=${CUDA_CACHE_DISABLE:-0}
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/cuda/compute-cache}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
    export WANDB_DISABLED=true
    export RUN_DIR='"$RUN_DIR"'
    export LOG_DIR="$RUN_DIR/tb"
    export JSON_METRICS="$RUN_DIR/metrics.json"
    export RUN_LOG="$RUN_DIR/run.log"

    mkdir -p "$LOG_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$CUDA_CACHE_PATH"
    cd /opt/nemo-rl
    git config --global --add safe.directory /opt/nemo-rl || true

    uv run examples/run_dpo.py \
      --config examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-megatron.v2.yaml \
      policy.model_name=Qwen/Qwen3-0.6B \
      dpo.max_num_steps=3 \
      dpo.val_batches=1 \
      dpo.val_period=3 \
      policy.train_global_batch_size=8 \
      policy.megatron_cfg.tensor_model_parallel_size=1 \
      policy.megatron_cfg.sequence_parallel=false \
      logger.tensorboard_enabled=true \
      logger.log_dir="$LOG_DIR" \
      logger.wandb_enabled=false \
      logger.monitor_gpus=true \
      checkpointing.enabled=false \
      cluster.gpus_per_node=2 \
      cluster.num_nodes=1 \
      2>&1 | tee "$RUN_LOG"

    uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
    uv run tests/check_metrics.py "$JSON_METRICS" '"'"'data["train/loss"]["3"] < 6'"'"'
  '
