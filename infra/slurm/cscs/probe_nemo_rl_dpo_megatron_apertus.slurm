#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-probe
#SBATCH --output=logs/nemo_rl_dpo_apertus_%j.out
#SBATCH --error=logs/nemo_rl_dpo_apertus_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE

set -euo pipefail

# Run from the repo root (`cd <repo> && sbatch ...`) or anywhere with `sbatch --chdir=<repo> ...`.
# REPO_DIR defaults to the job working dir ($PWD), which is also where the relative #SBATCH logs/ land.
# (sbatch copies this script to /var/spool/slurmd, so BASH_SOURCE doesn't resolve here.)
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/nemo_rl_dpo_apertus_${SLURM_JOB_ID}}

# Apertus stack: bridge fork (apertus support) shadows the image's bundled
# bridge; megatron-core comes from the worker venvs. All overridable.
# NOTE: XIELU_SITE points at the shared prebuilt CUDA xielu kernel in the wheelhouse (ABI-bound: cp313 / torch 2.10+cu129).
# Unset it for the eager fallback (correct, slower, warns).
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}
APERTUS_CKPT=${APERTUS_CKPT:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200}
APERTUS_TOKENIZER=${APERTUS_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed}

mkdir -p "$RUN_DIR"
echo "batch_host=$(hostname) run_dir=$RUN_DIR"

unset LD_PRELOAD

srun \
  --mpi=pmix \
  --environment="$CONTAINER_ENV" \
  --ntasks=1 \
  --gpus-per-task=4 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
    export WANDB_DISABLED=true
    export REPO_DIR='"$REPO_DIR"'
    export BRIDGE_ROOT='"$BRIDGE_ROOT"'
    export XIELU_SITE='"$XIELU_SITE"'
    # our nemo_rl checkout first (Apertus deltas: xIELU fix + #2329), then bridge fork src, then optional CUDA xielu.
    # megatron-core comes from the worker venvs — NeMo-RL v0.6.0 needs its own mcore version.
    export PYTHONPATH="$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}"
    export RUN_DIR='"$RUN_DIR"'
    export LOG_DIR="$RUN_DIR/tb"
    export JSON_METRICS="$RUN_DIR/metrics.json"
    export RUN_LOG="$RUN_DIR/run.log"
    APERTUS_CKPT='"$APERTUS_CKPT"'
    APERTUS_TOKENIZER='"$APERTUS_TOKENIZER"'

    mkdir -p "$LOG_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
    cd "$REPO_DIR"
    git config --global --add safe.directory "$REPO_DIR" || true

    # Apertus runtime guard: fail loud if nemo_rl is the stock /opt copy rather than this checkout.
    # See nemo_rl_apertus/runtime_guard.py.
    python -c "from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()"

    echo "=== prologue: apertus unit suite + tokenizer check in the real runtime env ==="
    # mcore extra env = the one Megatron workers run under (the base
    # /opt/nemo_rl_venv has no TE/modelopt in the public image); the tokenizer
    # argument additionally asserts the chat template loads
    uv run --locked --extra mcore python \
      "$BRIDGE_ROOT/src/megatron/bridge/models/apertus/test_apertus_provider.py" "$APERTUS_TOKENIZER"

    echo "=== DPO probe: 3 steps, TP2, seq 2048 ==="
    uv run examples/run_dpo.py \
      --config examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-megatron.v2.yaml \
      policy.model_name="$APERTUS_CKPT" \
      policy.tokenizer.name="$APERTUS_TOKENIZER" \
      dpo.max_num_steps=3 \
      dpo.val_batches=1 \
      dpo.val_period=3 \
      policy.train_global_batch_size=8 \
      policy.max_total_sequence_length=2048 \
      policy.megatron_cfg.tensor_model_parallel_size=2 \
      policy.megatron_cfg.sequence_parallel=false \
      logger.tensorboard_enabled=true \
      logger.log_dir="$LOG_DIR" \
      logger.wandb_enabled=false \
      logger.monitor_gpus=true \
      checkpointing.enabled=false \
      cluster.gpus_per_node=4 \
      cluster.num_nodes=1 \
      2>&1 | tee "$RUN_LOG"

    uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
    uv run tests/check_metrics.py "$JSON_METRICS" '"'"'data["train/loss"]["3"] < 6'"'"'
  '
