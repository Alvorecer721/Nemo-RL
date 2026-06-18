#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-dpo-tt
#SBATCH --output=logs/dpo_toolthinking_%j.out
#SBATCH --error=logs/dpo_toolthinking_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --export=NONE

set -euo pipefail

###############################################################################
# Apertus-1.5 8B offline-DPO smoke probe over the synthetic tools/thinking set.
# Generates the test data (tools/build_offline_dpo_apertus_testset.py), then runs
# examples/run_dpo_apertus.py (registers ToolThinkingPreferenceProcessor) on
# probe-dpo-apertus1p5-8b-toolthinking-1n4g-megatron.yaml. Verifies the fused
# tools/thinking/multi-turn path trains (loss finite). NOT a science run.
###############################################################################
# Run from the repo root (`cd <repo> && sbatch ...`) or `sbatch --chdir=<repo> ...`.
# REPO_DIR defaults to $PWD, where the relative #SBATCH logs/ also land (run `mkdir -p logs` first).
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/dpo_toolthinking_${SLURM_JOB_ID}}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/probe-dpo-apertus1p5-8b-toolthinking-1n4g-megatron.yaml}

# Apertus stack: bridge fork (apertus support) + the prebuilt CUDA xielu kernel
# (ABI-bound cp313 / torch 2.10+cu129; unset XIELU_SITE for the eager fallback).
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.1-cp313}
# Synthetic test set knobs (8 combos × N samples, with a held-out val split).
TESTSET_DIR=${TESTSET_DIR:-$REPO_DIR/data/dpo_apertus_testset}
N_PER_COMBO=${N_PER_COMBO:-3}
VAL_FRACTION=${VAL_FRACTION:-0.25}
# Opt-in worker-venv rebuild — pass on EVERY stock-image run: the rebuild lands in /opt/ray_venvs (the container's ephemeral overlay) and does NOT persist, and the checkout's uv.lock diverges from the baked venvs (else the vLLM worker's numpy fails to load libscipy_openblas). Default off; `sbatch --export=ALL,NRL_FORCE_REBUILD_VENVS=true ...`. See infra/slurm/cscs/README.md (incl. NEMO_RL_VENV_DIR for real reuse).
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false}
# W&B: WANDB_API_KEY via the environment only (never on a command line). This probe defaults W&B OFF
# (recipe logger.wandb_enabled: false) — pass logger.wandb_enabled=true to enable, or
# WANDB_DISABLED=true to force off.
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_DISABLED=${WANDB_DISABLED:-}
WANDB_OVERRIDE=""
[ "${WANDB_DISABLED:-}" = true ] && WANDB_OVERRIDE="logger.wandb_enabled=false"

mkdir -p "$RUN_DIR"
echo "batch_host=$(hostname) run_dir=$RUN_DIR recipe=$(basename "$RECIPE")"

# Force srun to carry the full job env (incl. WANDB_API_KEY/WANDB_DISABLED) into the container.
export SLURM_EXPORT_ENV=ALL
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
    export REPO_DIR='"$REPO_DIR"'
    export BRIDGE_ROOT='"$BRIDGE_ROOT"'
    export XIELU_SITE='"$XIELU_SITE"'
    export NRL_FORCE_REBUILD_VENVS='"$NRL_FORCE_REBUILD_VENVS"'
    # our checkout first (Apertus deltas + nemo_rl_apertus), then bridge fork src, then CUDA xielu.
    export PYTHONPATH="$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}"
    export RUN_DIR='"$RUN_DIR"'
    export LOG_DIR="$RUN_DIR/tb"
    export JSON_METRICS="$RUN_DIR/metrics.json"
    export RUN_LOG="$RUN_DIR/run.log"
    TESTSET_DIR='"$TESTSET_DIR"'
    RECIPE='"$RECIPE"'

    mkdir -p "$LOG_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
    cd "$REPO_DIR"
    git config --global --add safe.directory "$REPO_DIR" || true

    # Apertus runtime guard: fail loud if nemo_rl is the stock /opt copy rather than this checkout.
    python -c "from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()"

    echo "=== prologue: generate synthetic tools/thinking preference set ==="
    python tools/build_offline_dpo_apertus_testset.py \
      --out "$TESTSET_DIR" --n-per-combo '"$N_PER_COMBO"' --val-fraction '"$VAL_FRACTION"'

    echo "=== DPO probe: tools/thinking set, TP2, $(basename "$RECIPE") ==="
    # Megatron-core runs in the Ray worker venvs; the driver does not need --extra mcore
    # (matches submit_nemo_rl_dpo.slurm, the proven Apertus offline-DPO launcher).
    uv run --locked examples/run_dpo_apertus.py \
      --config "$RECIPE" \
      logger.tensorboard_enabled=true \
      logger.log_dir="$LOG_DIR" '$WANDB_OVERRIDE' \
      2>&1 | tee "$RUN_LOG"

    echo "=== checking train loss is finite + bounded (NaN/inf fail the comparisons) ==="
    uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
    uv run tests/check_metrics.py "$JSON_METRICS" \
      "max(data[\"train/loss\"]) < 10" \
      "min(data[\"train/loss\"]) > 0"
  '
echo "END TIME: $(date)"
