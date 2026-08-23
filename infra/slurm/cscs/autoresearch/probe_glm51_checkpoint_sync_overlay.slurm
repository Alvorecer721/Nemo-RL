#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --account=infra01
#SBATCH --job-name=glm51-ckpt-ctrl
#SBATCH --output=logs/glm51_checkpoint_control_%j.out
#SBATCH --error=logs/glm51_checkpoint_control_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE

set -euo pipefail

PHASE=${1:?Usage: probe_glm51_checkpoint_sync_overlay.slurm save|resume CHECKPOINT_DIR sync|async}
CHECKPOINT_DIR=${2:?Usage: probe_glm51_checkpoint_sync_overlay.slurm save|resume CHECKPOINT_DIR sync|async}
SAVE_MODE=${3:-sync}
case "$PHASE" in
  save | resume) ;;
  *) echo "Unsupported phase: $PHASE" >&2; exit 1 ;;
esac
case "$SAVE_MODE" in
  sync)
    ASYNC_SAVE=false
    ASSUME_CONSTANT_STRUCTURE=false
    ;;
  async)
    ASYNC_SAVE=true
    ASSUME_CONSTANT_STRUCTURE=true
    ;;
  *) echo "Unsupported save mode: $SAVE_MODE" >&2; exit 1 ;;
esac

REPO_DIR=${REPO_DIR:-$SLURM_SUBMIT_DIR}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/glm51_checkpoint_control_${SAVE_MODE}_${PHASE}_${SLURM_JOB_ID}}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current}
APERTUS_CKPT=${APERTUS_CKPT:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200}
APERTUS_TOKENIZER=${APERTUS_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Apertus-v1.5-8B-official}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml}
SOURCE_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ -r "$RECIPE" ]] || { echo "Missing recipe: $RECIPE" >&2; exit 1; }
[[ -f "$XIELU_SITE/xielu/__init__.py" ]] || { echo "Missing xIELU site: $XIELU_SITE" >&2; exit 1; }
if [[ "$PHASE" == "save" ]]; then
  [[ ! -e "$CHECKPOINT_DIR" ]] || { echo "Refusing existing checkpoint directory: $CHECKPOINT_DIR" >&2; exit 1; }
else
  [[ -f "$CHECKPOINT_DIR/step_1/training_info.json" ]] || { echo "Missing step_1 checkpoint" >&2; exit 1; }
fi

mkdir -p "$RUN_DIR"
unset LD_PRELOAD

srun \
  --mpi=pmix \
  --cpu-bind=none \
  --environment="$CONTAINER_ENV" \
  --ntasks=1 \
  --gpus-per-task=4 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    unset NEMO_RL_PY_EXECUTABLES_SYSTEM
    export REPO_DIR='"$REPO_DIR"'
    export RUN_DIR='"$RUN_DIR"'
    export PHASE='"$PHASE"'
    export SAVE_MODE='"$SAVE_MODE"'
    export ASYNC_SAVE='"$ASYNC_SAVE"'
    export ASSUME_CONSTANT_STRUCTURE='"$ASSUME_CONSTANT_STRUCTURE"'
    export CHECKPOINT_DIR='"$CHECKPOINT_DIR"'
    export APERTUS_CKPT='"$APERTUS_CKPT"'
    export APERTUS_TOKENIZER='"$APERTUS_TOKENIZER"'
    export RECIPE='"$RECIPE"'
    export SOURCE_HEAD='"$SOURCE_HEAD"'
    export XIELU_SITE='"$XIELU_SITE"'
    export NEMO_RL_VENV_DIR=/opt/ray_venvs
    export NRL_IGNORE_VERSION_MISMATCH=1
    export PYTHONPATH="$XIELU_SITE:$REPO_DIR"
    export PYTHONUNBUFFERED=1
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
    export WANDB_DISABLED=true
    export LOG_DIR="$RUN_DIR/tb"
    export RUN_LOG="$RUN_DIR/run.log"
    MCORE_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python

    mkdir -p "$LOG_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
    cd "$REPO_DIR"
    test -x "$MCORE_PYTHON"

    "$MCORE_PYTHON" - <<'"'"'PY'"'"'
import os
from pathlib import Path

from nemo_rl.models.megatron.setup import _create_checkpoint_config

cfg = _create_checkpoint_config(
    "/tmp/pretrained",
    "/tmp/weights",
    "/tmp/optimizer",
    ckpt_cfg={
        "fully_parallel_save": False,
        "async_save": os.environ["ASYNC_SAVE"] == "true",
        "ckpt_load_validate_sharding_integrity": False,
    },
)
assert cfg.fully_parallel_save is False
assert cfg.async_save is (os.environ["ASYNC_SAVE"] == "true")
assert cfg.ckpt_load_validate_sharding_integrity is False
assert Path("/opt/ray_venvs").is_dir()
print("glm51_checkpoint_control_config=OK")
PY

    /opt/nemo_rl_venv/bin/python -m examples.run_dpo \
      --config "$RECIPE" \
      policy.model_name="$APERTUS_CKPT" \
      policy.tokenizer.name="$APERTUS_TOKENIZER" \
      dpo.max_num_steps=2 \
      dpo.val_batches=1 \
      dpo.val_period=100 \
      dpo.val_at_start=false \
      dpo.val_at_end=false \
      policy.train_global_batch_size=8 \
      policy.max_total_sequence_length=2048 \
      policy.megatron_cfg.tensor_model_parallel_size=2 \
      policy.megatron_cfg.sequence_parallel=false \
      policy.megatron_cfg.use_fused_linear_logprobs=true \
      policy.megatron_cfg.fused_linear_logprobs_chunk_size=256 \
      policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false \
      policy.megatron_cfg.checkpoint.fully_parallel_save=false \
      policy.megatron_cfg.checkpoint.async_save="$ASYNC_SAVE" \
      policy.megatron_cfg.checkpoint.ckpt_assume_constant_structure="$ASSUME_CONSTANT_STRUCTURE" \
      policy.megatron_cfg.checkpoint.ckpt_load_validate_sharding_integrity=false \
      logger.tensorboard_enabled=true \
      logger.log_dir="$LOG_DIR" \
      logger.wandb_enabled=false \
      logger.monitor_gpus=true \
      checkpointing.enabled=true \
      checkpointing.checkpoint_dir="$CHECKPOINT_DIR" \
      checkpointing.metric_name=null \
      checkpointing.keep_top_k=2 \
      checkpointing.save_period=1 \
      checkpointing.save_optimizer=true \
      cluster.gpus_per_node=4 \
      cluster.num_nodes=1 \
      2>&1 | tee "$RUN_LOG"

    test -f "$CHECKPOINT_DIR/step_2/training_info.json"
    if [[ "$PHASE" == "save" ]]; then
      test -f "$CHECKPOINT_DIR/step_1/training_info.json"
      mv "$CHECKPOINT_DIR/step_2" "$RUN_DIR/initial_step_2"
      echo "glm51_checkpoint_control_${SAVE_MODE}_save=OK"
    else
      grep -Fq "successfully loaded checkpoint" "$RUN_LOG"
      grep -Fq "at iteration 1" "$RUN_LOG"
      echo "glm51_checkpoint_control_${SAVE_MODE}_resume=OK"
    fi

    /opt/nemo_rl_venv/bin/python - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

artifact = {
    "phase": os.environ["PHASE"],
    "save_mode": os.environ["SAVE_MODE"],
    "source_head": os.environ["SOURCE_HEAD"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "checkpoint_dir": os.environ["CHECKPOINT_DIR"],
    "image_owned_worker_venvs": "/opt/ray_venvs",
    "terminal_green": True,
}
Path(os.environ["RUN_DIR"], "terminal.json").write_text(json.dumps(artifact, indent=2) + "\n")
PY
  '
