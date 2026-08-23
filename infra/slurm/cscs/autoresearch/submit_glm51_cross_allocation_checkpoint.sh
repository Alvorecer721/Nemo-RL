#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PHASE=${1:?Usage: submit_glm51_cross_allocation_checkpoint.sh save|resume}
case "$PHASE" in
  save | resume) ;;
  *) echo "Unsupported phase: $PHASE" >&2; exit 2 ;;
esac

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
EXPECTED_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
GLM_CKPT=${GLM_CKPT:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.1}
GLM_MEGATRON_CACHE=${GLM_MEGATRON_CACHE:-/iopsstor/scratch/cscs/xyixuan/.cache/huggingface/nemo_rl_glm51_tp1pp18ep4}
GLM_RESUME_ROOT=${GLM_RESUME_ROOT:-/iopsstor/scratch/cscs/xyixuan/nemo_rl_glm51_cross_allocation_resume/$EXPECTED_HEAD}
GLM_RESUME_CHECKPOINT_DIR=${GLM_RESUME_CHECKPOINT_DIR:-$GLM_RESUME_ROOT/checkpoints}
GLM_RUN_ROOT=${GLM_RUN_ROOT:-$REPO_DIR/logs/glm51_cross_allocation_$EXPECTED_HEAD}
GLM_PHASE_A_TERMINAL=${GLM_PHASE_A_TERMINAL:-$GLM_RUN_ROOT/save/terminal.json}
GLM_NUM_NODES=${GLM_NUM_NODES:-80}
# An explicitly empty reservation submits to the ordinary partition. Keep the
# certified 80-node reservation as the default without making it mandatory for
# larger capacity probes.
GLM_RESERVATION=${GLM_RESERVATION-SD-69241-apertus-1-5-0}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-68719476736}
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-30}
SBATCH_BIN=${SBATCH_BIN:-sbatch}

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ -r "$GLM_CKPT/model.safetensors.index.json" ]] || { echo "Missing GLM checkpoint: $GLM_CKPT" >&2; exit 1; }
CACHE_METADATA=$(find "$GLM_MEGATRON_CACHE" -mindepth 2 -maxdepth 3 -type f -name .metadata -print -quit 2>/dev/null)
[[ -n "$CACHE_METADATA" && -r "$CACHE_METADATA" ]] || { echo "Missing Megatron cache: $GLM_MEGATRON_CACHE" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=untracked)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }
if [[ "$PHASE" == "save" ]]; then
  [[ ! -e "$GLM_RESUME_CHECKPOINT_DIR" ]] || { echo "Refusing existing checkpoint namespace: $GLM_RESUME_CHECKPOINT_DIR" >&2; exit 1; }
  WALLTIME=06:00:00
  JOB_NAME=glm51-ckpt-save
else
  [[ -r "$GLM_RESUME_CHECKPOINT_DIR/step_1/policy/weights/iter_0000000/.metadata" ]] || { echo "Missing complete Phase A checkpoint" >&2; exit 1; }
  [[ -r "$GLM_PHASE_A_TERMINAL" ]] || { echo "Missing green Phase A artifact: $GLM_PHASE_A_TERMINAL" >&2; exit 1; }
  WALLTIME=04:00:00
  JOB_NAME=glm51-ckpt-resume
fi

SBATCH_RESERVATION_ARGS=()
if [[ -n "$GLM_RESERVATION" ]]; then
  SBATCH_RESERVATION_ARGS+=(--reservation="$GLM_RESERVATION")
fi

mkdir -p "$GLM_RUN_ROOT"
export COMMAND=infra/slurm/cscs/autoresearch/run_glm51_cross_allocation_checkpoint.sh
export CONTAINER_ENV
export GLM_CKPT GLM_MEGATRON_CACHE GLM_RESUME_CHECKPOINT_DIR GLM_RESUME_ROOT
export GLM_NUM_NODES GLM_PHASE_A_TERMINAL
export GLM_RECIPE=${GLM_RECIPE:-}
export GLM_EXPERIMENT_DIR=$REPO_DIR
export GLM_EXPECTED_SOURCE_HEAD=$EXPECTED_HEAD
export GLM_RESUME_PHASE=$PHASE
export GLM_RUN_ROOT
export GPUS_PER_NODE=4
export RAY_LOG_SYNC_FREQUENCY RAY_OBJECT_STORE_MEMORY
export RAY_SINGLE_SRUN=1
export BASE_LOG_DIR=$GLM_RUN_ROOT/ray

unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts

cd "$REPO_DIR"
exec "$SBATCH_BIN" \
  --account=infra01 \
  --partition=normal \
  "${SBATCH_RESERVATION_ARGS[@]}" \
  --nodes="$GLM_NUM_NODES" \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --segment=4 \
  --mem=850000M \
  --exclusive \
  --time="$WALLTIME" \
  --job-name="$JOB_NAME" \
  --output="$GLM_RUN_ROOT/slurm_${PHASE}_%j.out" \
  --error="$GLM_RUN_ROOT/slurm_${PHASE}_%j.err" \
  --export=ALL \
  "$REPO_DIR/ray.sub"
