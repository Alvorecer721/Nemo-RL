#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
EXPECTED_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
GLM_CKPT=${GLM_CKPT:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.1}
GLM_MEGATRON_CACHE=${GLM_MEGATRON_CACHE:-/iopsstor/scratch/cscs/xyixuan/.cache/huggingface/nemo_rl_glm51_tp1pp18ep4}
GLM_RECIPE=${GLM_RECIPE:-$REPO_DIR/examples/configs/recipes/llm/autoresearch/grpo-glm5.1-80n4g-megatron-tp2pp18ep16-async-vllm-tp32-r3-10step.yaml}
GLM_RUN_ROOT=${GLM_RUN_ROOT:-/iopsstor/scratch/cscs/xyixuan/nemo_rl_glm51_r3_10step/$EXPECTED_HEAD}
GLM_RESERVATION=${GLM_RESERVATION-SD-69241-apertus-1-5-0}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-68719476736}
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-30}
SBATCH_BIN=${SBATCH_BIN:-sbatch}

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ -r "$GLM_RECIPE" ]] || { echo "Missing recipe: $GLM_RECIPE" >&2; exit 1; }
[[ -r "$GLM_CKPT/model.safetensors.index.json" ]] || { echo "Missing GLM checkpoint: $GLM_CKPT" >&2; exit 1; }
CACHE_METADATA=$(find "$GLM_MEGATRON_CACHE" -mindepth 2 -maxdepth 3 -type f -name .metadata -print -quit 2>/dev/null)
[[ -n "$CACHE_METADATA" && -r "$CACHE_METADATA" ]] || { echo "Missing Megatron cache: $GLM_MEGATRON_CACHE" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=untracked)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }
SUBMODULE_STATUS=$(git -C "$REPO_DIR" submodule status --recursive)
INVALID_SUBMODULES=$(printf '%s\n' "$SUBMODULE_STATUS" | awk 'substr($0, 1, 1) == "-" || substr($0, 1, 1) == "+"')
[[ -z "$INVALID_SUBMODULES" ]] || {
  echo "Submodules are uninitialized or do not match the recorded gitlinks: $INVALID_SUBMODULES" >&2
  exit 1
}

SBATCH_RESERVATION_ARGS=()
if [[ -n "$GLM_RESERVATION" ]]; then
  SBATCH_RESERVATION_ARGS+=(--reservation="$GLM_RESERVATION")
fi

mkdir -p "$GLM_RUN_ROOT"
export COMMAND=infra/slurm/cscs/autoresearch/run_glm51_r3_10step.sh
export CONTAINER_ENV GLM_CKPT GLM_MEGATRON_CACHE GLM_RECIPE
export GLM_EXPERIMENT_DIR=$REPO_DIR
export GLM_EXPECTED_SOURCE_HEAD=$EXPECTED_HEAD
export GLM_RUN_DIR=$GLM_RUN_ROOT
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
  --nodes=80 \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --segment=4 \
  --mem=850000M \
  --exclusive \
  --time=08:00:00 \
  --job-name=glm51-r3-10step \
  --output="$GLM_RUN_ROOT/slurm_%j.out" \
  --error="$GLM_RUN_ROOT/slurm_%j.err" \
  --export=ALL \
  "$REPO_DIR/ray.sub"
