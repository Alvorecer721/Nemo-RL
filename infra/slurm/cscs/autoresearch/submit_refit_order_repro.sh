#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
EXPECTED_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
RUN_ROOT=${REFIT_REPRO_RUN_ROOT:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/nemo_rl_refit_order_repro/$EXPECTED_HEAD}
SBATCH_LOG_ROOT=${REFIT_REPRO_LOG_ROOT:-$REPO_DIR/.tmp/slurm-logs/refit-order-repro/$EXPECTED_HEAD}
RESERVATION=${REFIT_REPRO_RESERVATION-SD-69241-apertus-1-5-0}
SBATCH_BIN=${SBATCH_BIN:-sbatch}

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }

reservation_args=()
if [[ -n "$RESERVATION" ]]; then
  reservation_args+=(--reservation="$RESERVATION")
fi

mkdir -p "$SBATCH_LOG_ROOT"
export CONTAINER_ENV
export REFIT_REPRO_REPO_DIR=$REPO_DIR
export REFIT_REPRO_EXPECTED_HEAD=$EXPECTED_HEAD
export REFIT_REPRO_RUN_ROOT=$RUN_ROOT

unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts

cd "$REPO_DIR"
exec "$SBATCH_BIN" \
  --parsable \
  --account=infra01 \
  --partition=normal \
  "${reservation_args[@]}" \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --segment=4 \
  --cpus-per-task=64 \
  --mem=200000M \
  --exclusive \
  --time=00:45:00 \
  --job-name=refit-order-repro \
  --output="$SBATCH_LOG_ROOT/slurm_%j.out" \
  --error="$SBATCH_LOG_ROOT/slurm_%j.err" \
  --export=ALL \
  "$REPO_DIR/infra/slurm/cscs/autoresearch/run_refit_order_repro.sh"
