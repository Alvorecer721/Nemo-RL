#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=${REFIT_SC_REPO_DIR:?}
EXPECTED_HEAD=${REFIT_SC_EXPECTED_HEAD:?}
CONTAINER_ENV=${CONTAINER_ENV:?}
RUN_ROOT=${REFIT_SC_RUN_ROOT:?}
RUN_DIR=$RUN_ROOT/${SLURM_JOB_ID:?}
ARM_TIMEOUT_S=${REFIT_SC_ARM_TIMEOUT_S:-900}
CONTAINER_REPO_DIR=/opt/nemo-rl

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || {
  echo "Source HEAD changed after submission" >&2
  exit 1
}
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
if ! mkdir "$RUN_DIR"; then
  echo "Refusing to reuse SingleController PP2 run directory: $RUN_DIR" >&2
  exit 1
fi

printf 'job_id=%s\nrepo=%s\nhead=%s\nrun_dir=%s\n' \
  "$SLURM_JOB_ID" "$REPO_DIR" "$EXPECTED_HEAD" "$RUN_DIR"

unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts

printf 'arm\tstreams\timplicit_order\texit_code\tstatus\tsync_count\n' >"$RUN_DIR/results.tsv"

run_arm() {
  local arm=$1
  local streams=$2
  local implicit_order=$3
  local harness_log=$RUN_DIR/${arm}.harness.log
  local arm_dir=$RUN_DIR/$arm
  local exit_code status sync_count

  echo "arm_start name=$arm streams=$streams implicit_order=$implicit_order"
  set +e
  srun \
    --cpu-bind=none \
    --environment="$CONTAINER_ENV" \
    --container-mounts="$REPO_DIR:$CONTAINER_REPO_DIR" \
    --ntasks=1 \
    --gpus-per-task=3 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    /usr/bin/env \
      REFIT_SC_REPO_DIR="$CONTAINER_REPO_DIR" \
      REFIT_SC_EXPECTED_HEAD="$EXPECTED_HEAD" \
      REFIT_SC_ARM_DIR="$arm_dir" \
      REFIT_SC_ARM_NAME="$arm" \
      REFIT_SC_STEPS="${REFIT_SC_STEPS:-12}" \
      REFIT_SC_ARM_TIMEOUT_S="$ARM_TIMEOUT_S" \
      NRL_REFIT_NUM_STREAMS="$streams" \
      NCCL_LAUNCH_ORDER_IMPLICIT="$implicit_order" \
      /bin/bash "$CONTAINER_REPO_DIR/infra/slurm/cscs/autoresearch/run_refit_sc_pp2_arm.sh" \
      >"$harness_log" 2>&1
  exit_code=$?
  set -e

  status=$(awk -F= '$1 == "status" {print $2}' "$arm_dir/summary.txt" 2>/dev/null || true)
  sync_count=$(awk -F= '$1 == "sync_count" {print $2}' "$arm_dir/summary.txt" 2>/dev/null || true)
  status=${status:-error}
  sync_count=${sync_count:-0}
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$streams" "$implicit_order" "$exit_code" "$status" "$sync_count" \
    >>"$RUN_DIR/results.tsv"
  echo "arm_end name=$arm exit_code=$exit_code status=$status sync_count=$sync_count log=$arm_dir/run.log"
  LAST_ARM_STATUS=$status
}

run_arm streams2_implicit0 2 0
if [[ "$LAST_ARM_STATUS" == error ]]; then
  echo invalid_baseline >"$RUN_DIR/verdict.txt"
  echo "REFIT_SC_PP2_MATRIX=invalid_baseline"
  cat "$RUN_DIR/results.tsv"
  exit 1
fi
run_arm streams1_implicit0 1 0
if [[ "$LAST_ARM_STATUS" == error ]]; then
  echo invalid_controls >"$RUN_DIR/verdict.txt"
  echo "REFIT_SC_PP2_MATRIX=invalid_controls"
  cat "$RUN_DIR/results.tsv"
  exit 1
fi
run_arm streams2_implicit1 2 1

arm_a=$(awk -F '\t' '$1 == "streams2_implicit0" {print $5}' "$RUN_DIR/results.tsv")
arm_b=$(awk -F '\t' '$1 == "streams1_implicit0" {print $5}' "$RUN_DIR/results.tsv")
arm_c=$(awk -F '\t' '$1 == "streams2_implicit1" {print $5}' "$RUN_DIR/results.tsv")

if [[ "$arm_b" != pass || "$arm_c" != pass ]]; then
  verdict=invalid_controls
elif [[ "$arm_a" == timeout ]]; then
  verdict=reproduced
elif [[ "$arm_a" == pass ]]; then
  verdict=not_reproduced
else
  verdict=invalid_baseline
fi

printf '%s\n' "$verdict" >"$RUN_DIR/verdict.txt"
echo "REFIT_SC_PP2_MATRIX=$verdict"
cat "$RUN_DIR/results.tsv"

if [[ "$verdict" == invalid_* ]]; then
  exit 1
fi
