#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=${REFIT_REPRO_REPO_DIR:?}
EXPECTED_HEAD=${REFIT_REPRO_EXPECTED_HEAD:?}
CONTAINER_ENV=${CONTAINER_ENV:?}
RUN_ROOT=${REFIT_REPRO_RUN_ROOT:?}
RUN_DIR=$RUN_ROOT/${SLURM_JOB_ID:?}
ITERATIONS=${REFIT_REPRO_ITERATIONS:-400}
TRANSFERS=${REFIT_REPRO_TRANSFERS_PER_STAGE:-32}
TENSOR_MIB=${REFIT_REPRO_TENSOR_MIB:-8}
ARM_TIMEOUT_S=${REFIT_REPRO_ARM_TIMEOUT_S:-600}

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || {
  echo "Source HEAD changed after submission" >&2
  exit 1
}
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
if ! mkdir "$RUN_DIR"; then
  echo "Refusing to reuse refit-order run directory: $RUN_DIR" >&2
  exit 1
fi

printf 'job_id=%s\nrepo=%s\nhead=%s\nrun_dir=%s\n' \
  "$SLURM_JOB_ID" "$REPO_DIR" "$EXPECTED_HEAD" "$RUN_DIR"

unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts

printf 'arm\tstreams\timplicit_order\texit_code\tstatus\n' >"$RUN_DIR/results.tsv"

run_arm() {
  local arm=$1
  local streams=$2
  local implicit_order=$3
  local arm_log=$RUN_DIR/${arm}.log
  local exit_code
  local status

  echo "arm_start name=$arm streams=$streams implicit_order=$implicit_order"
  set +e
  srun \
    --cpu-bind=none \
    --environment="$CONTAINER_ENV" \
    --ntasks=1 \
    --gpus-per-task=4 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    /bin/bash -lc '
      set -euo pipefail
      export PYTHONPATH='"$REPO_DIR"'
      export PYTHONUNBUFFERED=1
      export NCCL_DEBUG=INFO
      export NCCL_DEBUG_SUBSYS=INIT,COLL
      export NCCL_LAUNCH_ORDER_IMPLICIT='"$implicit_order"'
      export NRL_REFIT_NUM_STREAMS='"$streams"'
      export NRL_XFERDTENSOR_PYTHON=1
      cd '"$REPO_DIR"'
      test "$(git rev-parse HEAD)" = '"$EXPECTED_HEAD"'
      test -z "$(git status --porcelain --untracked-files=no --ignore-submodules=all)"
      timeout --signal=TERM --kill-after=15s '"$ARM_TIMEOUT_S"'s \
        /opt/nemo_rl_venv/bin/python -m torch.distributed.run \
          --standalone --nproc-per-node=3 \
          tests/functional/refit_nccl_ordering_repro.py \
          --streams '"$streams"' \
          --iterations '"$ITERATIONS"' \
          --transfers-per-stage '"$TRANSFERS"' \
          --tensor-mib '"$TENSOR_MIB"'
    ' >"$arm_log" 2>&1
  exit_code=$?
  set -e

  if grep -Fq 'REFIT_ORDER_REPRO=PASS' "$arm_log"; then
    status=pass
  elif grep -Fq 'deadline exceeded' "$arm_log" || \
       [[ $exit_code -eq 124 || $exit_code -eq 137 || $exit_code -eq 143 ]]; then
    status=timeout
  else
    status=error
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$streams" "$implicit_order" "$exit_code" "$status" \
    >>"$RUN_DIR/results.tsv"
  echo "arm_end name=$arm exit_code=$exit_code status=$status log=$arm_log"
}

run_arm streams2_implicit0 2 0
run_arm streams1_implicit0 1 0
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
echo "REFIT_ORDER_MATRIX=$verdict"
cat "$RUN_DIR/results.tsv"

if [[ "$verdict" == invalid_* ]]; then
  exit 1
fi
