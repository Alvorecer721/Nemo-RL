#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

classify_arm_status() {
  local exit_code=$1
  local validation_exit_code=$2
  local arm_log=$3
  if [[ $exit_code -eq 0 && $validation_exit_code -eq 0 ]]; then
    echo pass
  elif [[ $exit_code -eq 124 ]] || grep -Fq 'deadline exceeded' "$arm_log"; then
    echo timeout
  else
    echo error
  fi
}

if [[ ${1:-} == --classify-status ]]; then
  [[ $# -eq 4 ]] || { echo "usage: $0 --classify-status EXIT VALIDATION_EXIT LOG" >&2; exit 2; }
  classify_arm_status "$2" "$3" "$4"
  exit 0
fi

REPO_DIR=${REFIT_REPRO_REPO_DIR:?}
EXPECTED_HEAD=${REFIT_REPRO_EXPECTED_HEAD:?}
CONTAINER_ENV=${REFIT_REPRO_CONTAINER_ENV:?}
RUN_ROOT=${REFIT_REPRO_RUN_ROOT:?}
RUN_DIR=$RUN_ROOT/${SLURM_JOB_ID:?}
ITERATIONS=${REFIT_REPRO_ITERATIONS:?}
TRANSFERS=${REFIT_REPRO_TRANSFERS_PER_STAGE:?}
TENSOR_MIB=${REFIT_REPRO_TENSOR_MIB:?}
ITERATION_TIMEOUT_S=${REFIT_REPRO_ITERATION_TIMEOUT_S:?}
COORDINATION_TIMEOUT_S=${REFIT_REPRO_COORDINATION_TIMEOUT_S:?}
ARM_TIMEOUT_S=${REFIT_REPRO_ARM_TIMEOUT_S:?}

# A submission from another allocation can carry a CPU mask that does not fit
# this job. Pyxis variables from the submitting shell can likewise select an
# unrelated container or SSH hook, so remove both spellings before every srun.
while IFS= read -r name; do
  unset "$name"
done < <(compgen -A variable SLURM_CPU_BIND)
while IFS='=' read -r name _; do
  case "$name" in
    SLURM_SPANK_* | _SLURM_SPANK_*) unset "$name" ;;
  esac
done < <(env)

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ $(git -c core.fsmonitor=false -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || {
  echo "Source HEAD changed after submission" >&2
  exit 1
}
SOURCE_STATUS=$(git -c core.fsmonitor=false -C "$REPO_DIR" status --porcelain=v1 --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }

mkdir -p "$RUN_ROOT"
if ! mkdir "$RUN_DIR"; then
  echo "Refusing to reuse refit ordering run directory: $RUN_DIR" >&2
  exit 1
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#NODES[@]} -eq 2 ]] || { echo "Expected two nodes, got ${#NODES[@]}" >&2; exit 1; }
MASTER_ADDR=${NODES[0]}
MASTER_PORT=${REFIT_REPRO_MASTER_PORT:-29571}

printf 'job_id=%s\nrepo=%s\nhead=%s\ncontainer_env=%s\nrun_dir=%s\nnodes=%s,%s\n' \
  "$SLURM_JOB_ID" "$REPO_DIR" "$EXPECTED_HEAD" "$CONTAINER_ENV" "$RUN_DIR" \
  "${NODES[0]}" "${NODES[1]}"

export AP_EXPERIMENT_DIR=$REPO_DIR
export AP_EXPECTED_SOURCE_HEAD=$EXPECTED_HEAD
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export UV_OFFLINE=1
unset NRL_IGNORE_VERSION_MISMATCH

# Importing nemo_rl runs the source/dependency fingerprint check; the helper
# also validates all baked worker interpreters against the matching image.
srun \
  --export=ALL \
  --cpu-bind=none \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=4 \
  --cpus-per-task=16 \
  --environment="$CONTAINER_ENV" \
  /bin/bash "$REPO_DIR/infra/slurm/cscs/autoresearch/preflight_gsm8k_runtime.sh"

printf 'arm\tstages\tstreams\texit_code\tvalidation_exit_code\tstatus\n' >"$RUN_DIR/results.tsv"

run_arm() {
  local arm=$1
  local stages=$2
  local streams=$3
  local port=$4
  local world_size=$((stages + 1))
  local arm_log=$RUN_DIR/${arm}.log
  local exit_code
  local validation_exit_code=-1
  local status

  echo "arm_start name=$arm stages=$stages streams=$streams world_size=$world_size"
  export MASTER_ADDR
  export MASTER_PORT=$port
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
  export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}
  export NCCL_LAUNCH_ORDER_IMPLICIT=0
  export NRL_REFIT_NUM_STREAMS=$streams
  export NRL_XFERDTENSOR_PYTHON=1
  unset NRL_XFERDTENSOR_GOLDEN

  set +e
  timeout --signal=TERM --kill-after=30s "${ARM_TIMEOUT_S}s" \
    srun \
      --export=ALL \
      --kill-on-bad-exit=1 \
      --cpu-bind=none \
      --gpu-bind=none \
      --distribution=block:block,Pack \
      --nodes=2 \
      --ntasks="$world_size" \
      --ntasks-per-node="$stages" \
      --gpus-per-node=4 \
      --cpus-per-task=16 \
      --environment="$CONTAINER_ENV" \
      /bin/bash -lc '
        set -euo pipefail
        export RANK=$SLURM_PROCID
        export WORLD_SIZE=$SLURM_NTASKS
        export LOCAL_RANK=$SLURM_LOCALID
        export PYTHONPATH='"$REPO_DIR"'
        if (( SLURM_PROCID < '"$stages"' )); then
          [[ $SLURM_NODEID -eq 0 && $SLURM_LOCALID -eq $SLURM_PROCID ]]
        else
          [[ $SLURM_PROCID -eq '"$stages"' ]]
          [[ $SLURM_NODEID -eq 1 && $SLURM_LOCALID -eq 0 ]]
        fi
        cd '"$REPO_DIR"'
        /root/.local/bin/uv run --no-config --no-project --offline \
          --python /opt/nemo_rl_venv/bin/python python \
          tests/functional/refit_nccl_ordering_repro.py \
          --stages '"$stages"' \
          --streams '"$streams"' \
          --iterations '"$ITERATIONS"' \
          --transfers-per-stage '"$TRANSFERS"' \
          --tensor-mib '"$TENSOR_MIB"' \
          --iteration-timeout-s '"$ITERATION_TIMEOUT_S"' \
          --coordination-timeout-s '"$COORDINATION_TIMEOUT_S"'
      ' >"$arm_log" 2>&1
  exit_code=$?
  set -e

  if [[ $exit_code -eq 0 ]]; then
    set +e
    timeout --signal=TERM --kill-after=5s 60s \
      srun \
        --export=ALL \
        --cpu-bind=none \
        --nodes=1 \
        --ntasks=1 \
        --gpus-per-node=4 \
        --cpus-per-task=16 \
        --environment="$CONTAINER_ENV" \
        /bin/bash -lc '
          set -euo pipefail
          export PYTHONPATH='"$REPO_DIR"'
          cd '"$REPO_DIR"'
          /root/.local/bin/uv run --no-config --no-project --offline \
            --python /opt/nemo_rl_venv/bin/python python \
            tests/functional/refit_nccl_ordering_repro.py \
            --stages '"$stages"' \
            --streams '"$streams"' \
            --iterations '"$ITERATIONS"' \
            --transfers-per-stage '"$TRANSFERS"' \
            --tensor-mib '"$TENSOR_MIB"' \
            --iteration-timeout-s '"$ITERATION_TIMEOUT_S"' \
            --coordination-timeout-s '"$COORDINATION_TIMEOUT_S"' \
            --validate-log '"$arm_log"'
        ' >>"$arm_log" 2>&1
    validation_exit_code=$?
    set -e
  fi

  status=$(classify_arm_status "$exit_code" "$validation_exit_code" "$arm_log")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$stages" "$streams" "$exit_code" "$validation_exit_code" "$status" \
    >>"$RUN_DIR/results.tsv"
  echo "arm_end name=$arm exit_code=$exit_code validation_exit_code=$validation_exit_code status=$status log=$arm_log"
}

run_arm pp2_streams1 2 1 "$MASTER_PORT"
run_arm pp2_streams2 2 2 "$((MASTER_PORT + 1))"
run_arm pp4_streams1 4 1 "$((MASTER_PORT + 2))"
run_arm pp4_streams2 4 2 "$((MASTER_PORT + 3))"

cat "$RUN_DIR/results.tsv"
if awk -F '\t' 'NR > 1 && $6 != "pass" { failed = 1 } END { exit failed }' "$RUN_DIR/results.tsv"; then
  echo "REFIT_ORDER_REPRO_MATRIX=PASS"
else
  echo "REFIT_ORDER_REPRO_MATRIX=FAIL" >&2
  exit 1
fi
