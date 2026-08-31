#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=${REFIT_SC_REPO_DIR:?}
EXPECTED_HEAD=${REFIT_SC_EXPECTED_HEAD:?}
CONTAINER_ENV=${CONTAINER_ENV:?}
RUN_ROOT=${REFIT_SC_RUN_ROOT:?}
RUN_DIR=$RUN_ROOT/${SLURM_JOB_ID:?}
ARM_TIMEOUT_S=${REFIT_SC_ARM_TIMEOUT_S:-1200}
[[ "$ARM_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || {
  echo "REFIT_SC_ARM_TIMEOUT_S must be a positive integer" >&2
  exit 1
}
STEP_TIMEOUT_S=$((ARM_TIMEOUT_S + 300))
RUNTIME_REPO_DIR=/opt/nemo-rl
SOURCE_MOUNTS=$REPO_DIR/nemo_rl:$RUNTIME_REPO_DIR/nemo_rl
SOURCE_MOUNTS+=,$REPO_DIR/nemo_rl_apertus:$RUNTIME_REPO_DIR/nemo_rl_apertus
SOURCE_MOUNTS+=,$REPO_DIR/examples:$RUNTIME_REPO_DIR/examples
SOURCE_MOUNTS+=,$REPO_DIR/infra:$RUNTIME_REPO_DIR/infra
SOURCE_MOUNTS+=,$REPO_DIR/tests:$RUNTIME_REPO_DIR/tests
SOURCE_MOUNTS+=,$REPO_DIR/tools:$RUNTIME_REPO_DIR/tools
GPUS_PER_NODE=4
EXPECTED_UNITS=8

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || {
  echo "Source HEAD changed after submission" >&2
  exit 1
}
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
if ! mkdir "$RUN_DIR"; then
  echo "Refusing to reuse SingleController PP4 run directory: $RUN_DIR" >&2
  exit 1
fi

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$head_node" | awk '{print $1; exit}')
if [[ -z "$HEAD_IP" ]]; then
  HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
fi
[[ -n "$HEAD_IP" ]] || { echo "Could not resolve Ray head node $head_node" >&2; exit 1; }

printf 'job_id=%s\nrepo=%s\nhead=%s\nrun_dir=%s\nray_head=%s\n' \
  "$SLURM_JOB_ID" "$REPO_DIR" "$EXPECTED_HEAD" "$RUN_DIR" "$HEAD_IP"

unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts
unset LD_PRELOAD
export SLURM_EXPORT_ENV=ALL

printf 'arm\tstreams\timplicit_order\texit_code\tstatus\tsync_count\n' >"$RUN_DIR/results.tsv"

run_arm() {
  local arm=$1
  local streams=$2
  local implicit_order=$3
  local ray_port=$4
  local harness_log=$RUN_DIR/${arm}.harness.log
  local arm_dir=$RUN_DIR/$arm
  local exit_code status sync_count

  echo "arm_start name=$arm streams=$streams implicit_order=$implicit_order"
  set +e
  timeout --signal=TERM --kill-after=30s "${STEP_TIMEOUT_S}s" srun \
    --mpi=pmix \
    --cpu-bind=none \
    --environment="$CONTAINER_ENV" \
    --container-mounts="$SOURCE_MOUNTS" \
    --nodes=2 \
    --ntasks-per-node=1 \
    --gpus-per-node="$GPUS_PER_NODE" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    /usr/bin/env \
      REFIT_SC_SOURCE_REPO_DIR="$REPO_DIR" \
      REFIT_SC_EXPECTED_HEAD="$EXPECTED_HEAD" \
      REFIT_SC_ARM_DIR="$arm_dir" \
      REFIT_SC_ARM_NAME="$arm" \
      REFIT_SC_STEPS="${REFIT_SC_STEPS:-12}" \
      REFIT_SC_ARM_TIMEOUT_S="$ARM_TIMEOUT_S" \
      REFIT_SC_PP_SIZE=4 \
      REFIT_SC_GEN_TP_SIZE=4 \
      REFIT_SC_GEN_NUM_NODES=1 \
      REFIT_SC_GEN_GPUS_PER_NODE=4 \
      REFIT_SC_CLUSTER_NUM_NODES=2 \
      REFIT_SC_CLUSTER_GPUS_PER_NODE=4 \
      NRL_REFIT_NUM_STREAMS="$streams" \
      NCCL_LAUNCH_ORDER_IMPLICIT="$implicit_order" \
      REFIT_SC_RAY_HEAD_IP="$HEAD_IP" \
      REFIT_SC_RAY_PORT="$ray_port" \
      REFIT_SC_EXPECTED_UNITS="$EXPECTED_UNITS" \
      REFIT_SC_RUNTIME_REPO_DIR="$RUNTIME_REPO_DIR" \
      /bin/bash -lc '
        set -euo pipefail
        export PYTHONPATH=$REFIT_SC_RUNTIME_REPO_DIR
        export PYTHONUNBUFFERED=1
        export NEMO_RL_VENV_DIR=/opt/ray_venvs
        unset NEMO_RL_PY_EXECUTABLES_SYSTEM
        export RAY_DEDUP_LOGS=0
        export RAY_raylet_start_wait_time_s=120
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
        export RAY_ADDRESS=${REFIT_SC_RAY_HEAD_IP}:${REFIT_SC_RAY_PORT}
        RAY_RES="{\"worker_units\": 4, \"slurm_managed_ray_cluster\": 1}"
        RAY_PORT_ARGS="--min-worker-port=10002 --max-worker-port=11000"

        cleanup_ray() {
          local rc=$?
          if [[ "$rc" -ne 0 && -d /tmp/ray/session_latest/logs ]]; then
            local failure_dir=$REFIT_SC_ARM_DIR/ray-failure-$SLURM_PROCID
            mkdir -p "$failure_dir"
            cp -a /tmp/ray/session_latest/logs/. "$failure_dir/" || true
          fi
          ray stop --force >/dev/null 2>&1 || true
          exit "$rc"
        }
        trap cleanup_ray EXIT

        if [[ "$SLURM_PROCID" -eq 0 ]]; then
          ray start --head --port="$REFIT_SC_RAY_PORT" \
            --node-ip-address="$REFIT_SC_RAY_HEAD_IP" \
            --num-cpus=64 --num-gpus=4 --resources="$RAY_RES" \
            $RAY_PORT_ARGS --disable-usage-stats
          units=0
          for _ in $(seq 1 120); do
            units=$(ray status 2>/dev/null | sed -n \
              "s#.*/\([0-9]*\)\.[0-9]* worker_units.*#\1#p" | head -1)
            [[ "${units:-0}" -eq "$REFIT_SC_EXPECTED_UNITS" ]] && break
            sleep 2
          done
          [[ "${units:-0}" -eq "$REFIT_SC_EXPECTED_UNITS" ]] || {
            echo "Ray cluster incomplete: ${units:-0}/$REFIT_SC_EXPECTED_UNITS worker units" >&2
            exit 1
          }
          /bin/bash "$REFIT_SC_RUNTIME_REPO_DIR/infra/slurm/cscs/autoresearch/run_refit_sc_pp2_arm.sh"
        else
          until (exec 3<>"/dev/tcp/${REFIT_SC_RAY_HEAD_IP}/${REFIT_SC_RAY_PORT}") 2>/dev/null; do
            sleep 2
          done
          exec 3>&- || true
          ray start --address="${REFIT_SC_RAY_HEAD_IP}:${REFIT_SC_RAY_PORT}" \
            --num-cpus=64 --num-gpus=4 --resources="$RAY_RES" \
            $RAY_PORT_ARGS --disable-usage-stats
          while (exec 3<>"/dev/tcp/${REFIT_SC_RAY_HEAD_IP}/${REFIT_SC_RAY_PORT}") 2>/dev/null; do
            exec 3>&- || true
            sleep 5
          done
        fi
      ' >"$harness_log" 2>&1
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

run_arm streams2_implicit0 2 0 6379
if [[ "$LAST_ARM_STATUS" == error ]]; then
  echo invalid_baseline >"$RUN_DIR/verdict.txt"
  echo "REFIT_SC_PP4_2N_MATRIX=invalid_baseline"
  cat "$RUN_DIR/results.tsv"
  exit 1
fi
run_arm streams1_implicit0 1 0 6380
if [[ "$LAST_ARM_STATUS" == error ]]; then
  echo invalid_controls >"$RUN_DIR/verdict.txt"
  echo "REFIT_SC_PP4_2N_MATRIX=invalid_controls"
  cat "$RUN_DIR/results.tsv"
  exit 1
fi
run_arm streams2_implicit1 2 1 6381

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
echo "REFIT_SC_PP4_2N_MATRIX=$verdict"
cat "$RUN_DIR/results.tsv"

if [[ "$verdict" == invalid_* ]]; then
  exit 1
fi
