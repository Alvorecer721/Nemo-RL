#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=glm5-vllm-tp32
#SBATCH --output=logs/glm51_exact_vllm_tp32_%j.out
#SBATCH --error=logs/glm51_exact_vllm_tp32_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE
#SBATCH --no-requeue

set -euo pipefail

REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/infra/slurm/cscs/autoresearch/glm51_exact_image.toml}
PROBE_SCRIPT=${PROBE_SCRIPT:-$REPO_DIR/infra/slurm/cscs/autoresearch/probe_glm51_vllm_tp32_async.py}
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/glm51_exact_vllm_tp32_${SLURM_JOB_ID}}
GLM_CKPT=${GLM_CKPT:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.1}
EXPECTED_SOURCE_HEAD=${EXPECTED_SOURCE_HEAD:-084ade845b8421ab82dcda1849d913da517f194e}
GPUS_PER_NODE=4
NUM_NODES=$SLURM_JOB_NUM_NODES
EXPECTED_UNITS=$((NUM_NODES * GPUS_PER_NODE))

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing CONTAINER_ENV=$CONTAINER_ENV" >&2; exit 1; }
[[ -r "$PROBE_SCRIPT" ]] || { echo "Missing PROBE_SCRIPT=$PROBE_SCRIPT" >&2; exit 1; }
[[ -r "$GLM_CKPT/model.safetensors.index.json" ]] || { echo "Missing GLM checkpoint at $GLM_CKPT" >&2; exit 1; }
mkdir -p "$RUN_DIR"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$head_node" | awk '{print $1; exit}')
if [[ -z "$HEAD_IP" ]]; then
  HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
fi
[[ -n "$HEAD_IP" ]] || { echo "Could not resolve Ray head node $head_node" >&2; exit 1; }
printf 'head=%s\nhead_ip=%s\nnodes=%s\ncontainer_env=%s\ncheckpoint=%s\n' \
  "$head_node" "$HEAD_IP" "$NUM_NODES" "$CONTAINER_ENV" "$GLM_CKPT"

export GLM_CKPT EXPECTED_SOURCE_HEAD EXPECTED_UNITS GPUS_PER_NODE NUM_NODES HEAD_IP PROBE_SCRIPT RUN_DIR
export SLURM_EXPORT_ENV=ALL
unset LD_PRELOAD

srun \
  --mpi=pmix \
  --environment="$CONTAINER_ENV" \
  --ntasks-per-node=1 \
  --gpus-per-node="$GPUS_PER_NODE" \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    unset PYTHONPATH
    export PYTHONUNBUFFERED=1
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export RAY_DEDUP_LOGS=0
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-$RUN_DIR/nccl.%h.%p.log}
    export VLLM_DISABLE_PYNCCL=1
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0
    [[ "${NEMO_RL_COMMIT:-}" == "$EXPECTED_SOURCE_HEAD" ]] || {
      echo "Exact-image source mismatch: expected $EXPECTED_SOURCE_HEAD, got ${NEMO_RL_COMMIT:-<unset>}" >&2
      exit 1
    }
    RAY_PORT=1200
    RAY_RES="{\"worker_units\": ${GPUS_PER_NODE}, \"slurm_managed_ray_cluster\": 1}"

    telemetry_file="$RUN_DIR/gpu.$(hostname).csv"
    nvidia-smi \
      --query-gpu=timestamp,name,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits \
      --loop=2 > "$telemetry_file" 2>&1 &
    telemetry_pid=$!
    cleanup() {
      kill "$telemetry_pid" >/dev/null 2>&1 || true
      if [[ "$SLURM_PROCID" -eq 0 ]]; then
        ray stop --force >/dev/null 2>&1 || true
      fi
    }
    trap cleanup EXIT

    if [[ "$SLURM_PROCID" -eq 0 ]]; then
      ray start --head --port="$RAY_PORT" --node-ip-address="$HEAD_IP" \
        --num-cpus=128 --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
        --min-worker-port=2000 --max-worker-port=2999 --disable-usage-stats
      units=0
      for _ in $(seq 1 180); do
        units=$(ray status 2>/dev/null | sed -n \
          "s#.*/\([0-9]*\)\.[0-9]* worker_units.*#\1#p" | head -1)
        [[ "${units:-0}" -eq "$EXPECTED_UNITS" ]] && break
        sleep 5
      done
      [[ "${units:-0}" -eq "$EXPECTED_UNITS" ]] || {
        echo "Ray cluster incomplete: ${units:-0}/$EXPECTED_UNITS worker units" >&2
        exit 1
      }
      PYTHONPATH=/opt/nemo-rl \
        /opt/nemo_rl_venv/bin/python "$PROBE_SCRIPT" \
        2>&1 | tee "$RUN_DIR/run.log"
      grep -Fq "glm5p1_vllm_tp32_async=OK" "$RUN_DIR/run.log"
    else
      until (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do sleep 3; done
      exec 3>&- || true
      ray start --address="${HEAD_IP}:${RAY_PORT}" \
        --num-cpus=128 --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
        --min-worker-port=2000 --max-worker-port=2999 --disable-usage-stats
      while (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do
        exec 3>&- || true
        sleep 10
      done
      ray stop --force || true
    fi
  '
