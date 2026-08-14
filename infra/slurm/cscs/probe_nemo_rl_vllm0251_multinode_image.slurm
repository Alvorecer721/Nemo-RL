#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#SBATCH --account=infra01
#SBATCH --job-name=nrl-vllm0251-mn
#SBATCH --output=logs/nrl_vllm0251_mn_%j.out
#SBATCH --error=logs/nrl_vllm0251_mn_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE
#SBATCH --no-requeue

set -euo pipefail

REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/nrl_vllm0251_mn_${SLURM_JOB_ID}}
APERTUS_CKPT=${APERTUS_CKPT:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200}
APERTUS_TOKENIZER=${APERTUS_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Apertus-v1.5-8B-official}
GPUS_PER_NODE=4
NUM_NODES=$SLURM_JOB_NUM_NODES
EXPECTED_UNITS=$((NUM_NODES * GPUS_PER_NODE))

[[ -r "$CONTAINER_ENV" ]] || {
  echo "Container EDF not found at CONTAINER_ENV=$CONTAINER_ENV" >&2
  exit 1
}
mkdir -p "$RUN_DIR"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$head_node" | awk '{print $1; exit}')
if [[ -z "$HEAD_IP" ]]; then
  HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
fi
[[ -n "$HEAD_IP" ]] || {
  echo "Could not resolve Ray head node $head_node" >&2
  exit 1
}
printf 'head=%s\nhead_ip=%s\nnodes=%s\ncontainer_env=%s\n' \
  "$head_node" "$HEAD_IP" "$NUM_NODES" "$CONTAINER_ENV"

export APERTUS_CKPT APERTUS_TOKENIZER EXPECTED_UNITS GPUS_PER_NODE HEAD_IP
export RUN_DIR
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
    # Preserve per-rank transport diagnostics: a stalled NCCL collective can
    # otherwise leave only the rank 0 pre-init line in the deduplicated Ray log.
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}
    export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-$RUN_DIR/nccl.%h.%p.log}
    # The vLLM 0.25.1 direct PyNccl warm-up stalls with the CSCS OFI plugin for
    # node-spanning communicators. Its PyTorch process-group fallback uses the
    # same NCCL transport and passes the equivalent eight-rank collective.
    # Hopper symmetric-memory rendezvous also requires a shared memory topology
    # that is unavailable across independent Clariden GH200 nodes.
    export VLLM_DISABLE_PYNCCL=${VLLM_DISABLE_PYNCCL:-1}
    export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
    RAY_PORT=6379
    RAY_RES="{\"worker_units\": ${GPUS_PER_NODE}, \"slurm_managed_ray_cluster\": 1}"

    if [[ "$SLURM_PROCID" -eq 0 ]]; then
      trap '"'"'ray stop --force >/dev/null 2>&1 || true'"'"' EXIT
      ray start --head --port="$RAY_PORT" --node-ip-address="$HEAD_IP" \
        --num-cpus=128 --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
        --min-worker-port=53100 --max-worker-port=54100 --disable-usage-stats
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
      cd /opt/nemo-rl
      /opt/nemo_rl_venv/bin/python \
        -m infra.slurm.cscs.probe_nemo_rl_vllm0251_multinode_image \
        2>&1 | tee "$RUN_DIR/run.log"
      grep -Fq "multinode_vllm_startup=OK" "$RUN_DIR/run.log"
    else
      until (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do
        sleep 3
      done
      exec 3>&- || true
      ray start --address="${HEAD_IP}:${RAY_PORT}" \
        --num-cpus=128 --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
        --min-worker-port=53100 --max-worker-port=54100 --disable-usage-stats
      while (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do
        exec 3>&- || true
        sleep 10
      done
      ray stop --force || true
    fi
  '
