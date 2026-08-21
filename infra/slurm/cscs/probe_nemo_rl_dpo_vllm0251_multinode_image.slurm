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
#SBATCH --job-name=nrl-vllm0251-dpo-mn
#SBATCH --output=logs/nrl_vllm0251_dpo_mn_%j.out
#SBATCH --error=logs/nrl_vllm0251_dpo_mn_%j.err
#SBATCH --time=01:30:00
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
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/nrl_vllm0251_dpo_mn_${SLURM_JOB_ID}}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current}
APERTUS_CKPT=${APERTUS_CKPT:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200}
APERTUS_TOKENIZER=${APERTUS_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Apertus-v1.5-8B-official}
RECIPE=${RECIPE:-/opt/nemo-rl/examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml}
MAX_STEPS=${MAX_STEPS:-2}
GPUS_PER_NODE=4
NUM_NODES=$SLURM_JOB_NUM_NODES
EXPECTED_UNITS=$((NUM_NODES * GPUS_PER_NODE))

[[ -r "$CONTAINER_ENV" ]] || {
  echo "Container EDF not found at CONTAINER_ENV=$CONTAINER_ENV" >&2
  exit 1
}
[[ -f "$XIELU_SITE/xielu/__init__.py" ]] || {
  echo "CUDA xIELU site not found at XIELU_SITE=$XIELU_SITE" >&2
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
export MAX_STEPS RECIPE RUN_DIR XIELU_SITE
export SLURM_EXPORT_ENV=ALL
unset LD_PRELOAD

srun \
  --mpi=pmix \
  --cpu-bind=none \
  --environment="$CONTAINER_ENV" \
  --ntasks-per-node=1 \
  --gpus-per-node="$GPUS_PER_NODE" \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="$XIELU_SITE"
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
    export WANDB_DISABLED=true
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}
    export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-$RUN_DIR/nccl.%h.%p.log}
    export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
    RAY_PORT=6379
    RAY_RES="{\"worker_units\": ${GPUS_PER_NODE}, \"slurm_managed_ray_cluster\": 1}"
    RAY_PORT_ARGS="--min-worker-port=10002 --max-worker-port=11000"

    copy_ray_failure_logs() {
      local rc=$?
      if [[ "$rc" -ne 0 && -d /tmp/ray/session_latest/logs ]]; then
        local failure_dir="$RUN_DIR/ray-failure-${SLURM_PROCID}"
        mkdir -p "$failure_dir"
        cp -a /tmp/ray/session_latest/logs/. "$failure_dir/" || true
      fi
      ray stop --force >/dev/null 2>&1 || true
      exit "$rc"
    }
    trap copy_ray_failure_logs EXIT

    if [[ "$SLURM_PROCID" -eq 0 ]]; then
      ray start --head --port="$RAY_PORT" --node-ip-address="$HEAD_IP" \
        --num-cpus=128 --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
        $RAY_PORT_ARGS --disable-usage-stats
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
      /opt/nemo_rl_venv/bin/python - <<'"'"'PY'"'"'
import ray

from nemo_rl.distributed.virtual_cluster import init_ray

init_ray()
alive_nodes = [node for node in ray.nodes() if node["Alive"]]
assert len(alive_nodes) == 2, alive_nodes
assert int(ray.cluster_resources()["GPU"]) == 8, ray.cluster_resources()
print(f"ray_nodes={len(alive_nodes)}")
print(f"ray_gpus={int(ray.cluster_resources()['"'"'GPU'"'"'])}")
ray.shutdown()
PY

      LOG_DIR="$RUN_DIR/tb"
      RUN_LOG="$RUN_DIR/run.log"
      JSON_METRICS="$RUN_DIR/metrics.json"
      mkdir -p "$LOG_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

      /opt/nemo_rl_venv/bin/python -m examples.run_dpo \
        --config "$RECIPE" \
        dpo.max_num_steps="$MAX_STEPS" \
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
        logger.tensorboard_enabled=true \
        logger.log_dir="$LOG_DIR" \
        logger.wandb_enabled=false \
        logger.monitor_gpus=true \
        checkpointing.enabled=false \
        cluster.gpus_per_node=4 \
        cluster.num_nodes=2 \
        2>&1 | tee "$RUN_LOG"

      if grep -Fq "Apertus XIELU: using eager fallback" "$RUN_LOG"; then
        echo "Multi-node DPO workers unexpectedly used the eager xIELU fallback" >&2
        exit 1
      fi
      /opt/nemo_rl_venv/bin/python /opt/nemo-rl/tests/json_dump_tb_logs.py \
        "$LOG_DIR" --output_path "$JSON_METRICS"
      /opt/nemo_rl_venv/bin/python /opt/nemo-rl/tests/check_metrics.py \
        "$JSON_METRICS" "data[\"train/loss\"][\"$MAX_STEPS\"] < 100"
      echo "baked_dpo_multinode_training=OK"
    else
      until (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do
        sleep 3
      done
      exec 3>&- || true
      ray start --address="${HEAD_IP}:${RAY_PORT}" \
        --num-cpus=128 --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
        $RAY_PORT_ARGS --disable-usage-stats
      while (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do
        exec 3>&- || true
        sleep 10
      done
      ray stop --force || true
    fi
  '
