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
#SBATCH --job-name=nrl-vllm0251-grpo-mn
#SBATCH --output=logs/nrl_vllm0251_grpo_mn_%j.out
#SBATCH --error=logs/nrl_vllm0251_grpo_mn_%j.err
#SBATCH --time=02:00:00
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
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/nrl_vllm0251_grpo_mn_${SLURM_JOB_ID}}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current}
RECIPE=${RECIPE:-/opt/nemo-rl/examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-probe.yaml}
MAX_STEPS=${MAX_STEPS:-10}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
NUM_PROMPTS=${NUM_PROMPTS:-4}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
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

export EXPECTED_UNITS GPUS_PER_NODE HEAD_IP MAX_NEW_TOKENS MAX_STEPS
export NUM_GENERATIONS NUM_PROMPTS RECIPE RUN_DIR XIELU_SITE
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
      /opt/nemo_rl_venv/bin/python - <<'PY'
from pathlib import Path

import ray

from nemo_rl.distributed.virtual_cluster import init_ray

init_ray()
alive_nodes = [node for node in ray.nodes() if node["Alive"]]
assert len(alive_nodes) == 2, alive_nodes
resources = ray.cluster_resources()
gpu_count = int(resources["GPU"])
assert gpu_count == 8, resources
for actor in (
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
):
    venv = Path("/opt/ray_venvs") / actor
    assert (venv / "bin/python").is_file(), venv
    assert (venv / "NEMO_RL_VENV_READY").is_file(), venv
print(f"ray_nodes={len(alive_nodes)}")
print(f"ray_gpus={gpu_count}")
print("baked_multinode_grpo_worker_venvs=OK")
ray.shutdown()
PY

      RUN_LOG="$RUN_DIR/run.log"
      mkdir -p "$RUN_DIR/tb" "$HF_HOME" "$HF_DATASETS_CACHE"

      /opt/nemo_rl_venv/bin/python -m examples.run_grpo \
        --config "$RECIPE" \
        grpo.max_num_steps="$MAX_STEPS" \
        grpo.num_prompts_per_step="$NUM_PROMPTS" \
        grpo.num_generations_per_prompt="$NUM_GENERATIONS" \
        grpo.seq_logprob_error_threshold=1.02 \
        policy.train_global_batch_size="$((NUM_PROMPTS * NUM_GENERATIONS))" \
        policy.generation.max_new_tokens="$MAX_NEW_TOKENS" \
        policy.generation.vllm_cfg.gpu_memory_utilization=0.40 \
        policy.generation.vllm_cfg.enforce_eager=false \
        policy.megatron_cfg.tensor_model_parallel_size=2 \
        policy.megatron_cfg.sequence_parallel=false \
        logger.log_dir="$RUN_DIR/tb" \
        logger.wandb_enabled=false \
        checkpointing.enabled=false \
        cluster.num_nodes=2 \
        cluster.gpus_per_node=4 \
        2>&1 | tee "$RUN_LOG"

      grep -Fq "========================= Step $MAX_STEPS/$MAX_STEPS" "$RUN_LOG"
      test "$(grep -Fc "Generation KL Error" "$RUN_LOG")" -ge "$MAX_STEPS"
      grep -Fq "GPU Memory after refit complete" "$RUN_LOG"
      if grep -Fq "Apertus XIELU: using eager fallback" "$RUN_LOG"; then
        echo "Multi-node GRPO workers unexpectedly used the eager xIELU fallback" >&2
        exit 1
      fi
      echo "baked_grpo_multinode_refit=OK"
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
