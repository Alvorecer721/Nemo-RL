#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=nemo-rl-dpo-2n
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/nemo_rl_dpo_megatron_2n_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/nemo_rl_dpo_megatron_2n_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
REPO_LOG_ROOT=${REPO_LOG_ROOT:-$REPO_DIR/logs}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_RAY_NODE=${CPUS_PER_RAY_NODE:-$SLURM_CPUS_PER_TASK}
DPO_MAX_STEPS=${DPO_MAX_STEPS:-3}
DPO_VAL_BATCHES=${DPO_VAL_BATCHES:-1}
DPO_VAL_PERIOD=${DPO_VAL_PERIOD:-$DPO_MAX_STEPS}
TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE:-8}
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
PYTHON_CMD=${PYTHON_CMD:-uv run}
if [[ -z "${METRIC_CHECK_EXPR:-}" ]]; then
  METRIC_CHECK_EXPR='data["train/num_valid_samples"]["'"$DPO_MAX_STEPS"'"] >= 1'
fi
RUN_NAME=${RUN_NAME:-nemo_rl_dpo_megatron_${SLURM_NNODES}n_${GPUS_PER_NODE}gpn_gbs${TRAIN_GLOBAL_BATCH_SIZE}_${SLURM_JOB_ID}}
RUN_DIR=${RUN_DIR:-$REPO_LOG_ROOT/$RUN_NAME}
RAY_PORT=${RAY_PORT:-9900}
RAY_CLIENT_SERVER_PORT=${RAY_CLIENT_SERVER_PORT:-9901}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
MIN_WORKER_PORT=${MIN_WORKER_PORT:-10002}
MAX_WORKER_PORT=${MAX_WORKER_PORT:-11000}

mkdir -p "$RUN_DIR"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${NODES[0]}
HEAD_IP_FILE="$RUN_DIR/ray_head_ip"
ENDED_FILE="$RUN_DIR/ENDED"

echo "batch_host=$(hostname)"
echo "container_env=$CONTAINER_ENV"
echo "run_dir=$RUN_DIR"
echo "nodes=${NODES[*]}"
echo "head_node=$HEAD_NODE"
echo "gpus_per_node=$GPUS_PER_NODE"
echo "dpo_max_steps=$DPO_MAX_STEPS"
echo "dpo_val_batches=$DPO_VAL_BATCHES"
echo "dpo_val_period=$DPO_VAL_PERIOD"
echo "train_global_batch_size=$TRAIN_GLOBAL_BATCH_SIZE"
echo "train_micro_batch_size=$TRAIN_MICRO_BATCH_SIZE"
echo "metric_check_expr=$METRIC_CHECK_EXPR"
echo "python_cmd=$PYTHON_CMD"

unset LD_PRELOAD
export PYTHONUNBUFFERED=1
export CUDA_CACHE_DISABLE=${CUDA_CACHE_DISABLE:-0}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/cuda/compute-cache}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_DISABLED=true
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
mkdir -p "$CUDA_CACHE_PATH"

cleanup() {
  touch "$ENDED_FILE" || true
}
trap cleanup EXIT

start_ray_head() {
  srun \
    --mpi=pmix \
    --overlap \
    --environment="$CONTAINER_ENV" \
    --container-name=ray-head \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task="$GPUS_PER_NODE" \
    --cpus-per-task="$CPUS_PER_RAY_NODE" \
    -w "$HEAD_NODE" \
    --output="$RUN_DIR/ray-head.log" \
    bash -lc '
      set -euo pipefail
      export PYTHONUNBUFFERED=1
      export CUDA_CACHE_DISABLE='"$CUDA_CACHE_DISABLE"'
      export CUDA_CACHE_PATH='"$CUDA_CACHE_PATH"'
      export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
      export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
      export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
      mkdir -p "$CUDA_CACHE_PATH"

      HEAD_IP=$(hostname -i | awk "{print \$1}")
      echo "$HEAD_IP" > '"$HEAD_IP_FILE"'
      echo "ray_head_host=$(hostname)"
      echo "ray_head_ip=$HEAD_IP"

      while true; do
        if [[ -f '"$ENDED_FILE"' ]]; then
          ray stop --force || true
          exit 0
        fi
        sleep 30
      done &

      ray start --head \
        --disable-usage-stats \
        --node-ip-address="$HEAD_IP" \
        --port='"$RAY_PORT"' \
        --ray-client-server-port='"$RAY_CLIENT_SERVER_PORT"' \
        --dashboard-port='"$RAY_DASHBOARD_PORT"' \
        --dashboard-host="$HEAD_IP" \
        --include-dashboard=True \
        --min-worker-port='"$MIN_WORKER_PORT"' \
        --max-worker-port='"$MAX_WORKER_PORT"' \
        --num-gpus='"$GPUS_PER_NODE"' \
        --num-cpus='"$CPUS_PER_RAY_NODE"' \
        --resources="{\"slurm_managed_ray_cluster\": 1}" \
        --block
    ' &
}

start_ray_worker() {
  local node=$1
  local worker_idx=$2
  local head_ip
  head_ip=$(cat "$HEAD_IP_FILE")

  srun \
    --mpi=pmix \
    --overlap \
    --environment="$CONTAINER_ENV" \
    --container-name="ray-worker-$worker_idx" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task="$GPUS_PER_NODE" \
    --cpus-per-task="$CPUS_PER_RAY_NODE" \
    -w "$node" \
    --output="$RUN_DIR/ray-worker-$worker_idx.log" \
    bash -lc '
      set -euo pipefail
      export PYTHONUNBUFFERED=1
      export CUDA_CACHE_DISABLE='"$CUDA_CACHE_DISABLE"'
      export CUDA_CACHE_PATH='"$CUDA_CACHE_PATH"'
      export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
      export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
      export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
      mkdir -p "$CUDA_CACHE_PATH"

      echo "ray_worker_host=$(hostname)"
      echo "ray_head='"$head_ip:$RAY_PORT"'"

      while true; do
        if [[ -f '"$ENDED_FILE"' ]]; then
          ray stop --force || true
          exit 0
        fi
        sleep 30
      done &

      ray start --address='"$head_ip:$RAY_PORT"' \
        --disable-usage-stats \
        --min-worker-port='"$MIN_WORKER_PORT"' \
        --max-worker-port='"$MAX_WORKER_PORT"' \
        --num-gpus='"$GPUS_PER_NODE"' \
        --num-cpus='"$CPUS_PER_RAY_NODE"' \
        --resources="{\"slurm_managed_ray_cluster\": 1}" \
        --block
    ' &
}

start_ray_head

for _ in $(seq 1 120); do
  if [[ -s "$HEAD_IP_FILE" ]]; then
    break
  fi
  sleep 2
done

if [[ ! -s "$HEAD_IP_FILE" ]]; then
  echo "Ray head did not publish an IP in time"
  exit 1
fi

for idx in "${!NODES[@]}"; do
  if [[ "$idx" -eq 0 ]]; then
    continue
  fi
  start_ray_worker "${NODES[$idx]}" "$idx"
done

RAY_HEAD_IP=$(cat "$HEAD_IP_FILE")
EXPECTED_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))
ONLINE_GPUS=0

for _ in $(seq 1 180); do
  if srun \
    --mpi=pmix \
    --overlap \
    --environment="$CONTAINER_ENV" \
    --container-name=ray-head \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    -w "$HEAD_NODE" \
    ray status --address="$RAY_HEAD_IP:$RAY_PORT" > "$RUN_DIR/ray-status.txt" 2>&1; then
    ONLINE_GPUS=$(awk '$2 == "GPU" {split($1, parts, "/"); print int(parts[2]); exit}' "$RUN_DIR/ray-status.txt")
    if [[ "${ONLINE_GPUS:-0}" -ge "$EXPECTED_GPUS" ]]; then
      break
    fi
  fi
  cat "$RUN_DIR/ray-status.txt" || true
  sleep 5
done

cat "$RUN_DIR/ray-status.txt" || true

if [[ "${ONLINE_GPUS:-0}" -lt "$EXPECTED_GPUS" ]]; then
  echo "Ray cluster did not expose expected GPUs: online=$ONLINE_GPUS expected=$EXPECTED_GPUS"
  exit 1
fi

LOG_DIR="$RUN_DIR/tb"
JSON_METRICS="$RUN_DIR/metrics.json"
RUN_LOG="$RUN_DIR/run.log"
RAY_PAYLOAD="$RUN_DIR/ray_dpo_payload.sh"
mkdir -p "$LOG_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$CUDA_CACHE_PATH"

cat > "$RAY_PAYLOAD" <<EOF
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_CACHE_DISABLE="$CUDA_CACHE_DISABLE"
export CUDA_CACHE_PATH="$CUDA_CACHE_PATH"
export NCCL_DEBUG=\${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=\${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export HF_HOME="$HF_HOME"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export WANDB_DISABLED=true
export RAY_ADDRESS="$RAY_HEAD_IP:$RAY_PORT"
export LOG_DIR="$LOG_DIR"
export JSON_METRICS="$JSON_METRICS"
export RUN_LOG="$RUN_LOG"
export PYTHONPATH="/opt/nemo-rl:\${PYTHONPATH:-}"
mkdir -p "\$CUDA_CACHE_PATH"
PYTHON_CMD="$PYTHON_CMD"
METRIC_CHECK_EXPR='$METRIC_CHECK_EXPR'

cd /opt/nemo-rl
git config --global --add safe.directory /opt/nemo-rl || true

\$PYTHON_CMD examples/run_dpo.py \
  --config examples/configs/recipes/llm/dpo-llama3.1-8b-instruct-4n8g-megatron.v2.yaml \
  policy.model_name=Qwen/Qwen3-0.6B \
  dpo.max_num_steps="$DPO_MAX_STEPS" \
  dpo.val_batches="$DPO_VAL_BATCHES" \
  dpo.val_period="$DPO_VAL_PERIOD" \
  policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE" \
  policy.train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE" \
  policy.megatron_cfg.tensor_model_parallel_size=1 \
  policy.megatron_cfg.sequence_parallel=false \
  logger.tensorboard_enabled=true \
  logger.log_dir="\$LOG_DIR" \
  logger.wandb_enabled=false \
  logger.monitor_gpus=true \
  checkpointing.enabled=false \
  cluster.gpus_per_node="$GPUS_PER_NODE" \
  cluster.num_nodes="$SLURM_NNODES" \
  2>&1 | tee "\$RUN_LOG"

\$PYTHON_CMD tests/json_dump_tb_logs.py "\$LOG_DIR" --output_path "\$JSON_METRICS"
\$PYTHON_CMD tests/check_metrics.py "\$JSON_METRICS" "\$METRIC_CHECK_EXPR"
EOF
chmod +x "$RAY_PAYLOAD"

srun \
  --mpi=pmix \
  --overlap \
  --environment="$CONTAINER_ENV" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  -w "$HEAD_NODE" \
  bash -lc '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export CUDA_CACHE_DISABLE='"$CUDA_CACHE_DISABLE"'
    export CUDA_CACHE_PATH='"$CUDA_CACHE_PATH"'
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
    ray job submit \
      --address='"http://$RAY_HEAD_IP:$RAY_DASHBOARD_PORT"' \
      -- \
      bash '"$RAY_PAYLOAD"'
  '
