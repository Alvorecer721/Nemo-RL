#!/bin/bash
# Committed Apertus-1.5 70B TP8/PP2 topology experiment for CSCS GH200.
# Usage: sbatch launch_tp8pp2.slurm <expected-experiment-commit>
#SBATCH --account=infra01
#SBATCH --job-name=nrl-ap70b-tp8pp2
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl/v0.7.0/.tmp/apertus70b-tp8pp2/reports/auto_research/20260822-apertus70b-topology/tp8pp2/slurm_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl/v0.7.0/.tmp/apertus70b-tp8pp2/reports/auto_research/20260822-apertus70b-topology/tp8pp2/slurm_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE
#SBATCH --no-requeue

set -euo pipefail

EXPECTED_EXPERIMENT_COMMIT=${1:?Usage: sbatch launch_tp8pp2.slurm expected-experiment-commit}
EXPERIMENT_ROOT=/capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl/v0.7.0/.tmp/apertus70b-tp8pp2
EXPERIMENT_BRANCH=autoresearch/2026-08-22-gh200-throughput/tp8pp2
EXPERIMENT_DIR=$EXPERIMENT_ROOT/reports/auto_research/20260822-apertus70b-topology/tp8pp2
REPO_DIR=/capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl/v0.7.0/.tmp/nemo-upstream-replay
SOURCE_COMMIT=a2de667546e760134ca50cd27fed5d556208e7be
CONTAINER_ENV=$EXPERIMENT_DIR/exact-e941.toml
RAY_PREFLIGHT=$EXPERIMENT_DIR/check_ray_cluster.py
RUN_SUMMARIZER=$EXPERIMENT_DIR/summarize_run.py
EXPERIMENT_CONFIG=$EXPERIMENT_DIR/tp8pp2.yaml
RUN_ROOT=$EXPERIMENT_DIR/runs
TOPOLOGY=tp8pp2
TP_SIZE=8
PP_SIZE=2
XIELU_SITE=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current
APERTUS_CKPT=/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-70b-sft-262k-3000
APERTUS_TOKENIZER=$APERTUS_CKPT
CONVERSION_CACHE=/iopsstor/scratch/cscs/xyixuan/.cache/huggingface/nemo_rl_apertus70b_perf_a2de
MAX_STEPS=10
TRAIN_GBS=32
SEQ_LENGTH=2048
GPUS_PER_NODE=4
NUM_NODES=$SLURM_JOB_NUM_NODES
EXPECTED_UNITS=$((NUM_NODES * GPUS_PER_NODE))
RUN_DIR=$RUN_ROOT/${TOPOLOGY}_${SLURM_JOB_ID}

actual_experiment_commit=$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)
actual_experiment_branch=$(git -C "$EXPERIMENT_ROOT" branch --show-current)
[[ "$actual_experiment_commit" == "$EXPECTED_EXPERIMENT_COMMIT" ]] || {
  echo "Experiment HEAD is $actual_experiment_commit, expected $EXPECTED_EXPERIMENT_COMMIT" >&2
  exit 1
}
[[ "$actual_experiment_branch" == "$EXPERIMENT_BRANCH" ]] || {
  echo "Experiment branch is $actual_experiment_branch, expected $EXPERIMENT_BRANCH" >&2
  exit 1
}
git -C "$EXPERIMENT_ROOT" diff --quiet || {
  echo "Experiment worktree has tracked unstaged changes" >&2
  exit 1
}
git -C "$EXPERIMENT_ROOT" diff --cached --quiet || {
  echo "Experiment worktree has tracked staged changes" >&2
  exit 1
}
[[ "$NUM_NODES" -eq 4 ]] || {
  echo "This comparison requires exactly four nodes, got $NUM_NODES" >&2
  exit 1
}
[[ "$EXPECTED_UNITS" -eq $((TP_SIZE * PP_SIZE)) ]] || {
  echo "TP*PP must equal the 16-GPU world size" >&2
  exit 1
}
[[ -r "$CONTAINER_ENV" ]] || {
  echo "Container EDF not found at $CONTAINER_ENV" >&2
  exit 1
}
[[ -r "$RAY_PREFLIGHT" ]] || {
  echo "Ray preflight script not found at $RAY_PREFLIGHT" >&2
  exit 1
}
[[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "NeMo-RL source is not at $SOURCE_COMMIT" >&2
  exit 1
}
[[ -r "$RUN_SUMMARIZER" ]] || {
  echo "Run summarizer not found at $RUN_SUMMARIZER" >&2
  exit 1
}
[[ -r "$EXPERIMENT_CONFIG" ]] || {
  echo "Experiment config not found at $EXPERIMENT_CONFIG" >&2
  exit 1
}
[[ -f "$XIELU_SITE/xielu/__init__.py" ]] || {
  echo "CUDA xIELU site not found at $XIELU_SITE" >&2
  exit 1
}
[[ -r "$APERTUS_CKPT/model.safetensors.index.json" ]] || {
  echo "Apertus 70B checkpoint is incomplete at $APERTUS_CKPT" >&2
  exit 1
}
mkdir -p "$RUN_DIR" "$CONVERSION_CACHE"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$head_node" | awk '{print $1; exit}')
if [[ -z "$HEAD_IP" ]]; then
  HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
fi
[[ -n "$HEAD_IP" ]] || {
  echo "Could not resolve Ray head node $head_node" >&2
  exit 1
}

{
  echo "job_id=$SLURM_JOB_ID"
  echo "topology=$TOPOLOGY"
  echo "tp=$TP_SIZE"
  echo "pp=$PP_SIZE"
  echo "dp=1"
  echo "cp=1"
  echo "nodes=$NUM_NODES"
  echo "gpus_per_node=$GPUS_PER_NODE"
  echo "head=$head_node"
  echo "head_ip=$HEAD_IP"
  echo "container_env=$CONTAINER_ENV"
  echo "image=/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-e9416845542a-6c7d469c3e2a.sqsh"
  echo "experiment_branch=$EXPERIMENT_BRANCH"
  echo "experiment_commit=$actual_experiment_commit"
  echo "source_root=$REPO_DIR"
  echo "source_commit=$SOURCE_COMMIT"
  echo "checkpoint=$APERTUS_CKPT"
  echo "recipe=$EXPERIMENT_CONFIG"
  echo "max_steps=$MAX_STEPS"
  echo "preference_pairs_per_step=$TRAIN_GBS"
  echo "effective_sequences_per_step=$((TRAIN_GBS * 2))"
  echo "max_sequence_length=$SEQ_LENGTH"
  echo "precision=bfloat16"
  echo "activation_checkpointing=true"
  echo "sequence_parallel=true"
  echo "conversion_cache=$CONVERSION_CACHE"
  echo "submitted_at=$(date --iso-8601=seconds)"
} | tee "$RUN_DIR/manifest.txt"

export APERTUS_CKPT APERTUS_TOKENIZER CONVERSION_CACHE EXPECTED_UNITS GPUS_PER_NODE HEAD_IP
export EXPERIMENT_CONFIG MAX_STEPS NUM_NODES PP_SIZE REPO_DIR RUN_DIR RUN_SUMMARIZER SEQ_LENGTH TOPOLOGY TP_SIZE TRAIN_GBS XIELU_SITE
export CHECK_MEGATRON_VENV=1 EXPECTED_NODES="$NUM_NODES" EXPECTED_PROJECT_ROOT="$REPO_DIR"
export RAY_PREFLIGHT SOURCE_COMMIT
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
    export PYTHONPATH="$REPO_DIR:$XIELU_SITE"
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
    export NRL_MEGATRON_CHECKPOINT_DIR="$CONVERSION_CACHE"
    export UV=/root/.local/bin/uv
    export UV_CACHE_DIR=/capstor/store/cscs/swissai/infra01/MLLM/uv-cache
    export UV_NO_PROGRESS=1
    export NEMO_RL_VENV_DIR="$REPO_DIR/.tmp/apertus70b_perf/worker_venvs_a2de"
    export WANDB_DISABLED=true
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}
    export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-$RUN_DIR/nccl.%h.%p.log}
    export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-180}
    RAY_PORT=1200
    RAY_RES="{\"worker_units\": ${GPUS_PER_NODE}, \"slurm_managed_ray_cluster\": 1}"
    RAY_PORT_ARGS="--min-worker-port=2000 --max-worker-port=2999"

    gpu_host=$(hostname)
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
      --format=csv,noheader,nounits -l 1 > "$RUN_DIR/gpu.${gpu_host}.csv" 2>&1 &
    gpu_monitor_pid=$!

    cleanup() {
      local rc=$?
      trap - EXIT
      kill "$gpu_monitor_pid" >/dev/null 2>&1 || true
      wait "$gpu_monitor_pid" >/dev/null 2>&1 || true
      if [[ "$rc" -ne 0 && -d /tmp/ray/session_latest/logs ]]; then
        local failure_dir="$RUN_DIR/ray-failure-${SLURM_PROCID}"
        mkdir -p "$failure_dir"
        cp -a /tmp/ray/session_latest/logs/. "$failure_dir/" || true
      fi
      ray stop --force >/dev/null 2>&1 || true
      exit "$rc"
    }
    trap cleanup EXIT

    if [[ "$SLURM_PROCID" -eq 0 ]]; then
      ray start --head --port="$RAY_PORT" --node-ip-address="$HEAD_IP" \
        --num-cpus=128 --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
        --ray-client-server-port=1201 --dashboard-port=8265 \
        --node-manager-port=1302 --object-manager-port=1304 \
        --runtime-env-agent-port=1306 --dashboard-agent-grpc-port=1308 \
        --dashboard-agent-listen-port=1312 --metrics-export-port=1310 \
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

      cd "$REPO_DIR"
      /opt/nemo_rl_venv/bin/python "$RAY_PREFLIGHT"

      LOG_DIR="$RUN_DIR/tb"
      RUN_LOG="$RUN_DIR/run.log"
      JSON_METRICS="$RUN_DIR/metrics.json"
      mkdir -p "$LOG_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

      run_cmd=(/opt/nemo_rl_venv/bin/python -m examples.run_dpo \
        --config "$EXPERIMENT_CONFIG" \
        logger.log_dir="$LOG_DIR")
      printf "%q " "${run_cmd[@]}" > "$RUN_DIR/command.txt"
      printf "\n" >> "$RUN_DIR/command.txt"
      "${run_cmd[@]}" 2>&1 | tee "$RUN_LOG"

      /opt/nemo_rl_venv/bin/python /opt/nemo-rl/tests/json_dump_tb_logs.py \
        "$LOG_DIR" --output_path "$JSON_METRICS"
      /opt/nemo_rl_venv/bin/python /opt/nemo-rl/tests/check_metrics.py \
        "$JSON_METRICS" "data[\"train/loss\"][\"$MAX_STEPS\"] < 100"

      /opt/nemo_rl_venv/bin/python "$RUN_SUMMARIZER" \
        --run-log "$RUN_LOG" \
        --metrics "$JSON_METRICS" \
        --output "$RUN_DIR/summary.json" \
        --job-id "$SLURM_JOB_ID" \
        --topology "$TOPOLOGY" \
        --tp "$TP_SIZE" \
        --pp "$PP_SIZE" \
        --world-size "$EXPECTED_UNITS" \
        --steps "$MAX_STEPS" \
        --preference-pairs "$TRAIN_GBS" \
        --sequence-length "$SEQ_LENGTH"

      echo "baked_apertus70b_${TOPOLOGY}_training=OK"
    else
      until (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do
        sleep 3
      done
      exec 3>&- || true
      ray start --address="${HEAD_IP}:${RAY_PORT}" \
        --num-cpus=128 --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
        --node-manager-port=1301 --object-manager-port=1303 \
        --runtime-env-agent-port=1305 --dashboard-agent-grpc-port=1307 \
        --dashboard-agent-listen-port=1311 --metrics-export-port=1309 \
        $RAY_PORT_ARGS --disable-usage-stats
      while (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do
        exec 3>&- || true
        sleep 10
      done
      ray stop --force || true
    fi
  '
