#!/bin/bash
# Online DPO two-job orchestrator (run from a login node).
#
# Brings up the judge server, discovers its OpenAI-compatible URL, health-checks
# it, then submits the training job with JUDGE_BASE_URL/JUDGE_MODEL/JUDGE_API_KEY
# injected. Server and training then run independently (mirrors the SwissAI SPIN
# orchestrator). Everything is configurable via env vars — no models/paths baked in.
#
# Two serving backends:
#   * default (self-contained): sbatch serve_judge.slurm (single-node vLLM server)
#   * MODEL_LAUNCH_DIR set: shell out to the SwissAI model-launch CLI for
#     router-balanced replicas (its submit_job.py submits its own job).
#
# Usage:
#   JUDGE_SERVE_MODEL=/path/to/judge JUDGE_API_KEY=$MYKEY \
#     infra/slurm/cscs/online_dpo_orchestrator.sh
set -euo pipefail

REPO_DIR=${REPO_DIR:-$PWD}
JUDGE_SERVE_MODEL=${JUDGE_SERVE_MODEL:?set JUDGE_SERVE_MODEL to the judge weights (HF path or repo id)}
JUDGE_SERVED_NAME=${JUDGE_SERVED_NAME:-$(basename "$JUDGE_SERVE_MODEL")}
JUDGE_SERVE_PORT=${JUDGE_SERVE_PORT:-8080}
JUDGE_API_KEY=${JUDGE_API_KEY:-EMPTY}
SERVE_LAUNCHER=${SERVE_LAUNCHER:-$REPO_DIR/infra/slurm/cscs/serve_judge.slurm}
TRAIN_LAUNCHER=${TRAIN_LAUNCHER:-$REPO_DIR/infra/slurm/cscs/submit_online_dpo.slurm}
LOG_DIR=${LOG_DIR:-$REPO_DIR/logs}
HEALTH_RETRIES=${HEALTH_RETRIES:-120}     # x 10s = up to 20 min for the server to come up
# SwissAI model-launch CLI dir (router-balanced replicas). Empty -> self-contained path.
MODEL_LAUNCH_DIR=${MODEL_LAUNCH_DIR:-}

mkdir -p "$LOG_DIR"
JUDGE_BASE_URL=""

if [ -n "$MODEL_LAUNCH_DIR" ]; then
  # --- SwissAI model-launch backend (router-balanced replicas) ---
  # Mirrors the SPIN orchestrator (online_PO/orchestrator.sh): submit_job.py submits
  # its own SLURM job that brings up SERVER_WORKERS vLLM/SGLang workers behind an
  # SGLang router. To match the real submit_job.py CLI:
  #   * --slurm-nodes (total nodes) is REQUIRED.
  #   * submit_job.py has NO --served-model-name / --tensor-parallel-size flags; those
  #     are vLLM args and must go inside --framework-args.
  #   * the env that RUNS submit_job.py (MODEL_LAUNCH_RUN_ENV) is separate from the
  #     worker SLURM env (--slurm-environment); the reference runs it under "activeuf".
  #   * submit_job.py writes the server log to <cwd>/logs/<jobid>/log.out, so it is run
  #     from SERVER_LOGS_DIR and we poll there for the "Router URL:" line.
  SERVER_FRAMEWORK=${SERVER_FRAMEWORK:-vllm}
  SERVER_WORKERS=${SERVER_WORKERS:-2}                # router-balanced replicas (the reference used 8)
  # The model-launch backend serves router-balanced replicas; --use-router needs workers>1
  # (single worker emits no "Router URL:" line -> URL discovery would hang). Fail fast and
  # point single-replica users at the self-contained serve_judge.slurm instead.
  if [ "$SERVER_WORKERS" -lt 2 ]; then
    echo "ERROR: the model-launch backend needs SERVER_WORKERS>=2 (got $SERVER_WORKERS). For a single judge replica, unset MODEL_LAUNCH_DIR to use serve_judge.slurm." >&2
    exit 1
  fi
  SERVER_NODES_PER_WORKER=${SERVER_NODES_PER_WORKER:-1}
  SERVER_NODES=${SERVER_NODES:-$((SERVER_WORKERS * SERVER_NODES_PER_WORKER))}  # total nodes (submit_job.py --slurm-nodes, required)
  SERVER_TP_SIZE=${SERVER_TP_SIZE:-4}                # tensor-parallel size per replica (a vLLM arg)
  SERVER_ACCOUNT=${SERVER_ACCOUNT:-infra01}
  JOB_TIME=${JOB_TIME:-12:00:00}
  RESERVATION=${RESERVATION:-}
  MODEL_LAUNCH_ENV=${MODEL_LAUNCH_ENV:-$MODEL_LAUNCH_DIR/serving/envs/${SERVER_FRAMEWORK}.toml}  # worker SLURM env
  ROUTER_ENV=${ROUTER_ENV:-$MODEL_LAUNCH_DIR/serving/envs/sglang.toml}                           # router SLURM env
  MODEL_LAUNCH_RUN_ENV=${MODEL_LAUNCH_RUN_ENV:-$MODEL_LAUNCH_ENV}  # env that runs submit_job.py (set to e.g. activeuf if the worker env lacks its deps)
  SERVER_LOGS_DIR=${SERVER_LOGS_DIR:-$LOG_DIR}
  mkdir -p "$SERVER_LOGS_DIR"
  # served-model-name + tensor-parallel-size are vLLM args -> inside --framework-args.
  framework_args="--model $JUDGE_SERVE_MODEL --served-model-name $JUDGE_SERVED_NAME --tensor-parallel-size $SERVER_TP_SIZE --host 0.0.0.0 --port $JUDGE_SERVE_PORT"
  echo "▶ Launching judge via model-launch ($MODEL_LAUNCH_DIR): ${SERVER_WORKERS} workers x ${SERVER_NODES_PER_WORKER} node(s), TP=${SERVER_TP_SIZE} (${SERVER_NODES} nodes total)..."
  submit_out=$(srun --overlap --nodes=1 --ntasks=1 \
    --environment="$MODEL_LAUNCH_RUN_ENV" --container-writable --container-workdir="$SERVER_LOGS_DIR" \
    bash -c "cd '$SERVER_LOGS_DIR' && ${RESERVATION:+export SBATCH_RESERVATION='$RESERVATION' && }python -u '$MODEL_LAUNCH_DIR/serving/submit_job.py' \
      --slurm-nodes $SERVER_NODES \
      --slurm-account '$SERVER_ACCOUNT' \
      --slurm-time '$JOB_TIME' \
      --workers $SERVER_WORKERS \
      --nodes-per-worker $SERVER_NODES_PER_WORKER \
      --use-router \
      --serving-framework '$SERVER_FRAMEWORK' \
      --worker-port $JUDGE_SERVE_PORT \
      --slurm-environment '$MODEL_LAUNCH_ENV' \
      --router-environment '$ROUTER_ENV' \
      --disable-ocf \
      --framework-args '$framework_args'" 2>&1)
  echo "$submit_out"
  server_jobid=$(echo "$submit_out" | sed -n 's/.*Job submitted successfully with ID: *\([0-9]*\).*/\1/p' | head -1)
  : "${server_jobid:?could not parse server job id from model-launch output}"
  server_log="$SERVER_LOGS_DIR/logs/$server_jobid/log.out"
  echo "▶ Waiting for router URL in $server_log ..."
  for _ in $(seq 1 "$HEALTH_RETRIES"); do
    if [ -f "$server_log" ]; then
      url=$(sed -n 's/.*Router URL: *\(http[^ ]*\).*/\1/p' "$server_log" | head -1 || true)
      [ -n "$url" ] && { JUDGE_BASE_URL="${url%/}/v1"; break; }
    fi
    sleep 10
  done
else
  # --- self-contained backend: sbatch serve_judge.slurm ---
  echo "▶ Launching self-contained judge server (serve_judge.slurm)..."
  server_jobid=$(JUDGE_SERVE_MODEL="$JUDGE_SERVE_MODEL" JUDGE_SERVED_NAME="$JUDGE_SERVED_NAME" \
    JUDGE_SERVE_PORT="$JUDGE_SERVE_PORT" JUDGE_API_KEY="$JUDGE_API_KEY" \
    sbatch --parsable \
      --export=ALL,JUDGE_SERVE_MODEL="$JUDGE_SERVE_MODEL",JUDGE_SERVED_NAME="$JUDGE_SERVED_NAME",JUDGE_SERVE_PORT="$JUDGE_SERVE_PORT",JUDGE_API_KEY="$JUDGE_API_KEY" \
      "$SERVE_LAUNCHER")
  : "${server_jobid:?sbatch did not return a job id}"
  server_log="$LOG_DIR/judge_server_${server_jobid}.out"
  echo "▶ Server job $server_jobid submitted; waiting for its URL in $server_log ..."
  for _ in $(seq 1 "$HEALTH_RETRIES"); do
    if [ -f "$server_log" ]; then
      url=$(sed -n 's/.*JUDGE SERVER URL: *\(http[^ ]*\).*/\1/p' "$server_log" | head -1 || true)
      [ -n "$url" ] && { JUDGE_BASE_URL="$url"; break; }
    fi
    sleep 10
  done
fi

: "${JUDGE_BASE_URL:?timed out waiting for the judge server URL}"
echo "✓ Judge URL: $JUDGE_BASE_URL"

# Health-check until the server answers (vLLM exposes /health at the root).
health_url="${JUDGE_BASE_URL%/v1}/health"
echo "▶ Health-checking $health_url ..."
for _ in $(seq 1 "$HEALTH_RETRIES"); do
  if curl -sf -o /dev/null "$health_url"; then
    echo "✓ Judge server healthy"
    break
  fi
  sleep 10
done

echo "▶ Submitting training job ($TRAIN_LAUNCHER) ..."
train_jobid=$(sbatch --parsable \
  --export=ALL,JUDGE_BASE_URL="$JUDGE_BASE_URL",JUDGE_MODEL="$JUDGE_SERVED_NAME",JUDGE_API_KEY="$JUDGE_API_KEY" \
  "$TRAIN_LAUNCHER")
echo "✓ Training job submitted: $train_jobid (judge server job: $server_jobid)"
echo "  Server and training now run independently. Cancel both with: scancel $server_jobid $train_jobid"
