#!/bin/bash
# General Apertus online-DPO launcher — the shared engine for the two presets:
#   * probe_online_dpo_1n_1judge.sh  (smoke: 1n4g DeepScaler + single-node judge)
#   * launch_online_dpo_maxmin.sh    (reference MaxMin set + Qwen judge)
# Both just pre-set config (RECIPE, the judge) and `exec` this script; it owns all the
# launch logic and is what you'd call directly for any other online-DPO run.
#
# It sets the judge-server + training config from the environment, then submits the
# orchestrator as a 1-node job. The orchestrator brings the judge up, discovers its
# OpenAI-compatible URL, health-checks it, and sbatches the training job
# (submit_online_dpo.slurm) with JUDGE_BASE_URL/JUDGE_MODEL/JUDGE_API_KEY injected.
# Judge server and training then run independently.
#
# DEFAULT judge backend: ONE single-replica judge node (serve_judge.slurm, no router).
# To scale to a router-balanced fleet (as the reference does), set MODEL_LAUNCH_DIR and
# bump SERVER_WORKERS/SERVER_NODES (see "Judge server config").
#
# >>> Required env (no models/paths/secrets baked in) <<<
#   - RECIPE             : online-DPO recipe yaml (the presets set this for you)
#   - JUDGE_SERVE_MODEL  : judge weights (HF path or repo id)
#   - JUDGE_API_KEY      : bearer token, from your shell env
# Optional: JUDGE_SERVED_NAME (== trainer's JUDGE_MODEL; default basename), JUDGE_SERVE_PORT,
#   JUDGE_SERVE_TP_SIZE, MODEL_LAUNCH_DIR (+ SERVER_*) for router-balanced serving.
#
# Run from the repo root so the relative logs/ resolve consistently.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
mkdir -p "$LOG_DIR"

# ── SLURM / account ──────────────────────────────────────────────────────
ACCOUNT="${ACCOUNT:-infra01}"
RESERVATION="${RESERVATION:-}"          # e.g. SD-69241-apertus-1-5-0; empty = open 'normal' queue
PARTITION="${PARTITION:-normal}"
JOB_TIME="${JOB_TIME:-12:00:00}"

# ── Judge server config (this is where you set replicas / nodes / TP) ────
# Default backend: single-node serve_judge.slurm (MODEL_LAUNCH_DIR empty), sized by
# JUDGE_SERVE_TP_SIZE. For router-balanced replicas, set MODEL_LAUNCH_DIR
# (e.g. $SCRATCH/model-launch/legacy) and SERVER_WORKERS>=2 (sized by SERVER_*);
# the reference launch_online_50k.sh used 8 workers x 1 node, TP=4.
JUDGE_SERVE_MODEL="${JUDGE_SERVE_MODEL:?set JUDGE_SERVE_MODEL to the judge weights (HF path or repo id)}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-$(basename "$JUDGE_SERVE_MODEL")}"   # trainer must use JUDGE_MODEL == this
JUDGE_SERVE_PORT="${JUDGE_SERVE_PORT:-8080}"
JUDGE_SERVE_TP_SIZE="${JUDGE_SERVE_TP_SIZE:-4}"    # GPUs for the single-node judge (serve_judge.slurm)
JUDGE_API_KEY="${JUDGE_API_KEY:?set JUDGE_API_KEY in your shell (do not commit secrets)}"

# Router-balanced backend (optional): set MODEL_LAUNCH_DIR to enable; needs SERVER_WORKERS>=2.
MODEL_LAUNCH_DIR="${MODEL_LAUNCH_DIR:-}"            # empty = single-node serve_judge.slurm
MODEL_LAUNCH_RUN_ENV="${MODEL_LAUNCH_RUN_ENV:-}"   # env that runs submit_job.py (reference: activeuf); empty = worker env
SERVER_FRAMEWORK="${SERVER_FRAMEWORK:-vllm}"
SERVER_WORKERS="${SERVER_WORKERS:-2}"              # router-balanced replicas (>=2; reference used 8)
SERVER_NODES_PER_WORKER="${SERVER_NODES_PER_WORKER:-1}"
SERVER_NODES="${SERVER_NODES:-$((SERVER_WORKERS * SERVER_NODES_PER_WORKER))}"   # total judge nodes
SERVER_TP_SIZE="${SERVER_TP_SIZE:-4}"             # tensor-parallel per replica (vLLM arg)

# ── Training recipe (set by the preset launchers; required when called directly) ──
RECIPE="${RECIPE:?set RECIPE to an online-DPO recipe yaml, or call a preset (probe_online_dpo_1n_1judge.sh / launch_online_dpo_maxmin.sh)}"

# W&B: export the key/flag so the orchestrator → trainer inherit them via --export=ALL (no key on a
# command line). The recipe's logger.wandb_enabled decides usage; WANDB_DISABLED=true forces off.
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_DISABLED=${WANDB_DISABLED:-}

# ── Submit the orchestrator (1 node) — it launches the judge, then training ──
if [ -n "$MODEL_LAUNCH_DIR" ]; then
  judge_desc="model-launch: ${SERVER_WORKERS} replicas x ${SERVER_NODES_PER_WORKER} node(s), TP=${SERVER_TP_SIZE}"
else
  judge_desc="serve_judge.slurm (single node)"
fi
echo "Submitting online-DPO orchestrator (judge: ${judge_desc}; recipe: $(basename "$RECIPE"))"

sbatch \
  --job-name="online-dpo-orch" \
  --account="$ACCOUNT" \
  ${RESERVATION:+--reservation="$RESERVATION"} \
  --partition="$PARTITION" \
  --time="$JOB_TIME" \
  --nodes=1 --ntasks-per-node=1 \
  --output="$LOG_DIR/online_dpo_orch_%j.out" \
  --error="$LOG_DIR/online_dpo_orch_%j.err" \
  --export=ALL,REPO_DIR="$REPO_DIR",LOG_DIR="$LOG_DIR",RECIPE="$RECIPE",JUDGE_SERVE_MODEL="$JUDGE_SERVE_MODEL",JUDGE_SERVED_NAME="$JUDGE_SERVED_NAME",JUDGE_SERVE_PORT="$JUDGE_SERVE_PORT",JUDGE_SERVE_TP_SIZE="$JUDGE_SERVE_TP_SIZE",JUDGE_API_KEY="$JUDGE_API_KEY",MODEL_LAUNCH_DIR="$MODEL_LAUNCH_DIR",MODEL_LAUNCH_RUN_ENV="$MODEL_LAUNCH_RUN_ENV",SERVER_FRAMEWORK="$SERVER_FRAMEWORK",SERVER_WORKERS="$SERVER_WORKERS",SERVER_NODES_PER_WORKER="$SERVER_NODES_PER_WORKER",SERVER_NODES="$SERVER_NODES",SERVER_TP_SIZE="$SERVER_TP_SIZE",SERVER_ACCOUNT="$ACCOUNT",JOB_TIME="$JOB_TIME",RESERVATION="$RESERVATION" \
  "$REPO_DIR/infra/slurm/cscs/online_dpo_orchestrator.sh"

echo "Orchestrator submitted. Watch its log in $LOG_DIR/online_dpo_orch_<jobid>.out;"
echo "it will print the judge URL and then sbatch submit_online_dpo.slurm."
