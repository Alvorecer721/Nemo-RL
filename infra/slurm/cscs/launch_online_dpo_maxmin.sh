#!/bin/bash
# Concrete MaxMin online-DPO launcher — pins the SAME judge model the SPIN reference
# uses (Qwen3.6-27B) and the MaxMin prompt set, then hands off (exec) to the general
# online_dpo_launcher.sh (which sbatches the orchestrator → judge server → training).
# Mirrors the reference online_PO/launch_online_50k.sh, a concrete config wrapper around
# its generic orchestrator.sh; here online_dpo_launcher.sh is that generic engine, shared
# with the smoke preset probe_online_dpo_1n_1judge.sh.
#
# Judge parity with the reference is exact on the methodology side (UltraFeedback,
# aspects=[helpfulness], max_tokens=1, temperature=0, top_logprobs=20, enable_thinking=
# false, reward = mean aspect expected-score) — set by the recipe's env.online_dpo_judge
# + UltraFeedbackJudge defaults. This script adds the serving side: which weights to
# serve and at what scale.
#
# PROBE SCALE by default: a single-node judge (TP=4 serves the 27B on one GH200 node) +
# the 1-node/4-GPU MaxMin recipe. The reference served 8 replicas behind an SGLang
# router and trained at bs=256/R=8/prompt8192 — see "reference scale" below to reproduce.
#
# >>> REPOINT BEFORE RUNNING (defaults are the reference's; override via env) <<<
#   - JUDGE_SERVE_MODEL / JUDGE_SERVED_NAME : the Qwen judge weights + its served name.
#   - JUDGE_API_KEY : taken from your shell env (never commit secrets).
#
# Run from the repo root:  JUDGE_API_KEY=$KEY infra/slurm/cscs/launch_online_dpo_maxmin.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export REPO_DIR

# ── Judge model: same Qwen3.6-27B the reference launch_online_50k.sh serves ──────────
# HF cache snapshot (model_type qwen3_5, 15 shards). The serving container mounts
# /iopsstor, so this absolute path is visible inside it.
export JUDGE_SERVE_MODEL="${JUDGE_SERVE_MODEL:-/iopsstor/scratch/cscs/rkreft/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"
export JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-Qwen/Qwen3.6-27B-judge}"   # trainer's JUDGE_MODEL == this
export JUDGE_SERVE_TP_SIZE="${JUDGE_SERVE_TP_SIZE:-4}"                     # 27B fits on one GH200 node (4 GPUs)
export JUDGE_API_KEY="${JUDGE_API_KEY:?set JUDGE_API_KEY in your shell (do not commit secrets)}"

# ── Reference scale (optional): 8 vLLM replicas x 1 node, TP=4, behind an SGLang router ─
# To reproduce the reference serving throughput, point at the SwissAI model-launch
# checkout and bump the replica topology (single-node serve_judge.slurm is the default):
#   export MODEL_LAUNCH_DIR=/iopsstor/scratch/cscs/$USER/model-launch/legacy
#   export SERVER_WORKERS=8 SERVER_NODES_PER_WORKER=1 SERVER_TP_SIZE=4 SERVER_FRAMEWORK=vllm

# ── MaxMin online-DPO recipe (prompt-only loader → judge → DPO) ──────────────────────
# Training HP here are PROBE scale (num_prompts_per_step=8, R=4, 3 steps). The reference
# ran bs=256, R=8, prompt 8192 / resp 2048, lr 5e-6, beta 0.1 (this recipe already uses
# beta 0.1, length-normalized) — scale the recipe up for a full run.
export RECIPE="${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/online-dpo-apertus1p5-8b-maxmin-megatron.yaml}"

echo "▶ MaxMin online-DPO: judge=$(basename "$JUDGE_SERVE_MODEL") served as '$JUDGE_SERVED_NAME' (TP=$JUDGE_SERVE_TP_SIZE)"
echo "  recipe=$(basename "$RECIPE"); backend=${MODEL_LAUNCH_DIR:+model-launch x${SERVER_WORKERS:-?}}${MODEL_LAUNCH_DIR:-serve_judge.slurm (single node)}"
exec "$REPO_DIR/infra/slurm/cscs/online_dpo_launcher.sh"
