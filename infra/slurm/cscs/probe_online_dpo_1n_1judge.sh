#!/bin/bash
# Smoke probe for Apertus online DPO — a thin preset over online_dpo_launcher.sh.
#
# DEFAULT TOPOLOGY: exactly ONE single-replica judge node (one vLLM via serve_judge.slurm,
# no router) + ONE 4-GPU trainer node (the 1n4g DeepScaler probe recipe) — two independent
# Slurm jobs. This script only picks the smoke RECIPE and hands off (exec) to the general
# launcher, which does the work (sbatch orchestrator → serve judge → discover URL →
# health-check → sbatch trainer).
#
# Provide any judge + key (no model is baked in here); single-node judge unless you set
# MODEL_LAUNCH_DIR (+ SERVER_WORKERS>=2) for a router-balanced fleet:
#   cd <repo> && JUDGE_SERVE_MODEL=... JUDGE_API_KEY=$KEY infra/slurm/cscs/probe_online_dpo_1n_1judge.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export REPO_DIR

# Smoke recipe (1n4g DeepScaler). The general launcher leaves MODEL_LAUNCH_DIR unset, so
# the judge is single-node by default. Override RECIPE / MODEL_LAUNCH_DIR to change either.
export RECIPE="${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/probe-online-dpo-apertus1p5-8b-1n4g-megatron.yaml}"

exec "$REPO_DIR/infra/slurm/cscs/online_dpo_launcher.sh"
