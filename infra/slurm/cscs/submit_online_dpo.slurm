#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-online-dpo
#SBATCH --output=logs/online_dpo_%j.out
#SBATCH --error=logs/online_dpo_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --export=NONE
#SBATCH --no-requeue

###############################################################################
# Apertus 1.5 8B ONLINE DPO — GRPO rollouts + HTTP judge ranking + DPO loss.
# Consumer side of the two-job architecture: this trains and calls an external
# OpenAI-compatible judge endpoint. Bring the judge up first (serve_judge.slurm
# or your own server) and pass its coordinates in:
#   JUDGE_BASE_URL  e.g. http://<server-host>:8080/v1
#   JUDGE_MODEL     the served-model-name (MUST equal the server's)
#   JUDGE_API_KEY   bearer token (or a placeholder if the server is open)
# Easiest: let infra/slurm/cscs/online_dpo_orchestrator.sh launch the server,
# discover the URL, and sbatch this with those injected.
#
# Manual use:
#   sbatch --export=ALL,JUDGE_BASE_URL=...,JUDGE_MODEL=...,JUDGE_API_KEY=... \
#          infra/slurm/cscs/submit_online_dpo.slurm
###############################################################################
set -euo pipefail
echo "START TIME: $(date)"

# Run from the repo root (`cd <repo> && sbatch <this>`) or `sbatch --chdir=<repo> <this>`;
# REPO_DIR defaults to $PWD, where the relative #SBATCH logs/ also land (run `mkdir -p logs` first).
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.1-cp313}  # shared CUDA xielu kernel; unset for eager fallback
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/probe-online-dpo-apertus1p5-8b-1n4g-megatron.yaml}
# Opt-in worker-venv rebuild — pass on EVERY stock-image run: the rebuild lands in /opt/ray_venvs (the container's ephemeral overlay) and does NOT persist, and the checkout's uv.lock diverges from the baked venvs (else the vLLM worker's numpy fails to load libscipy_openblas). Default off; `sbatch --export=ALL,NRL_FORCE_REBUILD_VENVS=true ...`. See infra/slurm/cscs/README.md (incl. NEMO_RL_VENV_DIR for real reuse).
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false}
# W&B: WANDB_API_KEY via env only (never on a command line; carried into the container by
# SLURM_EXPORT_ENV=ALL below). Recipe's logger.wandb_enabled decides usage; WANDB_DISABLED=true forces off.
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_DISABLED=${WANDB_DISABLED:-}
WANDB_OVERRIDE=""
[ "${WANDB_DISABLED:-}" = true ] && WANDB_OVERRIDE="logger.wandb_enabled=false"

# Judge endpoint (consumed by env.online_dpo_judge via the JUDGE_* env vars).
JUDGE_BASE_URL=${JUDGE_BASE_URL:-}
JUDGE_MODEL=${JUDGE_MODEL:-}
JUDGE_API_KEY=${JUDGE_API_KEY:-}
if [ -z "$JUDGE_BASE_URL" ] || [ -z "$JUDGE_MODEL" ]; then
  echo "WARN: JUDGE_BASE_URL / JUDGE_MODEL not set — the judge env will fail at first score." >&2
  echo "      Pass them via --export, or use online_dpo_orchestrator.sh." >&2
fi
# Force srun to propagate the full job env (incl. JUDGE_*) into the step/container,
# since #SBATCH --export=NONE would otherwise strip it (see submit_nemo_rl_dpo_multinode.slurm).
export SLURM_EXPORT_ENV=ALL

# Fail fast if submodules are not initialized — the driver imports megatron.core and the refit uses the bridge.
for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized. Run: git submodule update --init --recursive" >&2
    exit 1
  fi
done

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}
    export NRL_FORCE_REBUILD_VENVS=$NRL_FORCE_REBUILD_VENVS
    # Judge endpoint: base_url/model interpolated (non-secret); api key read from the
    # inherited container env so it never lands on the command line / in logs.
    export JUDGE_BASE_URL='$JUDGE_BASE_URL'
    export JUDGE_MODEL='$JUDGE_MODEL'
    export JUDGE_API_KEY=\${JUDGE_API_KEY:-}
    # W&B key/flag from the inherited container env (never on the command line).
    export WANDB_API_KEY=\${WANDB_API_KEY:-}
    export WANDB_DISABLED=\${WANDB_DISABLED:-}
    cd $REPO_DIR
    # Apertus runtime guard: fail loud if nemo_rl is the stock /opt copy rather than this checkout.
    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'
    uv run --locked --extra mcore python examples/run_online_dpo_apertus.py --config $RECIPE $WANDB_OVERRIDE
  "
echo "END TIME: $(date)"
