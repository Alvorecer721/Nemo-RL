#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-judge
#SBATCH --output=logs/judge_server_%j.out
#SBATCH --error=logs/judge_server_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --export=NONE
#SBATCH --no-requeue

###############################################################################
# Self-contained judge server for online DPO — a single-node vLLM OpenAI server.
# The serving side of the two-job architecture; the training job calls it via
# JUDGE_BASE_URL/JUDGE_MODEL/JUDGE_API_KEY. For multi-replica router-balanced
# serving, use the SwissAI model-launch tool instead (see online_dpo_orchestrator.sh,
# MODEL_LAUNCH_DIR). Everything below is configurable — no models/paths baked in.
#
#   sbatch --export=ALL,JUDGE_SERVE_MODEL=<hf-path>,JUDGE_SERVED_NAME=<name> \
#          infra/slurm/cscs/serve_judge.slurm
#
# This job prints a line "JUDGE SERVER URL: http://<host>:<port>/v1" once up;
# online_dpo_orchestrator.sh greps it to wire the training job.
###############################################################################
set -euo pipefail
echo "START TIME: $(date)"

REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
JUDGE_SERVE_MODEL=${JUDGE_SERVE_MODEL:?set JUDGE_SERVE_MODEL to the judge weights (HF path or repo id)}
JUDGE_SERVED_NAME=${JUDGE_SERVED_NAME:-$(basename "$JUDGE_SERVE_MODEL")}
JUDGE_SERVE_PORT=${JUDGE_SERVE_PORT:-8080}
JUDGE_SERVE_GPUS=${JUDGE_SERVE_GPUS:-${SLURM_GPUS_PER_NODE:-4}}
JUDGE_SERVE_TP_SIZE=${JUDGE_SERVE_TP_SIZE:-$JUDGE_SERVE_GPUS}
export JUDGE_API_KEY=${JUDGE_API_KEY:-EMPTY}  # exported so SLURM_EXPORT_ENV=ALL carries it into the container (read as VLLM_API_KEY there, never on the command line)
# Extra vLLM serve flags (e.g. "--max-model-len 8192 --gpu-memory-utilization 0.9").
JUDGE_SERVE_EXTRA_ARGS=${JUDGE_SERVE_EXTRA_ARGS:-}

SERVER_HOST=$(hostname -f 2>/dev/null || hostname)
echo "JUDGE SERVER URL: http://${SERVER_HOST}:${JUDGE_SERVE_PORT}/v1"
echo "JUDGE SERVED MODEL NAME: ${JUDGE_SERVED_NAME}"
echo "(set JUDGE_MODEL=${JUDGE_SERVED_NAME} on the training job — it must match)"

export SLURM_EXPORT_ENV=ALL
unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task="$JUDGE_SERVE_GPUS" --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    # API key via env (vLLM reads VLLM_API_KEY) so it never lands on the command line / in ps / in logs.
    export VLLM_API_KEY=\${JUDGE_API_KEY:-EMPTY}
    cd $REPO_DIR
    echo '=== starting vLLM OpenAI judge server for $JUDGE_SERVED_NAME (TP=$JUDGE_SERVE_TP_SIZE) ==='
    # vLLM's OpenAI server supports the logprobs/top_logprobs the UltraFeedback judge needs.
    # --extra vllm: vLLM is an optional-dependency extra (pyproject [project.optional-dependencies]),
    # so the bare driver sync omits the 'vllm' console script ('Failed to spawn: vllm').
    # --reinstall-package tokenizers: the fork relocks tokenizers 0.22.2; reconciling it into the
    # image's pre-baked /opt driver venv otherwise fails copying tokenizers.abi3.so ('File exists').
    # UV_LINK_MODE=copy avoids hardlink-into-overlay quirks on the container's writable layer.
    export UV_LINK_MODE=copy
    uv run --locked --extra vllm --reinstall-package tokenizers vllm serve '$JUDGE_SERVE_MODEL' \
      --served-model-name '$JUDGE_SERVED_NAME' \
      --host 0.0.0.0 --port $JUDGE_SERVE_PORT \
      --tensor-parallel-size $JUDGE_SERVE_TP_SIZE \
      $JUDGE_SERVE_EXTRA_ARGS
  "
echo "END TIME: $(date)"
