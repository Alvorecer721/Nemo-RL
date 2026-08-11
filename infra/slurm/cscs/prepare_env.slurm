#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-prepare-env
#SBATCH --output=logs/prepare_env_%j.out
#SBATCH --error=logs/prepare_env_%j.err
#SBATCH --time=01:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=debug
#SBATCH --export=NONE
#SBATCH --no-requeue

###############################################################################
# One-shot environment provisioning for this checkout: builds the project
# .venv and the worker venvs our recipes use, so training jobs start hot
# instead of paying the first-launch build inside their walltime.
#
# Safe to rerun (uv sync is convergent; ready venvs are skipped via the
# NEMO_RL_VENV_READY marker). Concurrent submissions serialize on an flock —
# two cold-cache builders otherwise deadlock on uv's per-package build locks.
# Optional: training launchers build on demand through the same code path;
# this just moves the wait out of them.
###############################################################################
set -euo pipefail
echo "START TIME: $(date)"

REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
UV=${UV:-/root/.local/bin/uv}
# Actor FQN filters for prefetch_venvs: the Megatron policy worker, both vLLM
# generation workers, the async/sync rollout actors, and NeMo-Gym — the full
# venv surface of the Apertus DPO/GRPO/async/gym recipes.
PREFETCH_FILTERS=${PREFETCH_FILTERS:-"megatron_policy_worker vllm_worker sync_rollout_actor async_utils nemo_gym"}

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src
    # Same env model as the training launchers (see probe_grpo_fixgate.slurm):
    # persistent venvs on capstor, team uv cache, image uv, no progress writer.
    export UV_CACHE_DIR=\${UV_CACHE_DIR:-/capstor/store/cscs/swissai/infra01/MLLM/uv-cache}
    export UV_NO_PROGRESS=1
    export UV_PROJECT_ENVIRONMENT=$REPO_DIR/.venv
    export NEMO_RL_VENV_DIR=$REPO_DIR/venvs
    cd $REPO_DIR
    mkdir -p venvs
    exec 9>venvs/.prepare_env.lock
    if ! flock -w 60 9; then
      echo 'Another prepare_env holds the lock; waiting up to walltime...'
      flock 9
    fi
    echo '=== building project venv (.venv) ==='
    \"$UV\" run --locked --extra mcore python -c 'print(\"project venv ready\")'
    echo '=== prefetching worker venvs ==='
    \"$UV\" run --locked --extra mcore python -m nemo_rl.utils.prefetch_venvs $PREFETCH_FILTERS
  "
echo "END TIME: $(date)"
