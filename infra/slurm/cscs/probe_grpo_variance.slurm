#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-probe
#SBATCH --output=logs/grpo_probe_variance_%j.out
#SBATCH --error=logs/grpo_probe_variance_%j.err
#SBATCH --time=01:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=debug
#SBATCH --export=NONE
#SBATCH --no-requeue

# Multi-step GRPO probe: characterize the Generation-KL-Error gauge across
# steps/batches (is 0.79 constant, length-correlated, or content-correlated?).
# Memory fix vs the first probe: vllm gpu_memory_utilization 0.55 -> 0.40.
set -euo pipefail
# Run from the repo root (`cd <repo> && sbatch <this>`) or anywhere with `sbatch --chdir=<repo> <this>`.
# REPO_DIR defaults to the job working dir ($PWD) — also where the relative #SBATCH logs/ land, so they agree.
# (sbatch copies this script to /var/spool/slurmd, so BASH_SOURCE doesn't resolve here; $PWD is the reliable handle.)
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}  # shared CUDA xielu kernel; unset for eager fallback
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/probe-grpo-apertus1p5-8b-1n4g-megatron.yaml}

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}
    cd $REPO_DIR
    # Apertus runtime guard: fail loud if nemo_rl is the stock /opt copy rather than this checkout (online GRPO otherwise silently regresses to gen-KL ~0.79).
    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'
    uv run --locked python examples/run_grpo.py --config $RECIPE \
      grpo.max_num_steps=4 \
      policy.generation.vllm_cfg.gpu_memory_utilization=0.40
  "
echo "END TIME: $(date)"
