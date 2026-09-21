#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=prepare-gym-math-40
#SBATCH --output=logs/prepare_gym_math_40_%j.out
#SBATCH --error=logs/prepare_gym_math_40_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=preemptable
#SBATCH --export=NONE
#SBATCH --no-requeue

set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm026_ncclext.toml}
export MATH_WITH_JUDGE_DATA_DIR=${MATH_WITH_JUDGE_DATA_DIR:-$REPO_DIR/outputs/nemo_gym/math_with_judge/${SLURM_JOB_ID}}
export MATH_WITH_JUDGE_DATASET_REVISION=${MATH_WITH_JUDGE_DATASET_REVISION:-b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29}
export MATH_WITH_JUDGE_PREPARATION_SCRIPT=$REPO_DIR/examples/nemo_gym/apertus/prepare_deepscaler_math_with_judge.py
export RECIPE=$REPO_DIR/examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-probe-gym-math_with_judge-remote-40step.yaml
export MATH_WITH_JUDGE_LAUNCHER=$REPO_DIR/infra/slurm/cscs/probe_grpo_gym_math_with_judge_remote_40step.slurm
UV=${UV:-/root/.local/bin/uv}

export SLURM_EXPORT_ENV=ALL
srun --export=ALL --mpi=pmix --environment="$CONTAINER_ENV" --chdir="$REPO_DIR" --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    cd $REPO_DIR
    $UV run --locked --extra mcore python \
      examples/nemo_gym/apertus/prepare_deepscaler_math_with_judge.py
  "
