#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-gym-mbpp
#SBATCH --output=logs/grpo_gym_mbpp_%j.out
#SBATCH --error=logs/grpo_gym_mbpp_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=preemptable
#SBATCH --export=NONE
#SBATCH --no-requeue

REPO_DIR=${REPO_DIR:-$PWD}
export RECIPE=${RECIPE:-"$REPO_DIR/examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-gym-workspace-mbpp.yaml"}
export MAX_STEPS=${MAX_STEPS:-40}

source "$REPO_DIR/infra/slurm/cscs/probe_grpo_gym.slurm"
