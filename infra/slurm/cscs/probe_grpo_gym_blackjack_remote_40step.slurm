#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-gym-blackjack-40
#SBATCH --output=logs/grpo_gym_blackjack_40_%j.out
#SBATCH --error=logs/grpo_gym_blackjack_40_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=preemptable
# The experiment owner explicitly authorized preemptable-partition execution
# without a reservation after the prior site reservation expired.
#SBATCH --export=NONE
#SBATCH --no-requeue

REPO_DIR=${REPO_DIR:-$PWD}
export RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-probe-gym-blackjack-remote-40step.yaml}
export MAX_STEPS=${MAX_STEPS:-40}

source "$REPO_DIR/infra/slurm/cscs/probe_grpo_gym.slurm"
