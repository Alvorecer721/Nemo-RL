#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-gym-mbpp-no72
#SBATCH --output=logs/grpo_gym_mbpp_no72_%j.out
#SBATCH --error=logs/grpo_gym_mbpp_no72_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=preemptable
#SBATCH --export=NONE
#SBATCH --no-requeue

REPO_DIR=${REPO_DIR:-$PWD}
export OVERLAY_CHECKPOINT="${OVERLAY_CHECKPOINT:-$REPO_DIR/outputs/model_overlays/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200-mbpp-no-eos72}"
export RECIPE=${RECIPE:-"$REPO_DIR/examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-gym-workspace-mbpp-no-eos72.yaml"}

source "$REPO_DIR/infra/slurm/cscs/train_grpo_gym_workspace_mbpp.slurm"
