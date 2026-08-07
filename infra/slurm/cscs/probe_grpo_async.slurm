#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-probe-async
#SBATCH --output=logs/grpo_probe_async_%j.out
#SBATCH --error=logs/grpo_probe_async_%j.err
#SBATCH --time=01:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=debug
#SBATCH --export=NONE
#SBATCH --no-requeue

# Async-GRPO variant of the fixgate probe: non-colocated 2+2 GPU split,
# AsyncLLM engine, lag-1 trajectories with importance-sampling correction.
# Same gate: Generation KL Error ~0.0003 per step. Thin wrapper so the two
# probes cannot drift: RECIPE is the only delta, sourced into the fixgate
# launcher (its #SBATCH headers are inert comments when sourced).
REPO_DIR=${REPO_DIR:-$PWD}
export RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/probe-grpo-apertus1p5-8b-1n4g-megatron-async.yaml}
source "$REPO_DIR/infra/slurm/cscs/probe_grpo_fixgate.slurm"
