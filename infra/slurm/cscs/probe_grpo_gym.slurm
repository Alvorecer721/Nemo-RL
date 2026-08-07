#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-probe-gym
#SBATCH --output=logs/grpo_probe_gym_%j.out
#SBATCH --error=logs/grpo_probe_gym_%j.err
#SBATCH --time=01:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=debug
#SBATCH --export=NONE
#SBATCH --no-requeue

# NeMo-Gym variant of the fixgate probe: rollouts go through Gym's HTTP-served
# agent (math_with_judge_simple_agent) against the async vLLM engine instead of
# the in-process generation path. Same gate: Generation KL Error ~0.0003 per
# step. Thin wrapper so the probes cannot drift: RECIPE is the only delta,
# sourced into the fixgate launcher (its #SBATCH headers are inert comments
# when sourced).
REPO_DIR=${REPO_DIR:-$PWD}
export RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/probe-grpo-apertus1p5-8b-1n4g-megatron-gym.yaml}
source "$REPO_DIR/infra/slurm/cscs/probe_grpo_fixgate.slurm"
