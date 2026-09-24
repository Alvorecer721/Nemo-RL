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
# the in-process generation path. Same gate: Generation KL Error <0.002 per
# step. Thin wrapper so the probes cannot drift: RECIPE is the only delta,
# sourced into the fixgate launcher (its #SBATCH headers are inert comments
# when sourced).
REPO_DIR=${REPO_DIR:-$PWD}
export CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/infra/slurm/cscs/environments/nemo_rl_vllm029.toml}
export RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-probe-gym.yaml}
export RUN_SCRIPT=${RUN_SCRIPT:-examples/nemo_gym/run_grpo_nemo_gym.py}
export RUN_PREFIX=${RUN_PREFIX:-"/root/.local/bin/uv run --locked --extra mcore --extra nemo_gym python"}
# Source overlays must rebuild their locked dependencies instead of suppressing
# the container fingerprint check. An exact matching image can opt into reuse.
unset NRL_IGNORE_VERSION_MISMATCH
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}
# `#SBATCH --export=NONE` otherwise propagates to the `srun` step.
export SLURM_EXPORT_ENV=ALL
source "$REPO_DIR/infra/slurm/cscs/probe_grpo_fixgate.slurm"
