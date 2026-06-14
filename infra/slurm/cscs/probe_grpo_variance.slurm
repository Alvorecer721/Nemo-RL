#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-probe
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/grpo_probe_variance_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/grpo_probe_variance_%j.err
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
REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL
RUNTIME_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/nemo-rl-worktrees/v060-mlm-restore
RECIPE=$REPO_DIR/examples/configs/recipes/llm/probe-grpo-apertus1p5-8b-1n4g-megatron.yaml

unset LD_PRELOAD
srun --mpi=pmix --environment=$REPO_DIR/docker/nemo_rl.toml --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=/iopsstor/scratch/cscs/\${USER}/.cache/huggingface
    export PYTHONPATH=$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/iopsstor/scratch/cscs/xyixuan/apertus/xielu-kernel/site-v060
    cd $RUNTIME_DIR
    uv run --locked python examples/run_grpo.py --config $RECIPE \
      grpo.max_num_steps=4 \
      policy.generation.vllm_cfg.gpu_memory_utilization=0.40
  "
echo "END TIME: $(date)"
