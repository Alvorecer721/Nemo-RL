#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-fmt
#SBATCH --output=logs/grpo_format_apertus_%j.out
#SBATCH --error=logs/grpo_format_apertus_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --no-requeue

# Apertus 1.5 8B GRPO with the structural format reward (no judge, no verifier).
# Inherits the probe recipe knobs (TP2/PP1, ApertusForCausalLM, sleep_level=2,
# vllm mem 0.40) and trains on a sampled slice of ifbench prompts so the
# unclosed-thinking + repetition penalties get a useful gradient.
set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/grpo-format-apertus1p5-8b-1n4g-megatron.yaml}
DATA_JSONL=${DATA_JSONL:-$REPO_DIR/nemo_rl_apertus/data/format_blend_ifbench.jsonl}
DATA_N=${DATA_N:-1000}
DATA_SEED=${DATA_SEED:-42}

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized. Run: git submodule update --init --recursive" >&2
    exit 1
  fi
done

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}
    cd $REPO_DIR

    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'

    # Preflight: download a sampled slice of ifbench prompts on first run.
    if [ ! -s '$DATA_JSONL' ]; then
      echo '[preflight] downloading ifbench prompts → $DATA_JSONL'
      uv run --locked --extra mcore python -m nemo_rl_apertus.data.download_ifbench_prompts \
        --n $DATA_N --seed $DATA_SEED --out '$DATA_JSONL'
    else
      echo '[preflight] reusing $DATA_JSONL'
    fi

    uv run --locked --extra mcore python examples/run_grpo_apertus_format.py --config $RECIPE \
      policy.generation.vllm_cfg.gpu_memory_utilization=0.40
  "
echo "END TIME: $(date)"
