#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-toolfix
#SBATCH --output=logs/rollouts_ifbench_passk_apertus_%j.out
#SBATCH --error=logs/rollouts_ifbench_passk_apertus_%j.err
#SBATCH --time=00:22:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=debug
#SBATCH --no-requeue
# Re-rollout only the tool-use prompts from the pass@8 sample, this time with
# ``tools=`` threaded through ``apply_chat_template``. ~103 prompts × 8 = ~824
# rollouts; comfortably fits a single debug node × 22 min.

set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}

SOURCE_JSONL=${SOURCE_JSONL:-logs/rollouts_blend_passk_apertus_think_8k.jsonl}
OUT_JSONL=${OUT_JSONL:-logs/rollouts_toolfix_apertus_think_8k.jsonl}
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized." >&2
    exit 1
  fi
done

if [ ! -s "$REPO_DIR/$SOURCE_JSONL" ]; then
  echo "ERROR: source pass@8 JSONL '$SOURCE_JSONL' missing." >&2
  exit 1
fi

mkdir -p "$REPO_DIR/logs"

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}
    cd $REPO_DIR
    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'
    uv run --locked --extra vllm python -m nemo_rl_apertus.rollouts_ifbench_passk_offline \
      --source-jsonl $SOURCE_JSONL \
      --temperature $TEMPERATURE \
      --top-p $TOP_P \
      --max-tokens $MAX_TOKENS \
      --out $OUT_JSONL
  "

echo "END TIME: $(date)"
