#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-vibe
#SBATCH --output=logs/vibe_apertus_%j.out
#SBATCH --error=logs/vibe_apertus_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=debug
#SBATCH --no-requeue
# Inherit the submitter's env (sbatch default) so ENABLE_THINKING / TEMPERATURE /
# TOP_P / MAX_TOKENS / TENSOR_PARALLEL_SIZE overrides reach this script. The
# GRPO probe launcher uses --export=NONE for hermetic runs; vibe tests are
# knob-driven so we want the overrides through.

# Apertus 1.5 SFT vibe test: vLLM offline + diverse prompts + detectors for
# thinking-when-disabled, doom-loop (5-gram repetition), and Chinese drift.
# Override knobs via env: ENABLE_THINKING=true sets enable_thinking on the chat
# template; TEMPERATURE / TOP_P / MAX_TOKENS / TENSOR_PARALLEL_SIZE tune sampling.
set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}
ENABLE_THINKING=${ENABLE_THINKING:-false}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
MAX_TOKENS=${MAX_TOKENS:-2048}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized. Run: git submodule update --init --recursive" >&2
    exit 1
  fi
done

THINKING_FLAG=""
if [ "$ENABLE_THINKING" = "true" ]; then
  THINKING_FLAG="--enable-thinking"
fi

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}
    cd $REPO_DIR
    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'
    uv run --locked --extra vllm python -m nemo_rl_apertus.vibe_test_offline \
      --temperature $TEMPERATURE \
      --top-p $TOP_P \
      --max-tokens $MAX_TOKENS \
      --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
      --out logs/vibe_apertus_\${SLURM_JOB_ID}.jsonl \
      $THINKING_FLAG
  "
echo "END TIME: $(date)"
