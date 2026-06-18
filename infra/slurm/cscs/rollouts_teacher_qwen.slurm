#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=qwen-teacher
#SBATCH --output=logs/rollouts_teacher_qwen_%j.out
#SBATCH --error=logs/rollouts_teacher_qwen_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --no-requeue
# Teacher rollouts on the Apertus-all-fail subset using Qwen3.5-9B. Each node
# runs one TP=4 vLLM instance and owns 1/4 of the prompts (round-robin shard).
# Default n-per-prompt=2 (Qwen is much stronger so pass@2 ≈ Apertus pass@8 in
# terms of "has a correct sample"). First run downloads the ~19 GB Qwen weights.

set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}

PROMPTS_JSONL=${PROMPTS_JSONL:?need PROMPTS_JSONL (all-fail subset of Apertus pass@8)}
OUT_BASE=${OUT_BASE:-logs/rollouts_teacher_qwen_think_8k}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
N_PER_PROMPT=${N_PER_PROMPT:-2}
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized." >&2
    exit 1
  fi
done

if [ ! -s "$REPO_DIR/$PROMPTS_JSONL" ]; then
  echo "ERROR: prompts JSONL '$PROMPTS_JSONL' missing or empty." >&2
  exit 1
fi

mkdir -p "$REPO_DIR/logs"

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" \
  --ntasks=4 --ntasks-per-node=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}
    cd $REPO_DIR
    uv run --locked --extra vllm python -m nemo_rl_apertus.rollouts_teacher_qwen_offline \
      --model $MODEL \
      --prompts-jsonl $PROMPTS_JSONL \
      --rank \$SLURM_NODEID \
      --world-size \$SLURM_NNODES \
      --n-per-prompt $N_PER_PROMPT \
      --temperature $TEMPERATURE \
      --top-p $TOP_P \
      --max-tokens $MAX_TOKENS \
      --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
      --out $OUT_BASE.rank\$SLURM_NODEID.jsonl
  "

COMBINED="$REPO_DIR/$OUT_BASE.jsonl"
echo "[postproc] concatenating per-rank files → $COMBINED"
cat "$REPO_DIR/$OUT_BASE".rank*.jsonl > "$COMBINED"
echo "[postproc] combined line count: $(wc -l < "$COMBINED")"

echo "END TIME: $(date)"
