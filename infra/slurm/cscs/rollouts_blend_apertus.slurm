#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-blend
#SBATCH --output=logs/rollouts_blend_apertus_%j.out
#SBATCH --error=logs/rollouts_blend_apertus_%j.err
#SBATCH --time=00:22:00
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=debug
#SBATCH --no-requeue
# Debug-qos cap: 90 node-minutes per job. 4 nodes × 22 min = 88 node-min, just under.
# First run pays the ~16GB HF download of the 7 subset blobs — subsequent runs cache-hit.

# Apertus 1.5 8B SFT offline rollouts on a 7-subset blend of
# nvidia/Nemotron-RL-Ultra-Training-Blends (ifbench/reasoning/rlhf/swe/mopd/rlvr1/rlvr2).
# 10K total prompts, equal-ish split, shuffled across subsets, then sharded
# across 4 nodes (2,500 prompts each) with TP=4 vLLM per node. Same sampling
# knobs and chat-template setup as the ifbench-only run.
set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}

N_PROMPTS=${N_PROMPTS:-10000}
CONFIGS=${CONFIGS:-ifbench,reasoning,rlhf,mopd,rlvr1,rlvr2}
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
OUT_BASE=${OUT_BASE:-logs/rollouts_blend_apertus_think_8k}

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized. Run: git submodule update --init --recursive" >&2
    exit 1
  fi
done

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
    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'
    uv run --locked --extra vllm python -m nemo_rl_apertus.rollouts_blend_offline \
      --n-prompts $N_PROMPTS \
      --configs $CONFIGS \
      --rank \$SLURM_NODEID \
      --world-size \$SLURM_NNODES \
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
