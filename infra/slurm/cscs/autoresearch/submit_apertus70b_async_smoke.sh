#!/bin/bash

set -euo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
RAY_SUB=${RAY_SUB:-$EXPERIMENT_DIR/ray.sub}
IMAGE=${IMAGE:-/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-e9416845542a-6c7d469c3e2a.sqsh}
APERTUS70B_CKPT=${APERTUS70B_CKPT:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-70b-sft-262k-3000}
APERTUS70B_TOKENIZER=${APERTUS70B_TOKENIZER:-$APERTUS70B_CKPT}
APERTUS70B_DAPO_ARROW=${APERTUS70B_DAPO_ARROW:-/iopsstor/scratch/cscs/xyixuan/.cache/huggingface/datasets/BytedTsinghua-SIA___dapo-math-17k/default/0.0.0/65877096c24ffa7abc4e4fa5edb95cf3413a5674/cache-ab5da60e235f6dad.arrow}
APERTUS70B_RUN_ROOT=${APERTUS70B_RUN_ROOT:-$EXPERIMENT_DIR/logs/apertus70b_async_smoke}
APERTUS70B_RUN_ID=${APERTUS70B_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
APERTUS70B_VENV_DIR=${APERTUS70B_VENV_DIR:-/opt/ray_venvs/apertus70b_async_e9416845542a}
APERTUS70B_MEGATRON_CACHE=${APERTUS70B_MEGATRON_CACHE:-/iopsstor/scratch/cscs/xyixuan/.cache/huggingface/nemo_rl_apertus70b_perf_a2de}
APERTUS70B_RECIPE=$EXPERIMENT_DIR/examples/configs/recipes/llm/autoresearch/grpo-apertus1p5-70b-5n4g-megatron-async-vllm-tp4-smoke.yaml
APERTUS70B_PREFLIGHT_ONLY=${APERTUS70B_PREFLIGHT_ONLY:-0}
NODES=5
TIME_LIMIT=03:00:00
if [[ "$APERTUS70B_PREFLIGHT_ONLY" == "1" ]]; then
  NODES=1
  TIME_LIMIT=00:30:00
fi

[[ -r "$IMAGE" ]] || { echo "Missing image: $IMAGE" >&2; exit 1; }
[[ -r "$APERTUS70B_CKPT/model.safetensors.index.json" ]] || { echo "Missing checkpoint: $APERTUS70B_CKPT" >&2; exit 1; }
[[ -r "$APERTUS70B_DAPO_ARROW" ]] || { echo "Missing dataset: $APERTUS70B_DAPO_ARROW" >&2; exit 1; }
[[ -r "$APERTUS70B_RECIPE" ]] || { echo "Missing recipe: $APERTUS70B_RECIPE" >&2; exit 1; }
[[ -r "$RAY_SUB" ]] || { echo "Missing ray.sub: $RAY_SUB" >&2; exit 1; }
mkdir -p "$APERTUS70B_RUN_ROOT"
APERTUS70B_EXPECTED_SOURCE_HEAD=$(git -C "$EXPERIMENT_DIR" rev-parse HEAD)
SOURCE_STATUS=$(git -C "$EXPERIMENT_DIR" status --porcelain --untracked-files=no --ignore-submodules=none)
[[ -z "$SOURCE_STATUS" ]] || { echo "Source tree is not clean: $SOURCE_STATUS" >&2; exit 1; }

export APERTUS70B_CKPT APERTUS70B_DAPO_ARROW APERTUS70B_EXPECTED_SOURCE_HEAD
export APERTUS70B_EXPERIMENT_DIR=$EXPERIMENT_DIR APERTUS70B_IMAGE=$IMAGE
export APERTUS70B_MEGATRON_CACHE APERTUS70B_PREFLIGHT_ONLY APERTUS70B_RECIPE
export APERTUS70B_RUN_ID APERTUS70B_RUN_ROOT APERTUS70B_TOKENIZER APERTUS70B_VENV_DIR
export COMMAND=infra/slurm/cscs/autoresearch/run_apertus70b_async_smoke.sh
export CONTAINER=$IMAGE
export GPUS_PER_NODE=4
export MOUNTS=/capstor,/iopsstor,/users

unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts

cd "$EXPERIMENT_DIR"
exec sbatch \
  --account=infra01 \
  --partition=normal \
  --reservation=SD-69241-apertus-1-5-0 \
  --nodes="$NODES" \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --time="$TIME_LIMIT" \
  --job-name=ap70b-async-e2e \
  --output="$APERTUS70B_RUN_ROOT/slurm_%j.out" \
  --error="$APERTUS70B_RUN_ROOT/slurm_%j.err" \
  --export=ALL \
  "$RAY_SUB"
