#!/bin/bash

set -euo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
RAY_SUB=${RAY_SUB:-$EXPERIMENT_DIR/ray.sub}
IMAGE=${IMAGE:-/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-084ade845b84-a14fb058fe83.sqsh}
APERTUS70B_CONTAINER_ENV=${APERTUS70B_CONTAINER_ENV:-$EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/apertus70b_exact_image.toml}
APERTUS70B_CKPT=${APERTUS70B_CKPT:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-70b-sft-262k-3000}
APERTUS70B_TOKENIZER=${APERTUS70B_TOKENIZER:-$APERTUS70B_CKPT}
APERTUS70B_DAPO_ARROW=${APERTUS70B_DAPO_ARROW:-/iopsstor/scratch/cscs/xyixuan/.cache/huggingface/datasets/BytedTsinghua-SIA___dapo-math-17k/default/0.0.0/65877096c24ffa7abc4e4fa5edb95cf3413a5674/cache-ab5da60e235f6dad.arrow}
APERTUS70B_RUN_ROOT=${APERTUS70B_RUN_ROOT:-$EXPERIMENT_DIR/logs/apertus70b_async_smoke}
APERTUS70B_RUN_ID=${APERTUS70B_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
APERTUS70B_VENV_DIR=${APERTUS70B_VENV_DIR:-/opt/ray_venvs}
APERTUS70B_MEGATRON_CACHE=${APERTUS70B_MEGATRON_CACHE:-/iopsstor/scratch/cscs/xyixuan/.cache/huggingface/nemo_rl_apertus70b_perf_a2de}
APERTUS70B_XIELU_SITE=${APERTUS70B_XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current}
APERTUS70B_RECIPE=$EXPERIMENT_DIR/examples/configs/recipes/llm/autoresearch/grpo-apertus1p5-70b-5n4g-megatron-async-vllm-tp4-smoke.yaml
APERTUS70B_RUNTIME_OVERLAY=$EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/runtime_overlay
APERTUS70B_IMAGE_SOURCE_HEAD=${APERTUS70B_IMAGE_SOURCE_HEAD:-084ade845b8421ab82dcda1849d913da517f194e}
APERTUS70B_PREFLIGHT_ONLY=${APERTUS70B_PREFLIGHT_ONLY:-0}
NODES=5
TIME_LIMIT=03:00:00
if [[ "$APERTUS70B_PREFLIGHT_ONLY" == "1" ]]; then
  NODES=1
  TIME_LIMIT=00:30:00
fi

[[ -r "$IMAGE" ]] || { echo "Missing image: $IMAGE" >&2; exit 1; }
[[ -r "$APERTUS70B_CONTAINER_ENV" ]] || { echo "Missing container EDF: $APERTUS70B_CONTAINER_ENV" >&2; exit 1; }
EDF_IMAGE=$(sed -n 's/^image = "\([^"]*\)"$/\1/p' "$APERTUS70B_CONTAINER_ENV")
[[ "$EDF_IMAGE" == "$IMAGE" ]] || {
  echo "Container EDF image mismatch: expected $IMAGE, found ${EDF_IMAGE:-<unset>}" >&2
  exit 1
}
[[ -r "$APERTUS70B_CKPT/model.safetensors.index.json" ]] || { echo "Missing checkpoint: $APERTUS70B_CKPT" >&2; exit 1; }
[[ -r "$APERTUS70B_DAPO_ARROW" ]] || { echo "Missing dataset: $APERTUS70B_DAPO_ARROW" >&2; exit 1; }
[[ -f "$APERTUS70B_XIELU_SITE/xielu/__init__.py" ]] || { echo "Missing CUDA xIELU site: $APERTUS70B_XIELU_SITE" >&2; exit 1; }
[[ -r "$APERTUS70B_RECIPE" ]] || { echo "Missing recipe: $APERTUS70B_RECIPE" >&2; exit 1; }
[[ -r "$APERTUS70B_RUNTIME_OVERLAY/apertus70b_local_dapo.py" ]] || { echo "Missing runtime overlay" >&2; exit 1; }
[[ -r "$RAY_SUB" ]] || { echo "Missing ray.sub: $RAY_SUB" >&2; exit 1; }
mkdir -p "$APERTUS70B_RUN_ROOT"
APERTUS70B_HARNESS_HEAD=$(git -C "$EXPERIMENT_DIR" rev-parse HEAD)
APERTUS70B_EXPECTED_SOURCE_HEAD=$APERTUS70B_IMAGE_SOURCE_HEAD
SOURCE_STATUS=$(git -C "$EXPERIMENT_DIR" status --porcelain --untracked-files=no --ignore-submodules=none)
[[ -z "$SOURCE_STATUS" ]] || { echo "Source tree is not clean: $SOURCE_STATUS" >&2; exit 1; }

export APERTUS70B_CKPT APERTUS70B_DAPO_ARROW APERTUS70B_EXPECTED_SOURCE_HEAD
export APERTUS70B_HARNESS_HEAD APERTUS70B_RUNTIME_OVERLAY
export APERTUS70B_EXPERIMENT_DIR=$EXPERIMENT_DIR APERTUS70B_IMAGE=$IMAGE
export APERTUS70B_MEGATRON_CACHE APERTUS70B_PREFLIGHT_ONLY APERTUS70B_RECIPE
export APERTUS70B_RUN_ID APERTUS70B_RUN_ROOT APERTUS70B_TOKENIZER APERTUS70B_VENV_DIR
export APERTUS70B_XIELU_SITE
export COMMAND=$EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/run_apertus70b_async_smoke.sh
export CONTAINER_ENV=$APERTUS70B_CONTAINER_ENV
export GPUS_PER_NODE=4
export MOUNTS=
export RAY_SINGLE_SRUN=1

cd "$EXPERIMENT_DIR"
SBATCH_BIN=$(command -v sbatch)
SUBMIT_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CSCS_USER=${USER:-$(id -un)}
SCRATCH=/iopsstor/scratch/cscs/$CSCS_USER
SUBMIT_EXPORTS=PATH,HOME,USER,SHELL,LANG,SCRATCH
SUBMIT_EXPORTS+=,APERTUS70B_CKPT,APERTUS70B_DAPO_ARROW,APERTUS70B_EXPECTED_SOURCE_HEAD
SUBMIT_EXPORTS+=,APERTUS70B_EXPERIMENT_DIR,APERTUS70B_HARNESS_HEAD,APERTUS70B_IMAGE
SUBMIT_EXPORTS+=,APERTUS70B_MEGATRON_CACHE,APERTUS70B_PREFLIGHT_ONLY,APERTUS70B_RECIPE
SUBMIT_EXPORTS+=,APERTUS70B_RUN_ID,APERTUS70B_RUN_ROOT,APERTUS70B_RUNTIME_OVERLAY
SUBMIT_EXPORTS+=,APERTUS70B_TOKENIZER,APERTUS70B_VENV_DIR,APERTUS70B_XIELU_SITE
SUBMIT_EXPORTS+=,BASE_LOG_DIR,COMMAND,CONTAINER_ENV,GPUS_PER_NODE,MOUNTS,RAY_SINGLE_SRUN

# Start the allocation from a clean ingress environment. The EDF and the one
# shared Slurm step are the only authorities for Slingshot credentials.
exec env -i \
  PATH="$SUBMIT_PATH" \
  HOME="${HOME:-/tmp}" \
  USER="$CSCS_USER" \
  SHELL=/bin/bash \
  LANG=C.UTF-8 \
  SCRATCH="$SCRATCH" \
  APERTUS70B_CKPT="$APERTUS70B_CKPT" \
  APERTUS70B_DAPO_ARROW="$APERTUS70B_DAPO_ARROW" \
  APERTUS70B_EXPECTED_SOURCE_HEAD="$APERTUS70B_EXPECTED_SOURCE_HEAD" \
  APERTUS70B_EXPERIMENT_DIR="$APERTUS70B_EXPERIMENT_DIR" \
  APERTUS70B_HARNESS_HEAD="$APERTUS70B_HARNESS_HEAD" \
  APERTUS70B_IMAGE="$APERTUS70B_IMAGE" \
  APERTUS70B_MEGATRON_CACHE="$APERTUS70B_MEGATRON_CACHE" \
  APERTUS70B_PREFLIGHT_ONLY="$APERTUS70B_PREFLIGHT_ONLY" \
  APERTUS70B_RECIPE="$APERTUS70B_RECIPE" \
  APERTUS70B_RUN_ID="$APERTUS70B_RUN_ID" \
  APERTUS70B_RUN_ROOT="$APERTUS70B_RUN_ROOT" \
  APERTUS70B_RUNTIME_OVERLAY="$APERTUS70B_RUNTIME_OVERLAY" \
  APERTUS70B_TOKENIZER="$APERTUS70B_TOKENIZER" \
  APERTUS70B_VENV_DIR="$APERTUS70B_VENV_DIR" \
  APERTUS70B_XIELU_SITE="$APERTUS70B_XIELU_SITE" \
  BASE_LOG_DIR="$EXPERIMENT_DIR" \
  COMMAND="$COMMAND" \
  CONTAINER_ENV="$CONTAINER_ENV" \
  GPUS_PER_NODE="$GPUS_PER_NODE" \
  MOUNTS="$MOUNTS" \
  RAY_SINGLE_SRUN="$RAY_SINGLE_SRUN" \
  "$SBATCH_BIN" \
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
  --export="$SUBMIT_EXPORTS" \
  "$RAY_SUB"
