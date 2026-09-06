#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
EXPECTED_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm026_ncclext.toml}
AP_VARIANT=${AP_VARIANT:?set AP_VARIANT to 70b-bench or 8b-smoke}
case "$AP_VARIANT" in
  70b-bench)
    RECIPE_NAME=grpo-apertus1p5-70b-16n4g-megatron-tp2pp4-sc-bench.yaml
    AP_CKPT_DEFAULT=/capstor/store/cscs/swissai/infra01/users/xyixuan/rl-bench/models/ap1p5-70b-sft-262k-2700_corr
    AP_EXPECTED_STEPS_DEFAULT=92
    AP_TIME_DEFAULT=04:00:00 ;;
  8b-smoke)
    RECIPE_NAME=grpo-apertus1p5-8b-3n4g-megatron-tp2pp2-sc-bench-smoke.yaml
    AP_CKPT_DEFAULT=/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200
    AP_EXPECTED_STEPS_DEFAULT=2
    AP_TIME_DEFAULT=01:00:00 ;;
  *) echo "Unknown AP_VARIANT: $AP_VARIANT" >&2; exit 1 ;;
esac
AP_RECIPE=${AP_RECIPE:-$REPO_DIR/examples/configs/recipes/llm/$RECIPE_NAME}
AP_CKPT=${AP_CKPT:-$AP_CKPT_DEFAULT}
AP_ANSWER_MARKER=${AP_ANSWER_MARKER:-boxed}
AP_SEED=${AP_SEED:-42}
case "$AP_VARIANT" in 70b-bench*) AP_WANDB_DISABLED=${AP_WANDB_DISABLED:-false} ;; *) AP_WANDB_DISABLED=${AP_WANDB_DISABLED:-true} ;; esac
AP_TOKENIZER=${AP_TOKENIZER:-/capstor/store/cscs/swissai/infra01/users/xyixuan/rl-bench/models/ap1p5-70b-sft-262k-2700_corr}
AP_RUN_ROOT=${AP_RUN_ROOT:-/iopsstor/scratch/cscs/xyixuan/nemo_rl_apertus_bench/$AP_VARIANT/$EXPECTED_HEAD/seed$AP_SEED}
SBATCH_LOG_ROOT=${SBATCH_LOG_ROOT:-$REPO_DIR/.tmp/slurm-logs/apertus-bench-$AP_VARIANT/$EXPECTED_HEAD}
AP_RESERVATION=${AP_RESERVATION-SD-69241-apertus-1-5-0}
AP_EXPECTED_STEPS=${AP_EXPECTED_STEPS:-$AP_EXPECTED_STEPS_DEFAULT}
AP_TIME=${AP_TIME:-$AP_TIME_DEFAULT}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-68719476736}
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-30}
SBATCH_BIN=${SBATCH_BIN:-sbatch}

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ -r "$AP_RECIPE" ]] || { echo "Missing recipe: $AP_RECIPE" >&2; exit 1; }
[[ -r "$AP_CKPT/model.safetensors.index.json" ]] || { echo "Missing checkpoint: $AP_CKPT" >&2; exit 1; }
[[ -r "$AP_TOKENIZER/chat_template.jinja" ]] || { echo "Missing tokenizer: $AP_TOKENIZER" >&2; exit 1; }
AP_TOTAL_NODES=$(python3 - "$AP_RECIPE" <<'PY'
import sys
from pathlib import Path
import yaml
path = Path(sys.argv[1]).resolve()
while path is not None:
    cfg = yaml.safe_load(path.read_text()) or {}
    num_nodes = (cfg.get("cluster") or {}).get("num_nodes")
    if num_nodes is not None:
        print(num_nodes)
        break
    parent = cfg.get("defaults")
    path = (path.parent / parent).resolve() if parent else None
PY
)
[[ -n "$AP_TOTAL_NODES" ]] || { echo "Recipe $AP_RECIPE must set cluster.num_nodes" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }
SUBMODULE_STATUS=$(git -C "$REPO_DIR" -c submodule.recurse=false submodule status)
INVALID_SUBMODULES=$(printf '%s\n' "$SUBMODULE_STATUS" | awk 'substr($0, 1, 1) == "-" || substr($0, 1, 1) == "+"')
[[ -z "$INVALID_SUBMODULES" ]] || { echo "Submodules are uninitialized or do not match gitlinks: $INVALID_SUBMODULES" >&2; exit 1; }

SBATCH_RESERVATION_ARGS=()
if [[ -n "$AP_RESERVATION" ]]; then
  SBATCH_RESERVATION_ARGS+=(--reservation="$AP_RESERVATION")
fi

mkdir -p "$SBATCH_LOG_ROOT"
export COMMAND=infra/slurm/cscs/autoresearch/run_apertus_bench.sh
export CONTAINER_ENV AP_CKPT AP_TOKENIZER AP_RECIPE AP_EXPECTED_STEPS AP_ANSWER_MARKER AP_SEED AP_WANDB_DISABLED
export AP_EXPERIMENT_DIR=$REPO_DIR
export AP_EXPECTED_SOURCE_HEAD=$EXPECTED_HEAD
export AP_RUN_DIR=$AP_RUN_ROOT
export GPUS_PER_NODE=4
export RAY_LOG_SYNC_FREQUENCY RAY_OBJECT_STORE_MEMORY
export RAY_SINGLE_SRUN=1
export BASE_LOG_DIR=$AP_RUN_ROOT/ray

unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts

cd "$REPO_DIR"
exec "$SBATCH_BIN" \
  --account=infra01 \
  --partition=normal \
  "${SBATCH_RESERVATION_ARGS[@]}" \
  --nodes="$AP_TOTAL_NODES" \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --segment=4 \
  --mem=850000M \
  --exclusive \
  --time="$AP_TIME" \
  --job-name="apertus-bench-$AP_VARIANT-seed$AP_SEED" \
  --output="$SBATCH_LOG_ROOT/slurm_%j.out" \
  --error="$SBATCH_LOG_ROOT/slurm_%j.err" \
  --export=ALL \
  "$REPO_DIR/ray.sub"
