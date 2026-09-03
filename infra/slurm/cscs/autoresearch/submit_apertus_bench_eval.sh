#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Greedy GSM8K-test evaluation of one Apertus checkpoint (start or final) on one
# node, scoring with the same bracket-answer reward as training (outcome only).

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
EXPECTED_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
AP_CKPT=${AP_CKPT:?path to the HF checkpoint to evaluate}
AP_TOKENIZER=${AP_TOKENIZER:-/capstor/store/cscs/swissai/infra01/users/xyixuan/rl-bench/models/ap1p5-70b-sft-262k-2700_corr}
AP_EVAL_DATA=${AP_EVAL_DATA:-/capstor/store/cscs/swissai/infra01/users/xyixuan/rl-bench/data/gsm8k_test.jsonl}
AP_EVAL_CONFIG=${AP_EVAL_CONFIG:-$REPO_DIR/examples/configs/evals/apertus_bench_gsm8k.yaml}
AP_EVAL_TAG=${AP_EVAL_TAG:?short tag for this evaluation, e.g. start or step46}
AP_RUN_ROOT=${AP_RUN_ROOT:-/iopsstor/scratch/cscs/xyixuan/nemo_rl_apertus_bench/eval/$EXPECTED_HEAD}
SBATCH_LOG_ROOT=${SBATCH_LOG_ROOT:-$REPO_DIR/.tmp/slurm-logs/apertus-bench-eval/$EXPECTED_HEAD}
AP_RESERVATION=${AP_RESERVATION-SD-69241-apertus-1-5-0}
AP_TIME=${AP_TIME:-01:00:00}
SBATCH_BIN=${SBATCH_BIN:-sbatch}

[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ -r "$AP_EVAL_CONFIG" ]] || { echo "Missing eval config: $AP_EVAL_CONFIG" >&2; exit 1; }
[[ -r "$AP_CKPT/model.safetensors.index.json" ]] || { echo "Missing checkpoint: $AP_CKPT" >&2; exit 1; }
[[ -r "$AP_TOKENIZER/chat_template.jinja" ]] || { echo "Missing tokenizer: $AP_TOKENIZER" >&2; exit 1; }
[[ -r "$AP_EVAL_DATA" ]] || { echo "Missing eval data: $AP_EVAL_DATA" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }

SBATCH_RESERVATION_ARGS=()
if [[ -n "$AP_RESERVATION" ]]; then
  SBATCH_RESERVATION_ARGS+=(--reservation="$AP_RESERVATION")
fi
mkdir -p "$SBATCH_LOG_ROOT"

unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts

exec "$SBATCH_BIN" \
  --account=infra01 \
  --partition=normal \
  "${SBATCH_RESERVATION_ARGS[@]}" \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --cpus-per-task=64 \
  --mem=400000M \
  --time="$AP_TIME" \
  --job-name="apertus-bench-eval-$AP_EVAL_TAG" \
  --output="$SBATCH_LOG_ROOT/slurm_%j.out" \
  --error="$SBATCH_LOG_ROOT/slurm_%j.err" \
  --export=NONE \
  --wrap="set -euo pipefail
REPO_DIR=$REPO_DIR
srun --cpu-bind=none --environment=$CONTAINER_ENV --ntasks=1 --gpus-per-task=4 --cpus-per-task=64 /bin/bash -lc '
  set -euo pipefail
  cd $REPO_DIR
  test \"\$(git rev-parse HEAD)\" = \"$EXPECTED_HEAD\"
  export PYTHONPATH=$REPO_DIR PYTHONUNBUFFERED=1
  export HF_HOME=/iopsstor/scratch/cscs/xyixuan/.cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
  export AP_CKPT=$AP_CKPT AP_TOKENIZER=$AP_TOKENIZER AP_EVAL_DATA=$AP_EVAL_DATA
  export AP_RUN_DIR=$AP_RUN_ROOT/$AP_EVAL_TAG-\$SLURM_JOB_ID
  export WANDB_DISABLED=true VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_DISABLE_PYNCCL=1
  mkdir -p \"\$AP_RUN_DIR\"
  /opt/nemo_rl_venv/bin/python -m examples.run_eval --config $AP_EVAL_CONFIG 2>&1 | tee \"\$AP_RUN_DIR/eval.log\"
  echo apertus_bench_eval=DONE tag=$AP_EVAL_TAG run_dir=\$AP_RUN_DIR
'"
