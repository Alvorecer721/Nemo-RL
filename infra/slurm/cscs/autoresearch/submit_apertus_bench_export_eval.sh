#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Export a finished bench run's final Megatron checkpoint to HF (TP4 on one
# node) and score it with the same greedy GSM8K-test eval as the start.

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
EXPECTED_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
AP_BENCH_RUN_DIR=${AP_BENCH_RUN_DIR:?bench run directory holding checkpoints/step_N}
AP_HF_BASE=${AP_HF_BASE:-/capstor/store/cscs/swissai/infra01/users/xyixuan/rl-bench/models/ap1p5-70b-sft-262k-2700_corr}
AP_TOKENIZER=${AP_TOKENIZER:-$AP_HF_BASE}
AP_EVAL_DATA=${AP_EVAL_DATA:-/capstor/store/cscs/swissai/infra01/users/xyixuan/rl-bench/data/gsm8k_test.jsonl}
AP_EVAL_CONFIG=${AP_EVAL_CONFIG:-$REPO_DIR/examples/configs/evals/apertus_bench_gsm8k.yaml}
AP_ANSWER_MARKER=${AP_ANSWER_MARKER:-boxed}
AP_EVAL_TAG=${AP_EVAL_TAG:-after}
AP_RESERVATION=${AP_RESERVATION-SD-69241-apertus-1-5-0}
AP_TIME=${AP_TIME:-02:00:00}
SBATCH_LOG_ROOT=${SBATCH_LOG_ROOT:-$REPO_DIR/.tmp/slurm-logs/apertus-bench-export-eval/$EXPECTED_HEAD}
SBATCH_BIN=${SBATCH_BIN:-sbatch}
MCORE_PYTHON=${MCORE_PYTHON:-/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}

AP_STEP=${AP_STEP:-}
STEP_DIR=$(ls -d "$AP_BENCH_RUN_DIR"/checkpoints/step_${AP_STEP:-*} 2>/dev/null | sort -V | tail -1)
[[ -n "$STEP_DIR" && -d "$STEP_DIR/policy/weights" ]] || { echo "No policy checkpoint under $AP_BENCH_RUN_DIR/checkpoints" >&2; exit 1; }
[[ -r "$CONTAINER_ENV" ]] || { echo "Missing container EDF: $CONTAINER_ENV" >&2; exit 1; }
[[ -r "$AP_HF_BASE/model.safetensors.index.json" ]] || { echo "Missing HF base: $AP_HF_BASE" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }

EXPORT_DIR=$AP_BENCH_RUN_DIR/hf_export${AP_STEP:+_step$AP_STEP}
SBATCH_RESERVATION_ARGS=()
[[ -n "$AP_RESERVATION" ]] && SBATCH_RESERVATION_ARGS+=(--reservation="$AP_RESERVATION")
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
  --mem=850000M \
  --time="$AP_TIME" \
  --job-name="apertus-bench-export-eval-$(basename "$AP_BENCH_RUN_DIR")" \
  --output="$SBATCH_LOG_ROOT/slurm_%j.out" \
  --error="$SBATCH_LOG_ROOT/slurm_%j.err" \
  --export=NONE \
  --wrap="set -euo pipefail
srun --cpu-bind=none --environment=$CONTAINER_ENV --ntasks=1 --gpus-per-task=4 --cpus-per-task=64 /bin/bash -lc '
  set -euo pipefail
  cd $REPO_DIR
  test \"\$(git rev-parse HEAD)\" = \"$EXPECTED_HEAD\"
  export PYTHONPATH=$REPO_DIR PYTHONUNBUFFERED=1
  export HF_HOME=/iopsstor/scratch/cscs/xyixuan/.cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
  export WANDB_DISABLED=true VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_DISABLE_PYNCCL=1
  if [ ! -r $EXPORT_DIR/model.safetensors.index.json ]; then
    test -x $MCORE_PYTHON
    $MCORE_PYTHON -c \"import torch, causal_conv1d_cuda; from megatron.bridge import AutoBridge\"
    rm -rf $EXPORT_DIR
    $MCORE_PYTHON -m torch.distributed.run --nproc-per-node=4 tools/export_megatron_to_hf.py --hf-base $AP_HF_BASE --megatron-ckpt $STEP_DIR/policy/weights --out $EXPORT_DIR --tokenizer $AP_TOKENIZER --tp 4 > $AP_BENCH_RUN_DIR/export.log 2>&1
  fi
  export AP_CKPT=$EXPORT_DIR AP_TOKENIZER=$AP_TOKENIZER AP_EVAL_DATA=$AP_EVAL_DATA AP_ANSWER_MARKER=$AP_ANSWER_MARKER
  export AP_RUN_DIR=$AP_BENCH_RUN_DIR/eval-$AP_EVAL_TAG-$AP_ANSWER_MARKER-\$SLURM_JOB_ID
  mkdir -p \"\$AP_RUN_DIR\"
  /opt/nemo_rl_venv/bin/python -m examples.run_eval --config $AP_EVAL_CONFIG 2>&1 | tee \"\$AP_RUN_DIR/eval.log\"
  echo apertus_bench_eval=DONE tag=$AP_EVAL_TAG run_dir=\$AP_RUN_DIR
'"
