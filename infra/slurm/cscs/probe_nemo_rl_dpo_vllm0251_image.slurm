#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#SBATCH --account=infra01
#SBATCH --job-name=nrl-vllm0251-dpo
#SBATCH --output=logs/nrl_vllm0251_dpo_%j.out
#SBATCH --error=logs/nrl_vllm0251_dpo_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE

set -euo pipefail

# This probe deliberately runs the copy of NeMo-RL and the frozen Ray worker
# environments baked into the custom image. PYTHONPATH contains only the
# ABI-matched CUDA xIELU site used by Megatron training; do not add a checkout
# path or point NEMO_RL_VENV_DIR at a persistent checkout venv, because either
# would stop this from validating the delivered image.
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/nrl_vllm0251_dpo_${SLURM_JOB_ID}}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current}
APERTUS_CKPT=${APERTUS_CKPT:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200}
APERTUS_TOKENIZER=${APERTUS_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Apertus-v1.5-8B-official}
RECIPE=${RECIPE:-/opt/nemo-rl/examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml}
CHECKPOINT_RESUME=${CHECKPOINT_RESUME:-false}
CHECKPOINT_MODE=${CHECKPOINT_MODE:-none}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}

if [[ "$CHECKPOINT_RESUME" == "true" && "$CHECKPOINT_MODE" == "none" ]]; then
  CHECKPOINT_MODE=same_allocation
fi
case "$CHECKPOINT_MODE" in
  none | save | resume | same_allocation) ;;
  *)
    echo "Unsupported CHECKPOINT_MODE=$CHECKPOINT_MODE" >&2
    exit 1
    ;;
esac

[[ -r "$CONTAINER_ENV" ]] || {
  echo "Container EDF not found at CONTAINER_ENV=$CONTAINER_ENV" >&2
  echo "Create it as described in infra/slurm/cscs/README.md." >&2
  exit 1
}
[[ -f "$XIELU_SITE/xielu/__init__.py" ]] || {
  echo "CUDA xIELU site not found at XIELU_SITE=$XIELU_SITE" >&2
  exit 1
}

mkdir -p "$RUN_DIR"
printf 'batch_host=%s\nrun_dir=%s\ncontainer_env=%s\nxielu_site=%s\n' \
  "$(hostname)" "$RUN_DIR" "$CONTAINER_ENV" "$XIELU_SITE"

unset LD_PRELOAD

srun \
  --mpi=pmix \
  --cpu-bind=none \
  --environment="$CONTAINER_ENV" \
  --ntasks=1 \
  --gpus-per-task=4 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    unset NEMO_RL_PY_EXECUTABLES_SYSTEM
    export XIELU_SITE='"$XIELU_SITE"'
    export PYTHONPATH="$XIELU_SITE"
    export PYTHONUNBUFFERED=1
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
    export WANDB_DISABLED=true
    export RUN_DIR='"$RUN_DIR"'
    export LOG_DIR="$RUN_DIR/tb"
    export JSON_METRICS="$RUN_DIR/metrics.json"
    export RUN_LOG="$RUN_DIR/run.log"
    export CHECKPOINT_MODE='"$CHECKPOINT_MODE"'
    export CHECKPOINT_DIR='"$CHECKPOINT_DIR"'
    APERTUS_CKPT='"$APERTUS_CKPT"'
    APERTUS_TOKENIZER='"$APERTUS_TOKENIZER"'
    RECIPE='"$RECIPE"'

    mkdir -p "$LOG_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
    cd /opt/nemo-rl

    /opt/nemo_rl_venv/bin/python - 2>&1 <<'"'"'PY'"'"' | tee "$RUN_DIR/preflight.log"
import os
import subprocess
from pathlib import Path

import nemo_rl

root = Path(nemo_rl.__file__).resolve()
assert root.is_relative_to("/opt/nemo-rl"), root
mcore_venv = Path("/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker")
assert os.environ["NEMO_RL_VENV_DIR"] == "/opt/ray_venvs"
assert (mcore_venv / "bin/python").is_file(), mcore_venv
assert (mcore_venv / "NEMO_RL_VENV_READY").is_file(), mcore_venv
print(f"baked_nemo_rl={root}")
print(f"baked_mcore_venv={mcore_venv}")

# Fail before the full model load if the shared extension does not match this
# frozen Megatron worker Python/Torch/CUDA ABI. The real DPO step below
# exercises the same autograd path in all four Ray Megatron workers.
probe = r"""\
import torch
import xielu

x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
alpha_p = torch.tensor([0.8], device="cuda", dtype=torch.bfloat16, requires_grad=True)
alpha_n = torch.tensor([0.6], device="cuda", dtype=torch.bfloat16, requires_grad=True)
xielu.xielu(x, alpha_p, alpha_n, 0.5, -1e-6).float().sum().backward()
assert x.grad is not None and torch.isfinite(x.grad.float()).all()
assert alpha_p.grad is not None and alpha_n.grad is not None
print(f"training_xielu={xielu.__file__}")
print("training_xielu_cuda_forward_backward=OK")
"""
subprocess.run([str(mcore_venv / "bin/python"), "-c", probe], check=True)
PY

    run_dpo() {
      local max_steps=$1
      local run_log=$2
      local checkpoint_enabled=false
      if [[ "$CHECKPOINT_MODE" != "none" ]]; then
        checkpoint_enabled=true
      fi
      /opt/nemo_rl_venv/bin/python -m examples.run_dpo \
        --config "$RECIPE" \
        policy.model_name="$APERTUS_CKPT" \
        policy.tokenizer.name="$APERTUS_TOKENIZER" \
        dpo.max_num_steps="$max_steps" \
        dpo.val_batches=1 \
        dpo.val_period=100 \
        dpo.val_at_start=false \
        dpo.val_at_end=false \
        policy.train_global_batch_size=8 \
        policy.max_total_sequence_length=2048 \
        policy.megatron_cfg.tensor_model_parallel_size=2 \
        policy.megatron_cfg.sequence_parallel=false \
        policy.megatron_cfg.use_fused_linear_logprobs=true \
        policy.megatron_cfg.fused_linear_logprobs_chunk_size=256 \
        policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false \
        policy.megatron_cfg.checkpoint.async_save="$checkpoint_enabled" \
        policy.megatron_cfg.checkpoint.ckpt_assume_constant_structure="$checkpoint_enabled" \
        logger.tensorboard_enabled=true \
        logger.log_dir="$LOG_DIR" \
        logger.wandb_enabled=false \
        logger.monitor_gpus=true \
        checkpointing.enabled="$checkpoint_enabled" \
        checkpointing.checkpoint_dir="$CHECKPOINT_DIR" \
        checkpointing.metric_name=null \
        checkpointing.keep_top_k=2 \
        checkpointing.save_period=1 \
        checkpointing.save_optimizer=true \
        cluster.gpus_per_node=4 \
        cluster.num_nodes=1 \
        2>&1 | tee "$run_log"
    }

    metric_step=1
    if [[ "$CHECKPOINT_MODE" == "resume" ]]; then
      test -f "$CHECKPOINT_DIR/step_1/training_info.json"
      /opt/nemo_rl_venv/bin/python - "$CHECKPOINT_DIR/step_1" <<'PY'
import sys

from nemo_rl.utils.checkpoint import CheckpointManager

weights_path, optimizer_path = CheckpointManager.get_resume_paths(sys.argv[1])
assert weights_path is not None
assert optimizer_path is not None, "embedded Megatron optimizer was not detected"
print("baked_dpo_resume_optimizer_detected=OK")
PY
      printf "=== cross-allocation resume from step 1 to step 2 ===\n"
      run_dpo 2 "$RUN_LOG"
      test -f "$CHECKPOINT_DIR/step_2/training_info.json"
      metric_step=2
    else
      initial_max_steps=1
      if [[ "$CHECKPOINT_MODE" == "save" || \
            "$CHECKPOINT_MODE" == "same_allocation" ]]; then
        # Use the same two-step scheduler horizon as the restore process. A
        # one-step save followed by a two-step resume would correctly fail
        # the Megatron optimizer-scheduler consistency check.
        initial_max_steps=2
      fi
      printf "=== %s-step baked-image DPO smoke ===\n" "$initial_max_steps"
      run_dpo "$initial_max_steps" "$RUN_LOG"
      metric_step=$initial_max_steps
    fi

    if grep -Fq "Apertus XIELU: using eager fallback" "$RUN_LOG"; then
      echo "DPO workers unexpectedly used the eager xIELU fallback" >&2
      exit 1
    fi
    grep -Fq "training_xielu_cuda_forward_backward=OK" \
      "$RUN_DIR/preflight.log"

    if [[ "$CHECKPOINT_MODE" == "save" ]]; then
      test -f "$CHECKPOINT_DIR/step_1/training_info.json"
      test -f "$CHECKPOINT_DIR/step_2/training_info.json"
      echo "baked_dpo_cross_allocation_save=OK"
    elif [[ "$CHECKPOINT_MODE" == "resume" ]]; then
      /opt/nemo_rl_venv/bin/python - "$CHECKPOINT_DIR/step_2/training_info.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    training_info = json.load(f)
assert training_info["total_steps"] == 2, training_info
print("baked_dpo_cross_allocation_resume=OK")
PY
    elif [[ "$CHECKPOINT_MODE" == "same_allocation" ]]; then
      test -f "$CHECKPOINT_DIR/step_1/training_info.json"
      test -f "$CHECKPOINT_DIR/step_2/training_info.json"
      test ! -e "$RUN_DIR/initial_step_2"
      mv "$CHECKPOINT_DIR/step_2" "$RUN_DIR/initial_step_2"
      printf "=== resume baked-image DPO from step 1 to step 2 ===\n"
      run_dpo 2 "$RUN_DIR/resume.log"
      test -f "$CHECKPOINT_DIR/step_2/training_info.json"
      /opt/nemo_rl_venv/bin/python - "$CHECKPOINT_DIR/step_2/training_info.json" <<'"'"'PY'"'"'
import json
import sys

with open(sys.argv[1]) as f:
    training_info = json.load(f)
assert training_info["total_steps"] == 2, training_info
print("baked_dpo_async_checkpoint_resume=OK")
PY
    fi

    /opt/nemo_rl_venv/bin/python /opt/nemo-rl/tests/json_dump_tb_logs.py \
      "$LOG_DIR" --output_path "$JSON_METRICS"
    /opt/nemo_rl_venv/bin/python /opt/nemo-rl/tests/check_metrics.py \
      "$JSON_METRICS" "data[\"train/loss\"][\"$metric_step\"] < 100"
  '
