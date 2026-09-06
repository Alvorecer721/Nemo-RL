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
#SBATCH --job-name=nrl-vllm0251-tp4
#SBATCH --output=logs/nrl_vllm0251_tp4_%j.out
#SBATCH --error=logs/nrl_vllm0251_tp4_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE
#SBATCH --no-requeue

set -euo pipefail

REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/nrl_vllm0251_tp4_${SLURM_JOB_ID}}
APERTUS_CKPT=${APERTUS_CKPT:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200}
APERTUS_TOKENIZER=${APERTUS_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Apertus-v1.5-8B-official}
PROBE_SCRIPT=${PROBE_SCRIPT:-$REPO_DIR/infra/slurm/cscs/probe_nemo_rl_vllm0251_image.py}

[[ -r "$CONTAINER_ENV" ]] || {
  echo "Container EDF not found at CONTAINER_ENV=$CONTAINER_ENV" >&2
  exit 1
}
[[ -r "$PROBE_SCRIPT" ]] || {
  echo "TP4 probe script not found at PROBE_SCRIPT=$PROBE_SCRIPT" >&2
  exit 1
}
mkdir -p "$RUN_DIR"
printf 'batch_host=%s\nrun_dir=%s\ncontainer_env=%s\n' \
  "$(hostname)" "$RUN_DIR" "$CONTAINER_ENV"

unset LD_PRELOAD
srun \
  --cpu-bind=none \
  --environment="$CONTAINER_ENV" \
  --ntasks=1 \
  --gpus-per-task=4 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    unset PYTHONPATH
    unset VLLM_DISABLE_PYNCCL
    unset VLLM_ALLREDUCE_USE_SYMM_MEM
    export PYTHONUNBUFFERED=1
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HF_HOME/vllm-0.25.1-tp4}
    export RUN_LOG='"$RUN_DIR"'/run.log
    export APERTUS_CKPT='"$APERTUS_CKPT"'
    export APERTUS_TOKENIZER='"$APERTUS_TOKENIZER"'
    export PROBE_SCRIPT='"$PROBE_SCRIPT"'
    export TENSOR_PARALLEL_SIZE=4
    export ENFORCE_EAGER=false
    export PROBE_NUM_PROMPTS=4
    export PROBE_MAX_TOKENS=64
    cd /opt/nemo-rl
    /usr/local/bin/python-VllmGenerationWorker "$PROBE_SCRIPT" \
      2>&1 | tee "$RUN_LOG"

    grep -Fq "baked_nemo_rl=/opt/nemo-rl/nemo_rl/__init__.py" "$RUN_LOG"
    grep -Fq "tensor_parallel_size=4" "$RUN_LOG"
    grep -Fq "enforce_eager=False" "$RUN_LOG"
    grep -Fq "FlashInfer MNNVL multicast is unavailable" "$RUN_LOG"
    grep -Fq "backend=trtllm" "$RUN_LOG"
    grep -Fq "Capturing CUDA graphs" "$RUN_LOG"
    grep -Fq "generated_prompts=4" "$RUN_LOG"
    grep -Fq "generated_tokens=256" "$RUN_LOG"
    grep -Fq "model_generation=OK" "$RUN_LOG"
    echo "baked_vllm_tp4=OK"
  '
