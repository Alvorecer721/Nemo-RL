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
#SBATCH --job-name=nrl-vllm0251-smoke
#SBATCH --output=logs/nrl_vllm0251_smoke_%j.out
#SBATCH --error=logs/nrl_vllm0251_smoke_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=normal
#SBATCH --export=NONE

set -euo pipefail

REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/nrl_vllm0251_smoke_${SLURM_JOB_ID}}
APERTUS_CKPT=${APERTUS_CKPT:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200}
APERTUS_TOKENIZER=${APERTUS_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Apertus-v1.5-8B-official}

[[ -r "$CONTAINER_ENV" ]] || {
  echo "Container EDF not found at CONTAINER_ENV=$CONTAINER_ENV" >&2
  echo "Create it as described in infra/slurm/cscs/README.md." >&2
  exit 1
}
mkdir -p "$RUN_DIR"
printf 'batch_host=%s\nrun_dir=%s\ncontainer_env=%s\n' \
  "$(hostname)" "$RUN_DIR" "$CONTAINER_ENV"

unset LD_PRELOAD

srun \
  --environment="$CONTAINER_ENV" \
  --ntasks=1 \
  --gpus-per-task=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    unset PYTHONPATH
    export PYTHONUNBUFFERED=1
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HF_HOME/vllm-0.25.1-smoke}
    export RUN_LOG='"$RUN_DIR"'/run.log
    export APERTUS_CKPT='"$APERTUS_CKPT"'
    export APERTUS_TOKENIZER='"$APERTUS_TOKENIZER"'
    PROBE_SCRIPT='"$REPO_DIR"'/infra/slurm/cscs/probe_nemo_rl_vllm0251_image.py

    cd /opt/nemo-rl
    /usr/local/bin/python-VllmGenerationWorker "$PROBE_SCRIPT" 2>&1 | tee "$RUN_LOG"
  '
