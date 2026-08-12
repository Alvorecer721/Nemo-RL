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
#SBATCH --job-name=ap1p5-probe-v0251
#SBATCH --output=logs/grpo_probe_vllm0251_%j.out
#SBATCH --error=logs/grpo_probe_vllm0251_%j.err
#SBATCH --time=01:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE
#SBATCH --no-requeue

# Fixgate on the vLLM 0.25.1 image (nemo-rl-apertus-vllm-0.25.1-*.sqsh).
# UV_CACHE_DIR points at the image-internal cache, which the build seeded for
# the bumped lock (torch 2.11 wheels included) — the team capstor cache is
# still keyed to the old lock and would trigger source rebuilds.
set -euo pipefail
export REPO_DIR=$PWD
export CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
export UV_CACHE_DIR=/root/.cache/uv
[[ -r "$CONTAINER_ENV" ]] || {
    echo "Container EDF not found at CONTAINER_ENV=$CONTAINER_ENV" >&2
    echo "Create it as described in infra/slurm/cscs/README.md." >&2
    exit 1
}
source "$REPO_DIR/infra/slurm/cscs/probe_grpo_fixgate.slurm"
