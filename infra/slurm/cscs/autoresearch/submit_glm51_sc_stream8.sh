#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
SOURCE_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)

export GLM_RECIPE=${GLM_RECIPE:-$REPO_DIR/examples/configs/recipes/llm/autoresearch/grpo-glm5.1-136n4g-megatron-tp2pp18ep16-ready-first-stream8-fused.yaml}
export GLM_RUN_ROOT=${GLM_RUN_ROOT:-/iopsstor/scratch/cscs/xyixuan/nemo_rl_glm51_ready_first_64inf_stream8_fused/$SOURCE_HEAD}
export SBATCH_LOG_ROOT=${SBATCH_LOG_ROOT:-$REPO_DIR/.tmp/slurm-logs/glm51-ready-first-64inf-stream8-fused/$SOURCE_HEAD}
export GLM_EXPECTED_MIN_GROUPS_FOR_STREAMING_TRAIN=8
export GLM_EXPECTED_FUSED_LINEAR_LOGPROBS=1

exec "$REPO_DIR/infra/slurm/cscs/autoresearch/submit_glm51_sc_scale.sh"
