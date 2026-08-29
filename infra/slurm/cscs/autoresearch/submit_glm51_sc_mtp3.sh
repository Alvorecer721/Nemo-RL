#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
SOURCE_HEAD=$(git -C "$REPO_DIR" rev-parse HEAD)

export GLM_RECIPE=${GLM_RECIPE:-$REPO_DIR/examples/configs/recipes/llm/autoresearch/grpo-glm5.1-136n4g-megatron-tp2pp18ep16-ready-first-mtp3.yaml}
export GLM_RUN_ROOT=${GLM_RUN_ROOT:-/iopsstor/scratch/cscs/xyixuan/nemo_rl_glm51_ready_first_64inf_mtp3/$SOURCE_HEAD}
export SBATCH_LOG_ROOT=${SBATCH_LOG_ROOT:-$REPO_DIR/.tmp/slurm-logs/glm51-ready-first-64inf-mtp3/$SOURCE_HEAD}
export GLM_EXPECTED_SPEC_TOKENS=3
export GLM_EXPECTED_SPEC_METHOD=deepseek_mtp

exec "$REPO_DIR/infra/slurm/cscs/autoresearch/submit_glm51_sc_scale.sh"
