#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
[[ $(git -C "$AP_EXPERIMENT_DIR" rev-parse HEAD) == "$AP_EXPECTED_SOURCE_HEAD" ]] || { echo 'Source HEAD drift'; exit 1; }
[[ -z $(git -C "$AP_EXPERIMENT_DIR" status --porcelain --untracked-files=no) ]] || { echo 'Source dirty'; exit 1; }
export PYTHONPATH=$AP_EXPERIMENT_DIR
unset NRL_IGNORE_VERSION_MISMATCH
/root/.local/bin/uv run --no-config --no-project --offline --python /opt/nemo_rl_venv/bin/python python - <<'PYRUNTIME'
from infra.slurm.cscs.autoresearch.launch_gsm8k_baked import configure_baked_workers
configure_baked_workers()
print('runtime_preflight=PASS; source fingerprint and baked workers match', flush=True)
PYRUNTIME
