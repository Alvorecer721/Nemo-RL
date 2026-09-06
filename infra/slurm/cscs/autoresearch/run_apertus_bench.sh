#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=${AP_EXPERIMENT_DIR:?}
EXPECTED_HEAD=${AP_EXPECTED_SOURCE_HEAD:?}
AP_CKPT=${AP_CKPT:?}
AP_TOKENIZER=${AP_TOKENIZER:?}
RECIPE=${AP_RECIPE:?}
RUN_ROOT=${AP_RUN_DIR:?}
EXPECTED_STEPS=${AP_EXPECTED_STEPS:?}
RUN_DIR=$RUN_ROOT/${NRL_SLURM_JOB_ID:?}
export AP_RUN_DIR=$RUN_DIR

[[ -r "$RECIPE" ]] || { echo "Missing recipe: $RECIPE" >&2; exit 1; }
[[ -r "$AP_CKPT/model.safetensors.index.json" ]] || { echo "Missing checkpoint: $AP_CKPT" >&2; exit 1; }
[[ -r "$AP_TOKENIZER/chat_template.jinja" ]] || { echo "Missing tokenizer: $AP_TOKENIZER" >&2; exit 1; }
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "Source HEAD changed after submission" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }

mkdir -p "$RUN_ROOT"
if ! mkdir "$RUN_DIR"; then
  echo "Refusing to reuse run directory: $RUN_DIR" >&2
  exit 1
fi
mkdir "$RUN_DIR/tb"
cd "$REPO_DIR"

AP_PHASE=environment
write_terminal() {
  local exit_code=$?
  trap - EXIT
  if [[ ! -e "$RUN_DIR/terminal.json" ]]; then
    AP_EXIT_CODE=$exit_code AP_FAILURE_PHASE=$AP_PHASE \
      /root/.local/bin/uv run --no-config --no-project --offline --python /opt/nemo_rl_venv/bin/python python - <<'PY' || true
import json
import os
from pathlib import Path

run_dir = Path(os.environ["AP_RUN_DIR"])
exit_code = int(os.environ["AP_EXIT_CODE"])
payload = {
    "source_head": os.environ["AP_EXPECTED_SOURCE_HEAD"],
    "slurm_job_id": os.environ.get("NRL_SLURM_JOB_ID"),
    "recipe": os.environ["AP_RECIPE"],
    "checkpoint": os.environ["AP_CKPT"],
    "tokenizer": os.environ["AP_TOKENIZER"],
    "runtime": "single-controller-transfer-queue",
    "expected_steps": int(os.environ["AP_EXPECTED_STEPS"]),
    "terminal_green": exit_code == 0,
    "failure_phase": None if exit_code == 0 else os.environ["AP_FAILURE_PHASE"],
    "exit_code": exit_code,
}
metrics = run_dir / "metrics.json"
if metrics.exists():
    payload["metrics_json"] = str(metrics)
(run_dir / "terminal.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  fi
  exit "$exit_code"
}
trap write_terminal EXIT

export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_REFIT_NUM_STREAMS=${NRL_REFIT_NUM_STREAMS:-2}
export PYTHONPATH=$REPO_DIR
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export UV=/root/.local/bin/uv
export UV_OFFLINE=1
unset NRL_IGNORE_VERSION_MISMATCH
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_DISABLE_PYNCCL=1
export WANDB_DISABLED=${AP_WANDB_DISABLED:-true}
unset NEMO_RL_PY_EXECUTABLES_SYSTEM

AP_PHASE=config_preflight
"$UV" run --no-config --no-project --offline --python /opt/nemo_rl_venv/bin/python python - <<'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()
resolved = OmegaConf.to_container(load_config(Path(os.environ["AP_RECIPE"])), resolve=True)
config = MasterConfig.model_validate(resolved)
validate_single_controller_config(config)
policy = config.policy
print(
    "apertus_bench_config=OK",
    f"model={policy['model_name']}",
    f"tp={policy['megatron_cfg']['tensor_model_parallel_size']}",
    f"pp={policy['megatron_cfg']['pipeline_model_parallel_size']}",
    f"vllm_tp={policy['generation']['vllm_cfg']['tensor_parallel_size']}",
    f"nodes={config.cluster['num_nodes']}",
    f"gen_nodes={policy['generation']['colocated']['resources']['num_nodes']}",
    f"steps={config.grpo.max_num_steps}",
    f"prompts={config.grpo.num_prompts_per_step}x{config.grpo.num_generations_per_prompt}",
    f"gbs={policy['train_global_batch_size']}",
    f"seq={policy['max_total_sequence_length']}",
    f"env={config.data['default']['env_name']}",
)
PY

RUN_LOG=$RUN_DIR/run.log
AP_PHASE=training
"$UV" run --no-config --no-project --offline --python /opt/nemo_rl_venv/bin/python python -m infra.slurm.cscs.autoresearch.launch_gsm8k_baked \
  --config "$RECIPE" >"$RUN_LOG" 2>&1

AP_PHASE=completion_log
grep -Fq "train step $EXPECTED_STEPS/$EXPECTED_STEPS" "$RUN_LOG"
grep -Fq "SC run complete:" "$RUN_LOG"

AP_PHASE=metrics_dump
"$UV" run --no-config --no-project --offline --python /opt/nemo_rl_venv/bin/python python tests/json_dump_tb_logs.py "$RUN_DIR/tb" \
  --output_path "$RUN_DIR/metrics.json" >"$RUN_DIR/metrics_dump.log" 2>&1
AP_PHASE=done
