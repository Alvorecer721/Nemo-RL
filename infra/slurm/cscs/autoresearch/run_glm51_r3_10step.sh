#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=${GLM_EXPERIMENT_DIR:?}
EXPECTED_HEAD=${GLM_EXPECTED_SOURCE_HEAD:?}
GLM_CKPT=${GLM_CKPT:?}
MEGATRON_CACHE=${GLM_MEGATRON_CACHE:?}
RECIPE=${GLM_RECIPE:?}
RUN_DIR=${GLM_RUN_DIR:?}

[[ -r "$RECIPE" ]] || { echo "Missing recipe: $RECIPE" >&2; exit 1; }
[[ -r "$GLM_CKPT/model.safetensors.index.json" ]] || { echo "Missing GLM checkpoint: $GLM_CKPT" >&2; exit 1; }
CACHE_METADATA=$(find "$MEGATRON_CACHE" -mindepth 2 -maxdepth 3 -type f -name .metadata -print -quit 2>/dev/null)
[[ -n "$CACHE_METADATA" && -r "$CACHE_METADATA" ]] || { echo "Missing converted Megatron cache: $MEGATRON_CACHE" >&2; exit 1; }
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "Source HEAD changed after submission" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=untracked)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }

mkdir -p "$RUN_DIR/tb" "$RUN_DIR/r3_trace"
cd "$REPO_DIR"

export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NEMO_RL_VENV_DIR=${GLM_ACTOR_VENV_DIR:-/opt/ray_venvs/glm51-r3-$EXPECTED_HEAD}
export NRL_MEGATRON_CHECKPOINT_DIR=$MEGATRON_CACHE
export NRL_REFIT_NUM_STREAMS=${NRL_REFIT_NUM_STREAMS:-2}
export NRL_ROUTER_REPLAY_VALIDATE=1
export NRL_R3_TRACE=1
export NRL_R3_TRACE_STEPS=2
export NRL_R3_TRACE_SAMPLES=1
export NRL_R3_TRACE_MICROBATCHES=1
export NRL_R3_TRACE_VERIFY_FORWARD=1
export NRL_R3_TRACE_DIR=$RUN_DIR/r3_trace
export PYTHONPATH=$REPO_DIR
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_DISABLE_PYNCCL=1
export WANDB_DISABLED=true
unset NEMO_RL_PY_EXECUTABLES_SYSTEM

/opt/nemo_rl_venv/bin/python - <<'PY'
import os
from pathlib import Path

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from omegaconf import OmegaConf

register_omegaconf_resolvers()
recipe = Path(os.environ["GLM_RECIPE"])
cfg = MasterConfig(**OmegaConf.to_container(load_config(recipe), resolve=True))
megatron = cfg.policy["megatron_cfg"]
generation = cfg.policy["generation"]
vllm = generation["vllm_cfg"]

assert (cfg.cluster["num_nodes"], cfg.cluster["gpus_per_node"]) == (80, 4)
assert generation["colocated"]["resources"]["num_nodes"] == 8
assert (
    megatron["tensor_model_parallel_size"],
    megatron["pipeline_model_parallel_size"],
    megatron["expert_tensor_parallel_size"],
    megatron["expert_model_parallel_size"],
) == (2, 18, 1, 16)
assert megatron["sequence_parallel"] is True
assert (vllm["tensor_parallel_size"], vllm["expert_parallel_size"]) == (32, 32)
assert generation["refit_transport"] == "nccl_reshard"
assert cfg.policy["router_replay"]["enabled"] is True
assert cfg.grpo.max_num_steps == 10
assert cfg.grpo.async_grpo.enabled is True
assert cfg.grpo.async_grpo.max_trajectory_age_steps == 1
assert cfg.checkpointing["enabled"] is False
print("glm51_r3_10step_config=OK tp=2 pp=18 etp=1 ep=16 dense_dp=8 expert_dp=1")
PY

RUN_LOG=$RUN_DIR/run.log
/opt/nemo_rl_venv/bin/python -m examples.run_grpo --config "$RECIPE" >"$RUN_LOG" 2>&1

grep -Eq "Step 10/10|Step: 10([^0-9]|$)" "$RUN_LOG"
if grep -Fq "R3 router replay fallback:" "$RUN_LOG"; then
  echo "Router Replay used missing-route fallback" >&2
  exit 1
fi

/opt/nemo_rl_venv/bin/python tools/check_r3_trace.py \
  "$RUN_DIR/r3_trace" --require-forward-verify \
  >"$RUN_DIR/r3_trace_validation.log" 2>&1

METRICS_JSON=$RUN_DIR/metrics.json
SUMMARY_JSON=$RUN_DIR/summary.json
/opt/nemo_rl_venv/bin/python tests/json_dump_tb_logs.py "$RUN_DIR/tb" \
  --output_path "$METRICS_JSON" >"$RUN_DIR/metrics_dump.log" 2>&1
/opt/nemo_rl_venv/bin/python \
  infra/slurm/cscs/autoresearch/validate_glm51_r3_10step.py \
  "$METRICS_JSON" --train-data-dir "$RUN_DIR/tb" --output "$SUMMARY_JSON" \
  >"$RUN_DIR/metrics_validation.log" 2>&1

/opt/nemo_rl_venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["GLM_RUN_DIR"])
payload = {
    "source_head": os.environ["GLM_EXPECTED_SOURCE_HEAD"],
    "slurm_job_id": os.environ["NRL_SLURM_JOB_ID"],
    "recipe": os.environ["GLM_RECIPE"],
    "router_replay_validate": True,
    "r3_trace_steps": 2,
    "checkpointing_enabled": False,
    "terminal_green": True,
    "metrics": json.loads((run_dir / "summary.json").read_text()),
}
(run_dir / "terminal.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

echo "glm51_r3_10step=OK"
