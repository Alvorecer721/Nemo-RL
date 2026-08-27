#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=${GLM_EXPERIMENT_DIR:?}
EXPECTED_HEAD=${GLM_EXPECTED_SOURCE_HEAD:?}
GLM_CKPT=${GLM_CKPT:?}
MEGATRON_CACHE=${GLM_MEGATRON_CACHE:?}
RECIPE=${GLM_RECIPE:?}
RUN_ROOT=${GLM_RUN_DIR:?}
EXPECTED_STEPS=${GLM_EXPECTED_STEPS:-3}
RUN_DIR=$RUN_ROOT/${NRL_SLURM_JOB_ID:?}
export GLM_RUN_DIR=$RUN_DIR

[[ -r "$RECIPE" ]] || { echo "Missing recipe: $RECIPE" >&2; exit 1; }
[[ -r "$GLM_CKPT/model.safetensors.index.json" ]] || { echo "Missing GLM checkpoint: $GLM_CKPT" >&2; exit 1; }
CACHE_METADATA=$(find "$MEGATRON_CACHE" -mindepth 2 -maxdepth 3 -type f -name .metadata -print -quit 2>/dev/null)
[[ -n "$CACHE_METADATA" && -r "$CACHE_METADATA" ]] || { echo "Missing converted Megatron cache: $MEGATRON_CACHE" >&2; exit 1; }
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "Source HEAD changed after submission" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status \
  --porcelain --untracked-files=no --ignore-submodules=all -- \
  .gitmodules \
  docker/nemo_rl_vllm0251.toml \
  examples/configs/recipes/llm/autoresearch \
  examples/run_grpo_single_controller.py \
  infra/slurm/cscs/autoresearch/run_glm51_sc_scale.sh \
  infra/slurm/cscs/autoresearch/validate_glm51_r3_10step.py \
  infra/slurm/cscs/autoresearch/validate_glm51_sc_scale.py \
  nemo_rl \
  nemo_rl_apertus \
  pyproject.toml \
  ray.sub \
  tests/json_dump_tb_logs.py \
  tools/check_r3_trace.py \
  uv.lock)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }

mkdir -p "$RUN_ROOT"
if ! mkdir "$RUN_DIR"; then
  echo "Refusing to reuse GLM SingleController scale attempt: $RUN_DIR" >&2
  exit 1
fi
mkdir "$RUN_DIR/tb" "$RUN_DIR/r3_trace"
cd "$REPO_DIR"

GLM_PHASE=environment
write_failure_terminal() {
  local exit_code=$?
  trap - EXIT
  if [[ $exit_code -ne 0 && ! -e "$RUN_DIR/terminal.json" ]]; then
    GLM_FAILURE_EXIT_CODE=$exit_code GLM_FAILURE_PHASE=$GLM_PHASE \
      /opt/nemo_rl_venv/bin/python - <<'PY' || true
import json
import os
from pathlib import Path

payload = {
    "source_head": os.environ["GLM_EXPECTED_SOURCE_HEAD"],
    "slurm_job_id": os.environ.get("NRL_SLURM_JOB_ID"),
    "recipe": os.environ["GLM_RECIPE"],
    "runtime": "single-controller-transfer-queue",
    "terminal_green": False,
    "failure_phase": os.environ["GLM_FAILURE_PHASE"],
    "exit_code": int(os.environ["GLM_FAILURE_EXIT_CODE"]),
}
(Path(os.environ["GLM_RUN_DIR"]) / "terminal.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
  fi
  exit "$exit_code"
}
trap write_failure_terminal EXIT

export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NEMO_RL_VENV_DIR=${GLM_ACTOR_VENV_DIR:-/opt/ray_venvs/glm51-sc-$EXPECTED_HEAD}
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

GLM_PHASE=config_preflight
/opt/nemo_rl_venv/bin/python - <<'PY'
import os
from pathlib import Path

from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from omegaconf import OmegaConf

register_omegaconf_resolvers()
recipe = Path(os.environ["GLM_RECIPE"])
cfg = MasterConfig(**OmegaConf.to_container(load_config(recipe), resolve=True))
megatron = cfg.policy["megatron_cfg"]
generation = cfg.policy["generation"]
vllm = generation["vllm_cfg"]
expected_sampler = os.environ["GLM_EXPECTED_SAMPLER"]
expected_total_nodes = int(os.environ["GLM_TOTAL_NODES"])
expected_generation_nodes = int(os.environ["GLM_GENERATION_NODES"])
expected_steps = int(os.environ["GLM_EXPECTED_STEPS"])

assert (cfg.cluster["num_nodes"], cfg.cluster["gpus_per_node"]) == (
    expected_total_nodes,
    4,
)
assert generation["colocated"]["resources"]["num_nodes"] == expected_generation_nodes
assert (
    megatron["tensor_model_parallel_size"],
    megatron["pipeline_model_parallel_size"],
    megatron["expert_tensor_parallel_size"],
    megatron["expert_model_parallel_size"],
) == (2, 18, 1, 16)
assert megatron["sequence_parallel"] is True
assert (vllm["tensor_parallel_size"], vllm["expert_parallel_size"]) == (32, 32)
assert expected_generation_nodes * 4 % vllm["tensor_parallel_size"] == 0
expected_vllm_dp = expected_generation_nodes * 4 // vllm["tensor_parallel_size"]
assert expected_total_nodes - expected_generation_nodes == 72
assert generation["refit_transport"] == "nccl_reshard"
assert cfg.policy["router_replay"]["enabled"] is True
assert cfg.policy["max_total_sequence_length"] == 4096
assert generation["max_new_tokens"] == 3584
assert vllm["max_model_len"] == 4096
assert vllm["gpu_memory_utilization"] == 0.60
assert cfg.grpo.max_num_steps == expected_steps
assert cfg.grpo.async_grpo is None
assert cfg.grpo.use_dynamic_sampling is False
assert cfg.loss_fn.use_importance_sampling_correction is True
assert cfg.loss_fn.force_on_policy_ratio is False
assert cfg.data_plane["enabled"] is True
assert cfg.data_plane["impl"] == "transfer_queue"
assert cfg.data_plane["backend"] == "simple"
assert cfg.async_rl.sampler.name == expected_sampler
assert cfg.async_rl.sampler.max_staleness_versions == 1
assert cfg.async_rl.min_groups_for_streaming_train == 16
assert cfg.async_rl.max_inflight_prompts == 32
assert cfg.async_rl.max_buffered_rollouts == 128
assert cfg.checkpointing["enabled"] is False
print(
    "glm51_sc_scale_config=OK tp=2 pp=18 etp=1 ep=16 "
    "dense_dp=8 expert_dp=1 total_seq=4096 max_new=3584 "
    f"vllm_tp=32 vllm_dp={expected_vllm_dp} "
    f"transport=transfer-queue sampler={expected_sampler} steps={expected_steps}"
)
PY

RUN_LOG=$RUN_DIR/run.log
GLM_PHASE=training
/opt/nemo_rl_venv/bin/python -m examples.run_grpo_single_controller \
  --config "$RECIPE" >"$RUN_LOG" 2>&1

GLM_PHASE=completion_log
grep -Fq "train step $EXPECTED_STEPS/$EXPECTED_STEPS" "$RUN_LOG"
grep -Fq "SC run complete:" "$RUN_LOG"
if grep -Fq "R3 router replay fallback:" "$RUN_LOG"; then
  echo "Router Replay used missing-route fallback" >&2
  exit 1
fi

GLM_PHASE=route_trace_validation
/opt/nemo_rl_venv/bin/python tools/check_r3_trace.py \
  "$RUN_DIR/r3_trace" --transport-contract transfer-queue \
  --require-forward-verify --require-cp-identity \
  >"$RUN_DIR/r3_trace_validation.log" 2>&1

METRICS_JSON=$RUN_DIR/metrics.json
SUMMARY_JSON=$RUN_DIR/summary.json
GLM_PHASE=metrics_dump
/opt/nemo_rl_venv/bin/python tests/json_dump_tb_logs.py "$RUN_DIR/tb" \
  --output_path "$METRICS_JSON" >"$RUN_DIR/metrics_dump.log" 2>&1
GLM_PHASE=metrics_validation
/opt/nemo_rl_venv/bin/python \
  infra/slurm/cscs/autoresearch/validate_glm51_sc_scale.py \
  "$METRICS_JSON" --expected-steps "$EXPECTED_STEPS" \
  --train-data-dir "$RUN_DIR/tb" --output "$SUMMARY_JSON" \
  >"$RUN_DIR/metrics_validation.log" 2>&1

GLM_PHASE=terminal_artifact
/opt/nemo_rl_venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["GLM_RUN_DIR"])
payload = {
    "source_head": os.environ["GLM_EXPECTED_SOURCE_HEAD"],
    "slurm_job_id": os.environ["NRL_SLURM_JOB_ID"],
    "recipe": os.environ["GLM_RECIPE"],
    "runtime": "single-controller-transfer-queue",
    "sampler": os.environ["GLM_EXPECTED_SAMPLER"],
    "training_nodes": int(os.environ["GLM_TOTAL_NODES"])
    - int(os.environ["GLM_GENERATION_NODES"]),
    "generation_nodes": int(os.environ["GLM_GENERATION_NODES"]),
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

echo "glm51_sc_scale=OK"
