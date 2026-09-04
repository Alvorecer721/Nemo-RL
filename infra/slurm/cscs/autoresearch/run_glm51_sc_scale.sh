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
EXPECTED_STEPS=${GLM_EXPECTED_STEPS:-10}
EXPECTED_MIN_GROUPS_FOR_STREAMING_TRAIN=${GLM_EXPECTED_MIN_GROUPS_FOR_STREAMING_TRAIN:?}
EXPECTED_FUSED_LINEAR_LOGPROBS=${GLM_EXPECTED_FUSED_LINEAR_LOGPROBS:?}
EXPECTED_SPEC_TOKENS=${GLM_EXPECTED_SPEC_TOKENS:-0}
EXPECTED_SPEC_METHOD=${GLM_EXPECTED_SPEC_METHOD:-none}
RUN_DIR=$RUN_ROOT/${NRL_SLURM_JOB_ID:?}
export GLM_RUN_DIR=$RUN_DIR

[[ -r "$RECIPE" ]] || { echo "Missing recipe: $RECIPE" >&2; exit 1; }
[[ $EXPECTED_FUSED_LINEAR_LOGPROBS == [01] ]] || { echo "GLM_EXPECTED_FUSED_LINEAR_LOGPROBS must be 0 or 1; got '$EXPECTED_FUSED_LINEAR_LOGPROBS'" >&2; exit 1; }
[[ -r "$GLM_CKPT/model.safetensors.index.json" ]] || { echo "Missing GLM checkpoint: $GLM_CKPT" >&2; exit 1; }
CACHE_METADATA=$(find "$MEGATRON_CACHE" -mindepth 2 -maxdepth 3 -type f -name .metadata -print -quit 2>/dev/null)
[[ -n "$CACHE_METADATA" && -r "$CACHE_METADATA" ]] || { echo "Missing converted Megatron cache: $MEGATRON_CACHE" >&2; exit 1; }
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "Source HEAD changed after submission" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status \
  --porcelain --untracked-files=no --ignore-submodules=untracked -- \
  .gitmodules \
  3rdparty \
  docker/nemo_rl_vllm0251.toml \
  examples/configs \
  examples/prompts \
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
SUBMODULE_STATUS=$(git -C "$REPO_DIR" submodule status --recursive)
INVALID_SUBMODULES=$(printf '%s\n' "$SUBMODULE_STATUS" | awk 'substr($0, 1, 1) == "-" || substr($0, 1, 1) == "+"')
[[ -z "$INVALID_SUBMODULES" ]] || {
  echo "Submodules are uninitialized or do not match gitlinks: $INVALID_SUBMODULES" >&2
  exit 1
}

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

from infra.slurm.cscs.autoresearch.validate_glm51_sc_scale import (
    load_scale_config,
    validate_scale_config,
)

profile = validate_scale_config(
    load_scale_config(Path(os.environ["GLM_RECIPE"])),
    expected_total_nodes=int(os.environ["GLM_TOTAL_NODES"]),
    expected_generation_nodes=int(os.environ["GLM_GENERATION_NODES"]),
    expected_steps=int(os.environ["GLM_EXPECTED_STEPS"]),
    expected_sampler=os.environ["GLM_EXPECTED_SAMPLER"],
    expected_min_groups_for_streaming_train=int(
        os.environ["GLM_EXPECTED_MIN_GROUPS_FOR_STREAMING_TRAIN"]
    ),
    expected_speculative_tokens=int(os.environ["GLM_EXPECTED_SPEC_TOKENS"]),
    expected_speculative_method=os.environ["GLM_EXPECTED_SPEC_METHOD"],
    expected_fused_linear_logprobs=os.environ["GLM_EXPECTED_FUSED_LINEAR_LOGPROBS"]
    == "1",
)
print(profile.describe())
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
if [[ $EXPECTED_SPEC_TOKENS -gt 0 ]]; then
  grep -Fq "[mtp] Loaded MTP draft weights" "$RUN_LOG"
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
  --speculative-tokens "$EXPECTED_SPEC_TOKENS" \
  --output "$SUMMARY_JSON" \
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
    "speculative_method": os.environ["GLM_EXPECTED_SPEC_METHOD"],
    "speculative_tokens": int(os.environ["GLM_EXPECTED_SPEC_TOKENS"]),
    "fused_linear_logprobs": os.environ["GLM_EXPECTED_FUSED_LINEAR_LOGPROBS"] == "1",
    "min_groups_for_streaming_train": int(
        os.environ["GLM_EXPECTED_MIN_GROUPS_FOR_STREAMING_TRAIN"]
    ),
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
