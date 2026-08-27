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
RUN_DIR=$RUN_ROOT/${NRL_SLURM_JOB_ID:?}
export GLM_RUN_DIR=$RUN_DIR

[[ -r "$RECIPE" ]] || { echo "Missing recipe: $RECIPE" >&2; exit 1; }
[[ -r "$GLM_CKPT/model.safetensors.index.json" ]] || { echo "Missing GLM checkpoint: $GLM_CKPT" >&2; exit 1; }
CACHE_METADATA=$(find "$MEGATRON_CACHE" -mindepth 2 -maxdepth 3 -type f -name .metadata -print -quit 2>/dev/null)
[[ -n "$CACHE_METADATA" && -r "$CACHE_METADATA" ]] || { echo "Missing converted Megatron cache: $MEGATRON_CACHE" >&2; exit 1; }
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "Source HEAD changed after submission" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=untracked)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }

# A certification attempt must never inherit traces, metrics, or a terminal
# artifact from an earlier retry of the same source SHA. Slurm job IDs are
# unique allocation identities; fail if even that directory already exists.
mkdir -p "$RUN_ROOT"
if ! mkdir "$RUN_DIR"; then
  echo "Refusing to reuse GLM attempt directory: $RUN_DIR" >&2
  exit 1
fi
mkdir "$RUN_DIR/tb" "$RUN_DIR/r3_trace"
cd "$REPO_DIR"

# Always leave a machine-readable terminal artifact. The previous run completed
# training but failed its post-run checker and therefore left no terminal.json,
# making a harness false-red indistinguishable from a crashed trainer.
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

GLM_PHASE=config_preflight
/opt/nemo_rl_venv/bin/python -m \
  infra.slurm.cscs.autoresearch.glm51_r3_10step_profile \
  --config "$RECIPE"

RUN_LOG=$RUN_DIR/run.log
GLM_PHASE=training
/opt/nemo_rl_venv/bin/python -m examples.run_grpo --config "$RECIPE" >"$RUN_LOG" 2>&1

GLM_PHASE=completion_log
grep -Eq "Step 10/10|Step: 10([^0-9]|$)" "$RUN_LOG"
if grep -Fq "R3 router replay fallback:" "$RUN_LOG"; then
  echo "Router Replay used missing-route fallback" >&2
  exit 1
fi

GLM_PHASE=route_trace_validation
/opt/nemo_rl_venv/bin/python tools/check_r3_trace.py \
  "$RUN_DIR/r3_trace" --transport-contract legacy-async \
  --require-forward-verify --require-cp-identity \
  >"$RUN_DIR/r3_trace_validation.log" 2>&1

METRICS_JSON=$RUN_DIR/metrics.json
SUMMARY_JSON=$RUN_DIR/summary.json
GLM_PHASE=metrics_dump
/opt/nemo_rl_venv/bin/python tests/json_dump_tb_logs.py "$RUN_DIR/tb" \
  --output_path "$METRICS_JSON" >"$RUN_DIR/metrics_dump.log" 2>&1
GLM_PHASE=metrics_validation
/opt/nemo_rl_venv/bin/python \
  infra/slurm/cscs/autoresearch/validate_glm51_r3_10step.py \
  "$METRICS_JSON" --train-data-dir "$RUN_DIR/tb" --output "$SUMMARY_JSON" \
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
