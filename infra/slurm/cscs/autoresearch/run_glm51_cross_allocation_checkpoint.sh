#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PHASE=${GLM_RESUME_PHASE:?Set GLM_RESUME_PHASE to save or resume}
case "$PHASE" in
  save | resume) ;;
  *) echo "Unsupported phase: $PHASE" >&2; exit 2 ;;
esac

REPO_DIR=${GLM_EXPERIMENT_DIR:?}
EXPECTED_HEAD=${GLM_EXPECTED_SOURCE_HEAD:?}
GLM_CKPT=${GLM_CKPT:?}
CHECKPOINT_DIR=${GLM_RESUME_CHECKPOINT_DIR:?}
RUN_ROOT=${GLM_RUN_ROOT:?}
MEGATRON_CACHE=${GLM_MEGATRON_CACHE:?}
MEGATRON_SOURCE_OVERLAY=${GLM_MEGATRON_SOURCE_OVERLAY:-}
EXPECTED_MEGATRON_SOURCE_HEAD=${GLM_EXPECTED_MEGATRON_SOURCE_HEAD:-}
export GLM_NUM_NODES=${GLM_NUM_NODES:-80}
STALL_SECONDS=${GLM_CHECKPOINT_STALL_SECONDS:-1200}
POLL_SECONDS=${GLM_CHECKPOINT_POLL_SECONDS:-60}
RUN_DIR=$RUN_ROOT/$PHASE
RUN_LOG=$RUN_DIR/run.log
MONITOR_LOG=$RUN_DIR/checkpoint_progress.tsv
DIAGNOSTIC_LOG=$RUN_DIR/checkpoint_stall_diagnostics.txt
NODE_DIAGNOSTIC_SCRIPT=$REPO_DIR/infra/slurm/cscs/autoresearch/collect_ray_node_diagnostics.py

if [[ "$PHASE" == "save" ]]; then
  DEFAULT_RECIPE=$REPO_DIR/examples/configs/recipes/llm/autoresearch/grpo-glm5.1-80n4g-megatron-async-vllm-tp32-checkpoint-save.yaml
else
  DEFAULT_RECIPE=$REPO_DIR/examples/configs/recipes/llm/autoresearch/grpo-glm5.1-80n4g-megatron-async-vllm-tp32-checkpoint-resume.yaml
fi
RECIPE=${GLM_RECIPE:-$DEFAULT_RECIPE}
export GLM_RECIPE=$RECIPE

[[ -r "$RECIPE" ]] || { echo "Missing recipe: $RECIPE" >&2; exit 1; }
[[ -r "$GLM_CKPT/model.safetensors.index.json" ]] || { echo "Missing GLM checkpoint: $GLM_CKPT" >&2; exit 1; }
CACHE_METADATA=$(find "$MEGATRON_CACHE" -mindepth 2 -maxdepth 3 -type f -name .metadata -print -quit 2>/dev/null)
[[ -n "$CACHE_METADATA" && -r "$CACHE_METADATA" ]] || { echo "Missing converted Megatron cache: $MEGATRON_CACHE" >&2; exit 1; }
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || { echo "Source HEAD changed after submission" >&2; exit 1; }
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=untracked)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }
if [[ -n "$MEGATRON_SOURCE_OVERLAY" ]]; then
  [[ -n "$EXPECTED_MEGATRON_SOURCE_HEAD" ]] || { echo "Missing expected Megatron overlay HEAD" >&2; exit 1; }
  [[ -r "$MEGATRON_SOURCE_OVERLAY/megatron/core/optimizer/distrib_optimizer.py" ]] || { echo "Invalid Megatron source overlay: $MEGATRON_SOURCE_OVERLAY" >&2; exit 1; }
  [[ $(git -C "$MEGATRON_SOURCE_OVERLAY" rev-parse HEAD) == "$EXPECTED_MEGATRON_SOURCE_HEAD" ]] || { echo "Megatron source overlay HEAD changed after submission" >&2; exit 1; }
  MEGATRON_SOURCE_STATUS=$(git -C "$MEGATRON_SOURCE_OVERLAY" status --porcelain --untracked-files=no)
  [[ -z "$MEGATRON_SOURCE_STATUS" ]] || { echo "Megatron source overlay is dirty: $MEGATRON_SOURCE_STATUS" >&2; exit 1; }
elif [[ -n "$EXPECTED_MEGATRON_SOURCE_HEAD" ]]; then
  echo "Expected Megatron overlay HEAD set without an overlay path" >&2
  exit 1
fi
if [[ "$PHASE" == "save" ]]; then
  [[ ! -e "$CHECKPOINT_DIR" ]] || { echo "Refusing existing checkpoint namespace: $CHECKPOINT_DIR" >&2; exit 1; }
else
  [[ -r "$CHECKPOINT_DIR/step_1/policy/weights/iter_0000000/.metadata" ]] || { echo "Phase A checkpoint is incomplete" >&2; exit 1; }
fi

mkdir -p "$RUN_DIR" "$RUN_DIR/tb"
cd "$REPO_DIR"

export GLM_CKPT GLM_RESUME_CHECKPOINT_DIR GLM_RUN_DIR=$RUN_DIR
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# A source overlay changes the uv worker command from --directory /opt/nemo-rl
# to this worktree. Rebuilding the baked venv in place is unsafe on Enroot's
# writable overlay (uv can collide with lower-layer package directories), so
# materialize the diagnostic source environment in an empty, head-scoped path.
# The uv package cache remains image-owned and warm.
export NEMO_RL_VENV_DIR=${GLM_ACTOR_VENV_DIR:-/opt/ray_venvs/glm51-checkpoint-$EXPECTED_HEAD}
export NRL_MEGATRON_CHECKPOINT_DIR=$MEGATRON_CACHE
export NRL_REFIT_NUM_STREAMS=${NRL_REFIT_NUM_STREAMS:-2}
if [[ -n "$MEGATRON_SOURCE_OVERLAY" ]]; then
  export PYTHONPATH=$MEGATRON_SOURCE_OVERLAY:$REPO_DIR
else
  export PYTHONPATH=$REPO_DIR
fi
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
phase = os.environ["GLM_RESUME_PHASE"]
recipe = Path(os.environ["GLM_RECIPE"])
resolved = OmegaConf.to_container(load_config(recipe), resolve=True)
cfg = MasterConfig(**resolved)
megatron = cfg.policy["megatron_cfg"]
generation = cfg.policy["generation"]
vllm = generation["vllm_cfg"]
expected_nodes = int(os.environ["GLM_NUM_NODES"])
assert (cfg.cluster["num_nodes"], cfg.cluster["gpus_per_node"]) == (expected_nodes, 4)
assert generation["colocated"]["resources"]["num_nodes"] == 8
tp = megatron["tensor_model_parallel_size"]
pp = megatron["pipeline_model_parallel_size"]
etp = megatron["expert_tensor_parallel_size"]
ep = megatron["expert_model_parallel_size"]
assert (tp, pp, etp, ep) in {(1, 18, 1, 16), (2, 18, 1, 16)}
trainer_ranks = (expected_nodes - 8) * 4
dense_model_parallel_size = tp * pp
expert_model_parallel_size = etp * ep * pp
assert trainer_ranks % dense_model_parallel_size == 0
assert trainer_ranks % expert_model_parallel_size == 0
dense_data_parallel_size = trainer_ranks // dense_model_parallel_size
expert_data_parallel_size = trainer_ranks // expert_model_parallel_size
if tp > 1 and ep > 1:
    assert megatron["sequence_parallel"] is True
assert (vllm["tensor_parallel_size"], vllm["expert_parallel_size"]) == (32, 32)
assert megatron["checkpoint"]["async_save"] is True
assert megatron["checkpoint"]["fully_parallel_save"] is False
assert generation["refit_transport"] == "nccl_reshard"
assert cfg.checkpointing["save_optimizer"] is True
assert cfg.checkpointing["enabled"] is (phase == "save")
print(
    f"glm51_checkpoint_{phase}_config=OK "
    f"nodes={expected_nodes} tp={tp} pp={pp} etp={etp} ep={ep} "
    f"dense_dp={dense_data_parallel_size} expert_dp={expert_data_parallel_size}"
)

megatron_source_overlay = os.environ.get("GLM_MEGATRON_SOURCE_OVERLAY")
if megatron_source_overlay:
    from megatron.core.optimizer import distrib_optimizer

    imported_optimizer = Path(distrib_optimizer.__file__).resolve()
    expected_optimizer = (
        Path(megatron_source_overlay)
        / "megatron/core/optimizer/distrib_optimizer.py"
    ).resolve()
    assert imported_optimizer == expected_optimizer, (
        f"Megatron overlay mismatch: imported {imported_optimizer}, "
        f"expected {expected_optimizer}"
    )
    print(
        "glm51_megatron_source_overlay=OK "
        f"head={os.environ['GLM_EXPECTED_MEGATRON_SOURCE_HEAD']} "
        f"optimizer={imported_optimizer}"
    )
PY

if [[ "$PHASE" == "save" ]]; then
  mkdir -p "$CHECKPOINT_DIR"
fi

driver_pid=""
stop_driver() {
  if [[ -n "$driver_pid" ]] && kill -0 "$driver_pid" 2>/dev/null; then
    kill -TERM "$driver_pid" 2>/dev/null || true
  fi
}
trap stop_driver TERM HUP INT

/opt/nemo_rl_venv/bin/python -m examples.run_grpo --config "$RECIPE" >"$RUN_LOG" 2>&1 &
driver_pid=$!

if [[ "$PHASE" == "save" ]]; then
  printf 'timestamp\tshards\tbytes\n' >"$MONITOR_LOG"
  last_count=-1
  last_change=$(date +%s)
  save_started=0
  start_diagnostics_captured=0
  while kill -0 "$driver_pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    if grep -Fq "Saving checkpoint for step 1" "$RUN_LOG" 2>/dev/null; then
      save_started=1
    fi
    if [[ "$save_started" == 1 ]]; then
      if [[ "$start_diagnostics_captured" == 0 ]]; then
        /opt/nemo_rl_venv/bin/python "$NODE_DIAGNOSTIC_SCRIPT" \
          --output "$RUN_DIR/checkpoint_start_nodes.json" || true
        start_diagnostics_captured=1
      fi
      iteration_dir=$CHECKPOINT_DIR/tmp_step_1/policy/weights/iter_0000000
      if [[ ! -d "$iteration_dir" ]]; then
        iteration_dir=$CHECKPOINT_DIR/step_1/policy/weights/iter_0000000
      fi
      if [[ -d "$iteration_dir" ]]; then
        count=$(find "$iteration_dir" -maxdepth 1 -type f -name '*.distcp' | wc -l)
        bytes=$(find "$iteration_dir" -maxdepth 1 -type f -name '*.distcp' -printf '%s\n' | awk '{total += $1} END {printf "%.0f", total + 0}')
      else
        count=0
        bytes=0
      fi
      printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$count" "$bytes" >>"$MONITOR_LOG"
      if [[ "$count" -ne "$last_count" ]]; then
        last_count=$count
        last_change=$(date +%s)
      elif (( $(date +%s) - last_change >= STALL_SECONDS )); then
        /opt/nemo_rl_venv/bin/python "$NODE_DIAGNOSTIC_SCRIPT" \
          --output "$RUN_DIR/checkpoint_stall_nodes.json" || true
        {
          echo "checkpoint_stall_detected=$(date -u +%FT%TZ)"
          echo "completed_shards=$count"
          echo "completed_bytes=$bytes"
          echo "driver_pid=$driver_pid"
          free -b || true
          df -h "$CHECKPOINT_DIR" || true
          ps -eo pid,ppid,stat,pcpu,pmem,rss,etime,comm,args --sort=-rss | head -200 || true
          ray status || true
          echo "recent_checkpoint_errors:"
          grep -R -E "Traceback|AsyncRequest|checkpoint|distcp|No space|Out of memory|Killed" "${LOG_DIR:-$RUN_DIR}" 2>/dev/null | tail -1000 || true
        } >"$DIAGNOSTIC_LOG" 2>&1
        stop_driver
        wait "$driver_pid" || true
        echo "GLM checkpoint made no shard progress for ${STALL_SECONDS}s" >&2
        exit 124
      fi
    fi
  done
fi

set +e
wait "$driver_pid"
driver_rc=$?
set -e
[[ "$driver_rc" -eq 0 ]] || { echo "GLM driver failed with exit code $driver_rc" >&2; exit "$driver_rc"; }

if [[ "$PHASE" == "save" ]]; then
  iteration_dir=$CHECKPOINT_DIR/step_1/policy/weights/iter_0000000
  [[ -r "$iteration_dir/.metadata" ]] || { echo "Missing DCP metadata" >&2; exit 1; }
  [[ -r "$iteration_dir/metadata.json" ]] || { echo "Missing Megatron metadata" >&2; exit 1; }
  shard_count=$(find "$iteration_dir" -maxdepth 1 -type f -name '*.distcp' | wc -l)
  [[ "$shard_count" -eq 288 ]] || { echo "Expected 288 rank shards, found $shard_count" >&2; exit 1; }
  grep -Fq "Saving checkpoint for step 1" "$RUN_LOG"
  echo "glm51_cross_allocation_save=OK"
else
  grep -Fq "successfully loaded checkpoint" "$RUN_LOG"
  grep -Eq "Step 2/2|Step: 2([^0-9]|$)" "$RUN_LOG"
  grep -Fq "Restoring replay buffer from checkpoint" "$RUN_LOG"
  grep -Fq "Replay buffer restored from checkpoint" "$RUN_LOG"
  echo "glm51_cross_allocation_resume=OK"
fi

/opt/nemo_rl_venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["GLM_RUN_DIR"])
payload = {
    "phase": os.environ["GLM_RESUME_PHASE"],
    "source_head": os.environ["GLM_EXPECTED_SOURCE_HEAD"],
    "slurm_job_id": os.environ["NRL_SLURM_JOB_ID"],
    "checkpoint_dir": os.environ["GLM_RESUME_CHECKPOINT_DIR"],
    "ray_object_store_memory": int(os.environ["RAY_OBJECT_STORE_MEMORY"]),
    "terminal_green": True,
    "megatron_source_overlay": os.environ.get("GLM_MEGATRON_SOURCE_OVERLAY"),
    "megatron_source_overlay_head": os.environ.get(
        "GLM_EXPECTED_MEGATRON_SOURCE_HEAD"
    ),
}
(run_dir / "terminal.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
