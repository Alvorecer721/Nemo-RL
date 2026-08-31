#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_DIR=${REFIT_SC_REPO_DIR:?}
EXPECTED_HEAD=${REFIT_SC_EXPECTED_HEAD:?}
ARM_DIR=${REFIT_SC_ARM_DIR:?}
ARM_NAME=${REFIT_SC_ARM_NAME:?}
EXPECTED_STEPS=${REFIT_SC_STEPS:-12}
ARM_TIMEOUT_S=${REFIT_SC_ARM_TIMEOUT_S:-900}
RUN_LOG=$ARM_DIR/run.log

[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || {
  echo "Source HEAD changed after submission" >&2
  exit 1
}
SOURCE_STATUS=$(git -C "$REPO_DIR" status --porcelain --untracked-files=no --ignore-submodules=all)
[[ -z "$SOURCE_STATUS" ]] || { echo "Tracked source is dirty: $SOURCE_STATUS" >&2; exit 1; }

mkdir "$ARM_DIR"
cd "$REPO_DIR"

# Reuse the certified dependency image as-is. The experiment changes only Python
# source/configuration, so Ray actors use that image's Python and import this checkout.
export PYTHONPATH=$REPO_DIR
export PYTHONUNBUFFERED=1
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
# PY_EXECUTABLES.SYSTEM is the literal command ``python``. Put the certified
# environment first so Ray resolves that command to the same dependency layer as
# the driver, rather than the image's dependency-light base interpreter.
export VIRTUAL_ENV=/opt/nemo_rl_venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-$HF_HOME/nemo_rl/Qwen/Qwen3-0.6B}
export NRL_XFERDTENSOR_PYTHON=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export RAY_DEDUP_LOGS=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_DISABLE_PYNCCL=1
export WANDB_DISABLED=true

[[ $(command -v python) == /opt/nemo_rl_venv/bin/python ]] || {
  echo "Certified Python is not first on PATH: $(command -v python)" >&2
  exit 1
}
python -c 'import megatron, vllm; print("certified_actor_imports=OK")'

printf 'arm=%s\nhead=%s\nsteps=%s\nstreams=%s\nimplicit_order=%s\n' \
  "$ARM_NAME" "$EXPECTED_HEAD" "$EXPECTED_STEPS" \
  "${NRL_REFIT_NUM_STREAMS:?}" "${NCCL_LAUNCH_ORDER_IMPLICIT:?}"

set +e
timeout --signal=TERM --kill-after=30s "${ARM_TIMEOUT_S}s" \
  /opt/nemo_rl_venv/bin/python -m examples.run_grpo_single_controller \
    --config "$REPO_DIR/examples/configs/grpo_math_1B_megatron_single_controller.yaml" \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    grpo.seq_logprob_error_threshold=1000 \
    grpo.max_num_steps="$EXPECTED_STEPS" \
    grpo.val_period=-1 \
    grpo.val_at_start=false \
    grpo.val_at_end=false \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=512 \
    policy.megatron_cfg.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.pipeline_model_parallel_size=2 \
    policy.dtensor_cfg.enabled=false \
    policy.generation.backend=vllm \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.async_engine=true \
    policy.generation.refit_transport=nccl_reshard \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=3 \
    checkpointing.enabled=false \
    logger.log_dir="$ARM_DIR/tb" \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=true \
    logger.monitor_gpus=false \
    data_plane.enabled=true \
    data_plane.impl=transfer_queue \
    data_plane.backend=simple \
    async_rl.sampler.name=in_order \
    async_rl.sampler.max_lookahead_versions=1 \
    async_rl.min_groups_for_streaming_train=2 \
    async_rl.max_inflight_prompts=4 \
    async_rl.max_buffered_rollouts=4 \
    async_rl.generation_fleet_health.enabled=false \
    async_rl.generation_fleet_health.refit_timeout_s=90 \
    async_rl.stall_watchdog.interval_s=10 \
    async_rl.stall_watchdog.stall_timeout_s=180 \
    async_rl.stall_watchdog.stall_action=abort \
    >"$RUN_LOG" 2>&1
exit_code=$?
set -e

sync_count=$(grep -Fc '_sync_weights: sync done in' "$RUN_LOG" 2>/dev/null || true)
if [[ $exit_code -eq 0 ]] && \
   grep -Fq "train step $EXPECTED_STEPS/$EXPECTED_STEPS" "$RUN_LOG" && \
   grep -Fq 'SC run complete:' "$RUN_LOG" && \
   (( sync_count >= EXPECTED_STEPS )); then
  status=pass
elif grep -Eq 'RefitAborted|deadline exceeded|STALL DETECTED' "$RUN_LOG" || \
     [[ $exit_code -eq 124 || $exit_code -eq 137 || $exit_code -eq 143 ]]; then
  status=timeout
else
  status=error
fi

printf 'arm=%s\nexit_code=%s\nstatus=%s\nsync_count=%s\n' \
  "$ARM_NAME" "$exit_code" "$status" "$sync_count" >"$ARM_DIR/summary.txt"
echo "REFIT_SC_PP2=$status arm=$ARM_NAME exit_code=$exit_code sync_count=$sync_count"

if [[ "$status" == pass ]]; then
  exit 0
fi
exit 1
