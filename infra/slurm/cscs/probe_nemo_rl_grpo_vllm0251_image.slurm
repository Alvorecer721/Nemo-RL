#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#SBATCH --account=infra01
#SBATCH --job-name=nrl-vllm0251-grpo
#SBATCH --output=logs/nrl_vllm0251_grpo_%j.out
#SBATCH --error=logs/nrl_vllm0251_grpo_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE
#SBATCH --no-requeue

set -euo pipefail

REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm0251.toml}
RUN_DIR=${RUN_DIR:-$REPO_DIR/logs/nrl_vllm0251_grpo_${SLURM_JOB_ID}}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current}
GENERATION_ACTOR=${GENERATION_ACTOR:-nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker}
if [[ -z "${RECIPE:-}" ]]; then
  if [[ "$GENERATION_ACTOR" == *SGLangGenerationWorker ]]; then
    # The pinned upstream SGLang HTTP refit path is implemented by the
    # DTensor-v2 policy worker. Megatron+SGLang refit lands in a later change.
    RECIPE=/opt/nemo-rl/examples/configs/grpo_math_1B_sglang.yaml
  else
    RECIPE=/opt/nemo-rl/examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-probe.yaml
  fi
fi
if [[ -z "${POLICY_ACTOR:-}" ]]; then
  if [[ "$GENERATION_ACTOR" == *SGLangGenerationWorker ]]; then
    POLICY_ACTOR=nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
  else
    POLICY_ACTOR=nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
  fi
fi
MODEL_NAME=${MODEL_NAME:-}
MAX_STEPS=${MAX_STEPS:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
NUM_PROMPTS=${NUM_PROMPTS:-4}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MEGATRON_PP_SIZE=${MEGATRON_PP_SIZE:-1}
MEGATRON_TP_SIZE=${MEGATRON_TP_SIZE:-2}

for parallel_var in GPUS_PER_NODE MEGATRON_TP_SIZE MEGATRON_PP_SIZE; do
  parallel_value=${!parallel_var}
  if [[ ! "$parallel_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$parallel_var must be a positive integer, got $parallel_value" >&2
    exit 1
  fi
done
MODEL_PARALLEL_SIZE=$((MEGATRON_TP_SIZE * MEGATRON_PP_SIZE))
if (( GPUS_PER_NODE % MODEL_PARALLEL_SIZE != 0 )); then
  echo "TP*PP=$MODEL_PARALLEL_SIZE must divide GPUS_PER_NODE=$GPUS_PER_NODE" >&2
  exit 1
fi

[[ -r "$CONTAINER_ENV" ]] || {
  echo "Container EDF not found at CONTAINER_ENV=$CONTAINER_ENV" >&2
  exit 1
}
[[ -f "$XIELU_SITE/xielu/__init__.py" ]] || {
  echo "CUDA xIELU site not found at XIELU_SITE=$XIELU_SITE" >&2
  exit 1
}
mkdir -p "$RUN_DIR"
printf 'batch_host=%s\nrun_dir=%s\ncontainer_env=%s\n' \
  "$(hostname)" "$RUN_DIR" "$CONTAINER_ENV"

unset LD_PRELOAD
srun \
  --mpi=pmix \
  --cpu-bind=none \
  --environment="$CONTAINER_ENV" \
  --ntasks=1 \
  --gpus-per-task="$GPUS_PER_NODE" \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export PYTHONPATH='"$XIELU_SITE"'
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
    export WANDB_DISABLED=true
    export RUN_DIR='"$RUN_DIR"'
    export GENERATION_ACTOR='"$GENERATION_ACTOR"'
    export POLICY_ACTOR='"$POLICY_ACTOR"'
    export MODEL_NAME='"$MODEL_NAME"'
    export MAX_STEPS='"$MAX_STEPS"'
    export MAX_NEW_TOKENS='"$MAX_NEW_TOKENS"'
    export NUM_PROMPTS='"$NUM_PROMPTS"'
    export NUM_GENERATIONS='"$NUM_GENERATIONS"'
    export GPUS_PER_NODE='"$GPUS_PER_NODE"'
    export MEGATRON_PP_SIZE='"$MEGATRON_PP_SIZE"'
    export MEGATRON_TP_SIZE='"$MEGATRON_TP_SIZE"'
    RECIPE='"$RECIPE"'

    mkdir -p "$RUN_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
    cd /opt/nemo-rl

    extra_overrides=()
    if [[ "$GENERATION_ACTOR" == *SGLangGenerationWorker ]]; then
      extra_overrides+=(
        policy.generation.sglang_cfg.sglang_server_config.num_gpus="$GPUS_PER_NODE"
      )
    else
      extra_overrides+=(
        policy.generation.vllm_cfg.gpu_memory_utilization=0.40
        policy.generation.vllm_cfg.enforce_eager=true
      )
    fi
    if [[ -n "$MODEL_NAME" ]]; then
      extra_overrides+=(policy.model_name="$MODEL_NAME" policy.tokenizer.name="$MODEL_NAME")
    fi

    /opt/nemo_rl_venv/bin/python - <<'"'"'PY'"'"'
import os
from pathlib import Path

import nemo_rl

root = Path(nemo_rl.__file__).resolve()
assert root.is_relative_to("/opt/nemo-rl"), root
for actor in (
    os.environ["GENERATION_ACTOR"],
    os.environ["POLICY_ACTOR"],
):
    venv = Path("/opt/ray_venvs") / actor
    assert (venv / "bin/python").is_file(), venv
    assert (venv / "NEMO_RL_VENV_READY").is_file(), venv
print(f"baked_nemo_rl={root}")
print("baked_grpo_worker_venvs=OK")
PY

    if [[ "$RECIPE" == *apertus* ]]; then
      /opt/nemo_rl_venv/bin/python -c \
        "from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()"
    fi

    /opt/nemo_rl_venv/bin/python -m examples.run_grpo \
      --config "$RECIPE" \
      grpo.max_num_steps="$MAX_STEPS" \
      grpo.num_prompts_per_step="$NUM_PROMPTS" \
      grpo.num_generations_per_prompt="$NUM_GENERATIONS" \
      grpo.seq_logprob_error_threshold=1.02 \
      policy.train_global_batch_size="$((NUM_PROMPTS * NUM_GENERATIONS))" \
      policy.megatron_cfg.tensor_model_parallel_size="$MEGATRON_TP_SIZE" \
      policy.megatron_cfg.pipeline_model_parallel_size="$MEGATRON_PP_SIZE" \
      policy.generation.max_new_tokens="$MAX_NEW_TOKENS" \
      "${extra_overrides[@]}" \
      logger.log_dir="$RUN_DIR/tb" \
      logger.wandb_enabled=false \
      checkpointing.enabled=false \
      cluster.num_nodes=1 \
      cluster.gpus_per_node="$GPUS_PER_NODE" \
      2>&1 | tee "$RUN_DIR/run.log"

    if [[ "$GENERATION_ACTOR" == *SGLangGenerationWorker ]]; then
      # SGLang reports the refit contract through its HTTP responses. Require
      # successful weight updates and cache flushes throughout the run rather
      # than a NeMo-RL message that is not emitted by this backend version.
      test "$(grep -Fc "POST /update_weights_from_tensor HTTP/1.1\" 200 OK" \
        "$RUN_DIR/run.log")" -ge "$MAX_STEPS"
      test "$(grep -Fc "GET /flush_cache HTTP/1.1\" 200 OK" \
        "$RUN_DIR/run.log")" -ge "$MAX_STEPS"
    else
      grep -Fq "GPU Memory after refit complete" "$RUN_DIR/run.log"
    fi
    grep -Fq "========================= Step $MAX_STEPS/$MAX_STEPS" "$RUN_DIR/run.log"
    test "$(grep -Fc "Generation KL Error" "$RUN_DIR/run.log")" -ge "$MAX_STEPS"
    if grep -Fq "Apertus XIELU: using eager fallback" "$RUN_DIR/run.log"; then
      echo "GRPO training workers unexpectedly used the eager xIELU fallback" >&2
      exit 1
    fi
    echo "baked_grpo_refit=OK"
  '
