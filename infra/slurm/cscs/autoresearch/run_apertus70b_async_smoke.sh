#!/bin/bash

set -euo pipefail

APERTUS70B_EXPERIMENT_DIR=${APERTUS70B_EXPERIMENT_DIR:?}
APERTUS70B_CKPT=${APERTUS70B_CKPT:?}
APERTUS70B_TOKENIZER=${APERTUS70B_TOKENIZER:?}
APERTUS70B_DAPO_ARROW=${APERTUS70B_DAPO_ARROW:?}
APERTUS70B_RECIPE=${APERTUS70B_RECIPE:?}
APERTUS70B_RUN_ROOT=${APERTUS70B_RUN_ROOT:?}
APERTUS70B_RUN_ID=${APERTUS70B_RUN_ID:?}
APERTUS70B_VENV_DIR=${APERTUS70B_VENV_DIR:?}
APERTUS70B_MEGATRON_CACHE=${APERTUS70B_MEGATRON_CACHE:?}
APERTUS70B_XIELU_SITE=${APERTUS70B_XIELU_SITE:?}
APERTUS70B_EXPECTED_SOURCE_HEAD=${APERTUS70B_EXPECTED_SOURCE_HEAD:?}
APERTUS70B_HARNESS_HEAD=${APERTUS70B_HARNESS_HEAD:?}
APERTUS70B_IMAGE=${APERTUS70B_IMAGE:?}
APERTUS70B_RUNTIME_OVERLAY=${APERTUS70B_RUNTIME_OVERLAY:?}
APERTUS70B_RUN_DIR=$APERTUS70B_RUN_ROOT/run_$APERTUS70B_RUN_ID

export APERTUS70B_CKPT APERTUS70B_DAPO_ARROW APERTUS70B_EXPERIMENT_DIR
export APERTUS70B_MEGATRON_CACHE APERTUS70B_RECIPE APERTUS70B_RUN_DIR
export APERTUS70B_TOKENIZER
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER:-$(id -un)}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/nrl-matplotlib-${USER:-unknown}}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NEMO_RL_VENV_DIR=$APERTUS70B_VENV_DIR
export NRL_MEGATRON_CHECKPOINT_DIR=$APERTUS70B_MEGATRON_CACHE
export NRL_REFIT_NUM_STREAMS=${NRL_REFIT_NUM_STREAMS:-2}
export PYTHONPATH=$APERTUS70B_RUNTIME_OVERLAY:$APERTUS70B_XIELU_SITE:/opt/nemo-rl
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_DISABLE_PYNCCL=1
export WANDB_DISABLED=true

mkdir -p "$APERTUS70B_RUN_DIR" "$APERTUS70B_RUN_DIR/tb" "$MPLCONFIGDIR"
cd /opt/nemo-rl

[[ "${NEMO_RL_COMMIT:-}" == "$APERTUS70B_EXPECTED_SOURCE_HEAD" ]] || {
  echo "Exact-image source mismatch: expected $APERTUS70B_EXPECTED_SOURCE_HEAD, got ${NEMO_RL_COMMIT:-<unset>}" >&2
  exit 1
}
[[ -x /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python ]] || {
  echo "Baked Megatron worker environment is missing" >&2
  exit 1
}
[[ -x /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python ]] || {
  echo "Baked async vLLM worker environment is missing" >&2
  exit 1
}
/opt/nemo_rl_venv/bin/python - <<'PY'
from pathlib import Path
import nemo_rl

source = Path(nemo_rl.__file__).resolve()
assert source.is_relative_to(Path("/opt/nemo-rl")), source
print(f"exact_image_nemo_rl={source}")
PY

/opt/nemo_rl_venv/bin/python \
  "$APERTUS70B_EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/preflight_apertus70b_async_config.py" \
  | tee "$APERTUS70B_RUN_DIR/config_preflight.log"
grep -Fq "apertus70b_async_config_preflight=OK" "$APERTUS70B_RUN_DIR/config_preflight.log"

if [[ "${APERTUS70B_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  /opt/nemo_rl_venv/bin/python - <<'PY'
import torch
from xielu import xielu

x = torch.randn(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
alpha_p = torch.ones(1, device="cuda", dtype=torch.bfloat16, requires_grad=True)
alpha_n = torch.ones(1, device="cuda", dtype=torch.bfloat16, requires_grad=True)
xielu(x, alpha_p, alpha_n, 0.5, -1e-6).float().sum().backward()
assert x.grad is not None
assert alpha_p.grad is not None
assert alpha_n.grad is not None
print("apertus70b_training_xielu_cuda_forward_backward=OK")
PY
  /opt/nemo_rl_venv/bin/python -m pytest -q \
    --confcutdir="$APERTUS70B_EXPERIMENT_DIR/tests/unit/infra" \
    "$APERTUS70B_EXPERIMENT_DIR/tests/unit/infra/test_validate_apertus70b_async_smoke.py" \
    "$APERTUS70B_EXPERIMENT_DIR/tests/unit/infra/test_apertus70b_local_dapo.py"
  /opt/nemo_rl_venv/bin/ruff check \
    "$APERTUS70B_EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/preflight_apertus70b_async_config.py" \
    "$APERTUS70B_EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/validate_apertus70b_async_smoke.py" \
    "$APERTUS70B_EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/runtime_overlay/apertus70b_local_dapo.py" \
    "$APERTUS70B_EXPERIMENT_DIR/tests/unit/infra/test_validate_apertus70b_async_smoke.py" \
    "$APERTUS70B_EXPERIMENT_DIR/tests/unit/infra/test_apertus70b_local_dapo.py"
  /opt/nemo_rl_venv/bin/ruff format --check \
    "$APERTUS70B_EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/preflight_apertus70b_async_config.py" \
    "$APERTUS70B_EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/validate_apertus70b_async_smoke.py" \
    "$APERTUS70B_EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/runtime_overlay/apertus70b_local_dapo.py" \
    "$APERTUS70B_EXPERIMENT_DIR/tests/unit/infra/test_validate_apertus70b_async_smoke.py" \
    "$APERTUS70B_EXPERIMENT_DIR/tests/unit/infra/test_apertus70b_local_dapo.py"
  echo "apertus70b_async_preflight_only=OK"
  exit 0
fi

/opt/nemo_rl_venv/bin/python -m examples.run_grpo \
  --config "$APERTUS70B_RECIPE" \
  2>&1 | tee "$APERTUS70B_RUN_DIR/run.log"

if grep -Fq "Apertus XIELU: using eager fallback" "$APERTUS70B_RUN_DIR/run.log"; then
  echo "Apertus 70B training workers unexpectedly used eager xIELU" >&2
  exit 1
fi
grep -Fq "CUDA-fused xIELU not available" "$APERTUS70B_RUN_DIR/run.log"
if grep -Eq "Could not (apply|locate).*xIELU" "$APERTUS70B_RUN_DIR/run.log"; then
  echo "A required vLLM xIELU source patch failed open" >&2
  exit 1
fi

/opt/nemo_rl_venv/bin/python \
  "$APERTUS70B_EXPERIMENT_DIR/infra/slurm/cscs/autoresearch/validate_apertus70b_async_smoke.py" \
  --log-dir "$APERTUS70B_RUN_DIR/tb" \
  --run-log "$APERTUS70B_RUN_DIR/run.log" \
  --output "$APERTUS70B_RUN_ROOT/terminal_green_$APERTUS70B_RUN_ID.json" \
  --source-head "$APERTUS70B_EXPECTED_SOURCE_HEAD" \
  --harness-head "$APERTUS70B_HARNESS_HEAD" \
  --image "$APERTUS70B_IMAGE" \
  --run-id "$APERTUS70B_RUN_ID" \
  --steps 3 \
  --max-gen-kl 0.002 \
  | tee "$APERTUS70B_RUN_DIR/metrics_validation.log"
grep -Fq "apertus70b_async_smoke_metrics=OK" "$APERTUS70B_RUN_DIR/metrics_validation.log"
echo "apertus70b_async_e2e_smoke=OK"
