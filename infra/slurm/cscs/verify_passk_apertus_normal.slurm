#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-verify-n
#SBATCH --output=logs/verify_passk_apertus_normal_%j.out
#SBATCH --error=logs/verify_passk_apertus_normal_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --no-requeue
# Verifier sweep on the pass@8 rollouts. Same body as verify_passk_apertus.slurm,
# but on partition=normal so we can submit ≥2 instances (debug QoS limits to 2
# concurrent debug jobs per user).

set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}

PASSK_JSONL=${PASSK_JSONL:?PASSK_JSONL must be set}
OUT_JSONL=${OUT_JSONL:?OUT_JSONL must be set}
DEPS_DIR=${DEPS_DIR:-/iopsstor/scratch/cscs/nathanrchn/.cache/python_verify_deps}

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized." >&2
    exit 1
  fi
done

if [ ! -s "$REPO_DIR/$PASSK_JSONL" ]; then
  echo "ERROR: pass@k JSONL '$PASSK_JSONL' missing or empty." >&2
  exit 1
fi

mkdir -p "$REPO_DIR/logs" "$DEPS_DIR"

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src${XIELU_SITE:+:$XIELU_SITE}:$DEPS_DIR
    cd $REPO_DIR
    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'

    if [ ! -d '$DEPS_DIR/verifiable_instructions' ]; then
      echo '[preflight] installing verifier deps → $DEPS_DIR'
      uv run --locked --extra vllm pip install --target=$DEPS_DIR --no-cache-dir \
        'verifiable_instructions @ git+https://github.com/abukharin-nv/verifiable-instructions.git' \
        jsonschema pyyaml
    else
      echo '[preflight] verifier deps already present at $DEPS_DIR'
    fi

    uv run --locked --extra vllm python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\" || true

    uv run --locked --extra vllm python -m nemo_rl_apertus.verify_passk_offline \
      --passk-jsonl $PASSK_JSONL \
      --out $OUT_JSONL
  "

echo "END TIME: $(date)"
