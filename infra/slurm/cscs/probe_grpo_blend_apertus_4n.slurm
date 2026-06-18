#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-blendgrpo
#SBATCH --output=logs/grpo_blend_apertus_4n_%j.out
#SBATCH --error=logs/grpo_blend_apertus_4n_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --no-requeue

# Mixed-env GRPO smoke probe — 4 nodes, 7 judge-free NeMo-Gym agents, 5 steps.
# Validates: registry → vLLM HTTP server → gym dispatch per agent_ref →
# per-agent verifier reward → GRPO step. Smoke run only; tune later.

set -euo pipefail
echo "START TIME: $(date)"

REPO_DIR=${REPO_DIR:-$PWD}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/grpo-blend-apertus1p5-8b-4n-megatron.yaml}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
GYM_ROOT=${GYM_ROOT:-$REPO_DIR/3rdparty/Gym-workspace/Gym}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}
RUNTIME_DIR=${RUNTIME_DIR:-$REPO_DIR}
DATA_JSONL=${DATA_JSONL:-$REPO_DIR/nemo_rl_apertus/data/blend_judge_free_smoke.jsonl}
GPUS_PER_NODE=4

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized." >&2
    exit 1
  fi
done

if [ ! -s "$DATA_JSONL" ]; then
  echo "ERROR: data JSONL '$DATA_JSONL' missing. Run nemo_rl_apertus/data/build_blend_judge_free.py first." >&2
  exit 1
fi

NUM_NODES=$SLURM_JOB_NUM_NODES
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$head_node" | awk '{print $1; exit}')
if [ -z "$HEAD_IP" ]; then
  HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
fi
EXPECTED_UNITS=$((NUM_NODES * GPUS_PER_NODE))
echo "head: $head_node ($HEAD_IP), nodes: $NUM_NODES, expected worker_units: $EXPECTED_UNITS"

export HEAD_IP EXPECTED_UNITS NUM_NODES RECIPE BRIDGE_ROOT GYM_ROOT XIELU_SITE RUNTIME_DIR GPUS_PER_NODE
export SLURM_EXPORT_ENV=ALL

unset LD_PRELOAD
srun \
  --mpi=pmix \
  --environment="$CONTAINER_ENV" \
  --ntasks-per-node=1 \
  --gpus-per-node="$GPUS_PER_NODE" \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -c '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=/iopsstor/scratch/cscs/${USER}/.cache/huggingface
    export PYTHONPATH="${RUNTIME_DIR}:${BRIDGE_ROOT}/src:${GYM_ROOT}:${XIELU_SITE}"
    RAY_RES="{\"worker_units\": ${GPUS_PER_NODE}, \"slurm_managed_ray_cluster\": 1}"
    RAY_PORT=6379

    if [ "${SLURM_PROCID}" -eq 0 ]; then
        ray start --head --port=$RAY_PORT --node-ip-address="$HEAD_IP" \
            --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
            --min-worker-port=53100 --max-worker-port=54100 \
            --disable-usage-stats
        units=0
        for _ in $(seq 1 240); do
            units=$(ray status 2>/dev/null | sed -n "s#.*/\([0-9]*\)\.[0-9]* worker_units.*#\1#p" | head -1)
            [ "${units:-0}" -eq "$EXPECTED_UNITS" ] && break
            sleep 5
        done
        if [ "${units:-0}" -ne "$EXPECTED_UNITS" ]; then
            echo "[FATAL] ray cluster incomplete: ${units:-0}/$EXPECTED_UNITS worker_units" >&2
            exit 1
        fi
        echo "=== ray cluster complete: ${units}/$EXPECTED_UNITS units; starting driver ==="
        cd "$RUNTIME_DIR"
        python -c "from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()"
        uv run --locked --extra mcore --extra nemo_gym examples/run_grpo_blend_apertus.py --config "$RECIPE" cluster.num_nodes="$NUM_NODES"
        ray stop
    else
        until (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do sleep 3; done
        exec 3>&- || true
        ray start --address="${HEAD_IP}:${RAY_PORT}" \
            --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
            --min-worker-port=53100 --max-worker-port=54100 \
            --disable-usage-stats
        while (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; do
            exec 3>&- || true
            sleep 10
        done
        echo "[worker ${SLURM_PROCID}] head gone; stopping ray"
        ray stop || true
    fi
  '
echo "END TIME: $(date)"
