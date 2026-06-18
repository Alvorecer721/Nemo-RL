#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-dpov2
#SBATCH --output=logs/dpo_apertus_pilot_8n_%j.out
#SBATCH --error=logs/dpo_apertus_pilot_8n_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --no-requeue

# DPO pilot v2 — 8 nodes (TP=4 in-recipe + DP=8 across nodes), Apertus-only
# pairs, maxmin-style β=25 + average-log-probs loss. ~305 steps × ~12s ≈ 1h
# training + ~15 min init = ~1.25h walltime.

set -euo pipefail
echo "START TIME: $(date)"

REPO_DIR=${REPO_DIR:-$PWD}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/dpo-apertus1p5-8b-pilot-v2-megatron.yaml}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313}
RUNTIME_DIR=${RUNTIME_DIR:-$REPO_DIR}
GPUS_PER_NODE=4

for sm in 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized." >&2
    exit 1
  fi
done

NUM_NODES=$SLURM_JOB_NUM_NODES
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$head_node" | awk '{print $1; exit}')
if [ -z "$HEAD_IP" ]; then
  HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
fi
EXPECTED_UNITS=$((NUM_NODES * GPUS_PER_NODE))
echo "head: $head_node ($HEAD_IP), nodes: $NUM_NODES, expected worker_units: $EXPECTED_UNITS"

export HEAD_IP EXPECTED_UNITS NUM_NODES RECIPE BRIDGE_ROOT XIELU_SITE RUNTIME_DIR GPUS_PER_NODE
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
    export PYTHONPATH="${RUNTIME_DIR}:${BRIDGE_ROOT}/src:${XIELU_SITE}"
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
        uv run --locked --extra mcore examples/run_dpo.py --config "$RECIPE" cluster.num_nodes="$NUM_NODES"
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
