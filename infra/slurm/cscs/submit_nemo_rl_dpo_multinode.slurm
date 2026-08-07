#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-train
#SBATCH --output=logs/dpo_maxmin_mn_%j.out
#SBATCH --error=logs/dpo_maxmin_mn_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --export=NONE
#SBATCH --no-requeue
#SBATCH --dependency=singleton

set -euo pipefail
echo "START TIME: $(date)"

###############################################################################
# Multi-node NeMo-RL DPO via a manually-assembled Ray cluster.
# CSCS-native shape: ONE srun, one task (= one CE container) per node.
#   task 0  -> ray head + driver (run_dpo.py) + ray stop
#   task 1+ -> ray worker (--block; dies when the step ends)
# Same recipe/checkpoint_dir as the 1-node launcher: gbs is config-level, so
# N nodes execute identical training math (DP width changes, updates do not),
# and the run auto-resumes from the latest checkpoint.
###############################################################################
# Run from the repo root (`cd <repo> && sbatch <this>`) or anywhere with `sbatch --chdir=<repo> <this>`;
# the relative #SBATCH logs/ paths land in the same workdir.
REPO_DIR=${REPO_DIR:-$PWD}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml}
# The Ray-cluster assembly is algorithm-agnostic; point RUN_SCRIPT at another
# entry point (e.g. examples/run_grpo.py, with UV_EXTRA=mcore) to reuse it.
RUN_SCRIPT=${RUN_SCRIPT:-examples/run_dpo.py}
UV_EXTRA=${UV_EXTRA:-}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current}
RUNTIME_DIR=${RUNTIME_DIR:-$REPO_DIR}
# Raw Megatron base/reference loading (pretrained_checkpoint is native in
# v0.7.0): set PRETRAINED_CKPT_PATH to an iter_XXXXXXX dir to skip the HF
# conversion cache.
PRETRAINED_CKPT_PATH=${PRETRAINED_CKPT_PATH:-}
PRETRAINED_CKPT_FORMAT=${PRETRAINED_CKPT_FORMAT:-megatron_lm}
# Extra hydra overrides, inheritable at submission (`sbatch --export=NONE,EXTRA_OVERRIDES=...`)
# for short drills / one-offs without editing the recipe.
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
if [ -n "$PRETRAINED_CKPT_PATH" ]; then
    EXTRA_OVERRIDES="$EXTRA_OVERRIDES +checkpointing.pretrained_checkpoint.path=$PRETRAINED_CKPT_PATH +checkpointing.pretrained_checkpoint.format=$PRETRAINED_CKPT_FORMAT"
fi
GPUS_PER_NODE=4

NUM_NODES=$SLURM_JOB_NUM_NODES
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$head_node" | awk '{print $1; exit}')
if [ -z "$HEAD_IP" ]; then
    HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
fi
EXPECTED_UNITS=$((NUM_NODES * GPUS_PER_NODE))
echo "head: $head_node ($HEAD_IP), nodes: $NUM_NODES, expected worker_units: $EXPECTED_UNITS"

export HEAD_IP EXPECTED_UNITS NUM_NODES RECIPE REPO_DIR BRIDGE_ROOT XIELU_SITE RUNTIME_DIR GPUS_PER_NODE EXTRA_OVERRIDES RUN_SCRIPT UV_EXTRA
# sbatch --export=<list> propagates to srun via SLURM_EXPORT_ENV and would
# strip the exports above from the tasks; force full propagation.
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
    export PYTHONPATH="${REPO_DIR}:${BRIDGE_ROOT}/src:${XIELU_SITE}"
    # Persist the project + worker venvs on capstor (skip for the image tree,
    # whose /opt venvs are prebaked): the image defaults both to container-overlay
    # paths that die with the job, forcing a rebuild every run.
    # Call the uv shipped in the image (uv cache buckets are version-specific).
    UV=${UV:-/root/.local/bin/uv}
    if [ "${RUNTIME_DIR}" != /opt/nemo-rl ]; then
      # Team uv cache on capstor (complete for the release lock; see fixgate launcher).
      export UV_CACHE_DIR=${UV_CACHE_DIR:-/capstor/store/cscs/swissai/infra01/MLLM/uv-cache}
      # uv progress rendering has SIGPIPEd (exit 141) against the Slurm output stream
      # right after large first-time installs; plain log lines are enough here.
      export UV_NO_PROGRESS=1
      export UV_PROJECT_ENVIRONMENT="${RUNTIME_DIR}/.venv"
      export NEMO_RL_VENV_DIR="${RUNTIME_DIR}/venvs"
    fi
    RAY_RES="{\"worker_units\": ${GPUS_PER_NODE}, \"slurm_managed_ray_cluster\": 1}"
    RAY_PORT=6379

    if [ "${SLURM_PROCID}" -eq 0 ]; then
        ray start --head --port=$RAY_PORT --node-ip-address="$HEAD_IP" \
            --num-gpus="$GPUS_PER_NODE" --resources="$RAY_RES" \
            --min-worker-port=53100 --max-worker-port=54100 \
            --disable-usage-stats
        # Wait until every node has contributed its worker_units.
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
        # shellcheck disable=SC2086  # EXTRA_OVERRIDES is intentionally word-split
        "$UV" run --locked ${UV_EXTRA:+--extra $UV_EXTRA} "$RUN_SCRIPT" --config "$RECIPE" cluster.num_nodes="$NUM_NODES" $EXTRA_OVERRIDES
        ray stop
    else
        # Workers: wait for the head port, join, then live exactly as long as
        # the head does (polling the port; --block would outlive the driver
        # and hang the step until walltime).
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
