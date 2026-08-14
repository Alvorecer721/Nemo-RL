#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-probe
#SBATCH --output=logs/grpo_probe_fixgate_%j.out
#SBATCH --error=logs/grpo_probe_fixgate_%j.err
#SBATCH --time=01:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=debug
#SBATCH --export=NONE
#SBATCH --no-requeue

# Multi-step GRPO probe: characterize the Generation-KL-Error gauge across
# steps/batches (is 0.79 constant, length-correlated, or content-correlated?).
# Memory fix vs the first probe: vllm gpu_memory_utilization 0.55 -> 0.40.
set -euo pipefail
# Run from the repo root (`cd <repo> && sbatch <this>`) or anywhere with `sbatch --chdir=<repo> <this>`.
# REPO_DIR defaults to the job working dir ($PWD) — also where the relative #SBATCH logs/ land, so they agree.
# (sbatch copies this script to /var/spool/slurmd, so BASH_SOURCE doesn't resolve here; $PWD is the reliable handle.)
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current}  # CUDA xielu kernel for the TRAINING side; see docs/apertus-xielu.md
# Benchmarks only: set VLLM_XIELU_SITE to inject the kernel into vLLM generation,
# overriding the recipe-level kernel-free default (bench/xielu_ab.slurm's kernel arm).
VLLM_XIELU_SITE=${VLLM_XIELU_SITE:-}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-probe.yaml}
# NeMo-Gym recipes must run the dedicated runner (it binds task_to_env to the
# gym actor); everything else uses the stock GRPO entry point.
RUN_SCRIPT=${RUN_SCRIPT:-examples/run_grpo.py}
MAX_STEPS=${MAX_STEPS:-3}
# Call the uv shipped in the image: uv cache buckets are version-specific, so a personal uv
# earlier on PATH misses the seeded cache and rebuilds from source. Expanded by this shell
# into the srun command (the body below is double-quoted), so it must be set out here.
UV=${UV:-/root/.local/bin/uv}

# Fail fast if submodules are not initialized — the driver imports megatron.core and the refit uses the bridge, so an empty submodule dir otherwise surfaces as a cryptic ModuleNotFoundError mid-run.
for sm in 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym 3rdparty/Automodel-workspace/Automodel; do
  if [ -z "$(ls -A "$REPO_DIR/$sm" 2>/dev/null)" ]; then
    echo "ERROR: submodule '$sm' is empty/uninitialized. Run: git submodule update --init --recursive" >&2
    exit 1
  fi
done

unset LD_PRELOAD
srun --mpi=pmix --environment="$CONTAINER_ENV" --ntasks=1 \
  --gpus-per-task=4 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    # Pin uv for worker-venv builders too (venvs.py honors UV): a personal uv
    # on PATH can predate the project's [tool.uv] fields and fail the build.
    export UV=$UV
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    export PYTHONPATH=$REPO_DIR:$BRIDGE_ROOT/src:$XIELU_SITE
    # Persist the project + worker venvs on capstor: the image defaults both to
    # container-overlay paths that die with the job, forcing a rebuild every run.
    # Team uv cache on capstor, seeded from the image cache (complete for the
    # release lock): skips source builds and downloads on fresh checkouts, its
    # additions persist across jobs, and for checkouts on capstor uv hardlinks
    # instead of copying. Checkouts on other filesystems still work but copy.
    export UV_CACHE_DIR=${UV_CACHE_DIR:-/capstor/store/cscs/swissai/infra01/MLLM/uv-cache}
    # uv progress rendering has SIGPIPEd (exit 141) against the Slurm output stream
    # right after large first-time installs; plain log lines are enough here.
    export UV_NO_PROGRESS=1
    export UV_PROJECT_ENVIRONMENT=$REPO_DIR/.venv
    export NEMO_RL_VENV_DIR=$REPO_DIR/venvs
    cd $REPO_DIR
    # Apertus runtime guard: fail loud if nemo_rl is the stock /opt copy rather than this checkout (online GRPO otherwise silently regresses to gen-KL ~0.79).
    python -c 'from nemo_rl_apertus.runtime_guard import assert_apertus_runtime; assert_apertus_runtime()'
    "$UV" run --locked --extra mcore python $RUN_SCRIPT --config $RECIPE \
      grpo.max_num_steps=$MAX_STEPS \
      policy.generation.vllm_cfg.gpu_memory_utilization=0.40 ${VLLM_XIELU_SITE:+policy.generation.vllm_cfg.env_vars.PYTHONPATH=$VLLM_XIELU_SITE}
  " || probe_rc=$?

# Persist the KL gate evidence outside wipeable logs/: one TSV row per run so a
# probe's pass/fail survives log cleanup (sacct alone cannot prove steps ran).
{
  out_file=$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null | grep -oP 'StdOut=\K\S+')
  kl_values=$(grep -h "Generation KL Error" "$out_file" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | paste -sd, -)
  mkdir -p "$REPO_DIR/results"
  printf '%s\t%s\t%s\t%s\trc=%s\tkl=[%s]\n' \
    "$(date +%F_%T)" "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "$(basename "$RECIPE")" \
    "${probe_rc:-0}" "${kl_values:-NONE}" >> "$REPO_DIR/results/probe_ledger.tsv"
} || true

echo "END TIME: $(date)"
exit "${probe_rc:-0}"
