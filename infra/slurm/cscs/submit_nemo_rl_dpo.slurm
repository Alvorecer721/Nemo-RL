#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-train
#SBATCH --output=logs/dpo_maxmin_%j.out
#SBATCH --error=logs/dpo_maxmin_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --export=NONE
#SBATCH --no-requeue

set -euo pipefail
echo "START TIME: $(date)"

# ~1,921 steps x ~30-45 s exceeds one 12 h window: chain windows with
# singleton dependency; auto-resume picks up from the latest checkpoint.
AUTO_REQUEUE=${AUTO_REQUEUE:-true}
if [ "$AUTO_REQUEUE" = true ]; then
    sbatch --dependency=singleton --job-name="$SLURM_JOB_NAME" "$0"
fi

###############################################################################
# Apertus-1.5 8B DPO — MaxMin full set, reference-identical hyperparameters.
# Science lives in the RECIPE yaml (no science overrides here); this script
# owns the cluster only. gbs is config-level: 1 node executes the identical
# training math as N nodes (DP width changes, updates do not).
###############################################################################
# Run from the repo root (`cd <repo> && sbatch <this>`) or anywhere with `sbatch --chdir=<repo> <this>`;
# the relative #SBATCH logs/ paths land in the same workdir.
REPO_DIR=${REPO_DIR:-$PWD}
# NeMo-RL runtime: this checkout by default (Apertus deltas + bridge pin);
# set /opt/nemo-rl to run the image's stock tree instead.
RUNTIME_DIR=${RUNTIME_DIR:-$REPO_DIR}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-0.1.0-cp313-torch2.11.0-cu130}

FULL_SET=/capstor/store/cscs/swissai/infra01/datasets/alignment/preference_datasets/MaxMin_Tr_3600-FD-nemo-rl-binpref-full/train.parquet
SRC_SET=/capstor/store/cscs/swissai/infra01/datasets/alignment/preference_datasets/MaxMin_Tr_3600-Filtered-Decontaminated

unset LD_PRELOAD
srun \
  --mpi=pmix \
  --environment="$CONTAINER_ENV" \
  --ntasks=1 \
  --gpus-per-task=4 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HOME=/iopsstor/scratch/cscs/${USER}/.cache/huggingface
    export PYTHONPATH="'"$REPO_DIR"':'"$BRIDGE_ROOT"'/src:'"$XIELU_SITE"'"
    # Persist the project + worker venvs on capstor (skip for the image tree,
    # whose /opt venvs are prebaked): the image defaults both to container-overlay
    # paths that die with the job, forcing a rebuild every run.
    if [ "'"$RUNTIME_DIR"'" != /opt/nemo-rl ]; then
      # Image-seeded uv cache (complete for the release lock; see fixgate launcher).
      export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
      export UV_PROJECT_ENVIRONMENT="'"$RUNTIME_DIR"'/.venv"
      export NEMO_RL_VENV_DIR="'"$RUNTIME_DIR"'/venvs"
    fi
    cd "'"$RUNTIME_DIR"'"

    # Prologue: full-set parquet (all 245,898 pairs, reference data parity).
    if [ ! -f "'"$FULL_SET"'" ]; then
      echo "=== building full-set train parquet ==="
      uv run --locked python - <<PYEOF
from datasets import load_from_disk
from pathlib import Path
ds = load_from_disk("'"$SRC_SET"'")["train_split"]
def to_binpref(r):
    return {"prompt": r["chosen"][:-1],
            "chosen": r["chosen"][-1]["content"],
            "rejected": r["rejected"][-1]["content"]}
out = ds.map(to_binpref, remove_columns=ds.column_names, keep_in_memory=True)
dst = Path("'"$FULL_SET"'"); dst.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(dst)
print(f"full set: {len(out)} pairs -> {dst}")
PYEOF
    fi

    echo "=== DPO training: $(basename "'"$RECIPE"'") ==="
    uv run --locked examples/run_dpo.py --config "'"$RECIPE"'"
  '
echo "END TIME: $(date)"
