#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-train
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/dpo_maxmin_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL/logs/dpo_maxmin_%j.err
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
REPO_DIR=${REPO_DIR:-/iopsstor/scratch/cscs/xyixuan/apertus/Nemo-RL}
# NeMo-RL runtime: image default, or a filesystem worktree (validated branch
# checkout) to run changed core/lock without rebuilding the image.
RUNTIME_DIR=${RUNTIME_DIR:-/opt/nemo-rl}
RECIPE=${RECIPE:-$REPO_DIR/examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
BRIDGE_ROOT=${BRIDGE_ROOT:-$REPO_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
XIELU_SITE=${XIELU_SITE:-/iopsstor/scratch/cscs/xyixuan/apertus/xielu-kernel/site-v060}

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
    export PYTHONPATH="'"$BRIDGE_ROOT"'/src:'"$XIELU_SITE"'"
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
