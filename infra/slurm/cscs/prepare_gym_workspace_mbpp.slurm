#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=prepare-gym-mbpp
#SBATCH --output=logs/prepare_gym_mbpp_%j.out
#SBATCH --error=logs/prepare_gym_mbpp_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=preemptable
#SBATCH --export=NONE
#SBATCH --no-requeue

set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/infra/slurm/cscs/environments/nemo_rl_vllm029.toml}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/outputs/apertus-mbpp}
REVISION=${REVISION:-4bb6404fdc6cacfda99d4ac4205087b89d32030c}
UV=${UV:-/root/.local/bin/uv}

[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "ERROR: output directory already exists: $OUTPUT_DIR" >&2
  exit 1
}

export SLURM_EXPORT_ENV=ALL
srun --export=ALL --mpi=pmix --environment="$CONTAINER_ENV" --chdir="$REPO_DIR" --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    set -euo pipefail
    export HF_HOME=\${HF_HOME:-/iopsstor/scratch/cscs/\${USER}/.cache/huggingface}
    cd $REPO_DIR
    $UV run --locked --extra mcore python \
      examples/nemo_gym/apertus/prepare_mbpp_workspace.py \
      --revision $REVISION --output-dir $OUTPUT_DIR --max-steps 8
  "
