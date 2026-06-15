#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-smoke
#SBATCH --output=logs/nemo_rl_probe_%j.out
#SBATCH --error=logs/nemo_rl_probe_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl.toml}
PAYLOAD=${PAYLOAD:-$SCRIPT_DIR/probe_nemo_rl_env_payload.sh}

echo "batch_host=$(hostname)"
echo "container_env=$CONTAINER_ENV"
echo "payload=$PAYLOAD"

unset LD_PRELOAD

srun \
  --mpi=pmix \
  --environment="$CONTAINER_ENV" \
  --ntasks=1 \
  --gpus-per-task=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash "$PAYLOAD"
