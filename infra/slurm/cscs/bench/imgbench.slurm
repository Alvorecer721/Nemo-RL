#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=imgbench
#SBATCH --output=logs/imgbench/bench_%j.out
#SBATCH --error=logs/imgbench/bench_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --export=NONE
#SBATCH --no-requeue

# A/B the same sqsh served from iopsstor (flash) vs capstor (HDD Lustre):
# container start + the real cold-start workload (torch import storm,
# demand-paged from the image) + a raw sequential read sample. Each variant
# runs on a fresh node so the page cache cannot flatter the second reader.
set -euo pipefail
REPO_DIR=${REPO_DIR:-$PWD}
VARIANT=${VARIANT:?set VARIANT=iops|capstor}
EDF=$REPO_DIR/logs/imgbench/nemo_rl_$VARIANT.toml
# The variant EDFs are generated, not tracked: copy docker/nemo_rl.toml and swap the
# image line (see docs/apertus-xielu.md measurements section for the original A/B).
IMG=$(sed -n 's/^image = "\(.*\)"/\1/p' "$EDF")
echo "variant=$VARIANT node=$(hostname) img=$IMG"

# NOTE: this dd warms the page cache for the first 8 GiB before the container
# mount below, so the "cold" mount timing is partially warmed. Fine for the A/B
# (both variants share the bias); move the dd after the mount for absolute numbers.
echo "== raw sequential read (first 8 GiB, cold on this node) =="
dd if="$IMG" of=/dev/null bs=64M count=128 2>&1 | tail -1

unset LD_PRELOAD
echo "== container start + import storm =="
start=$(date +%s.%N)
srun --mpi=pmix --environment="$EDF" --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c "
    started=\$(date +%s.%N)
    awk -v a=\"\$started\" -v b=\"$start\" 'BEGIN{printf \"container up: %.1f s (mount + entry)\n\", a-b}'
    t0=\$(date +%s.%N)
    python -c 'import torch' 2>/dev/null
    t1=\$(date +%s.%N)
    awk -v a=\"\$t1\" -v b=\"\$t0\" 'BEGIN{printf \"import torch (cold pages): %.1f s\n\", a-b}'
    python -c 'import torch, transformers' 2>/dev/null
    t2=\$(date +%s.%N)
    awk -v a=\"\$t2\" -v b=\"\$t1\" 'BEGIN{printf \"import torch+transformers (warm torch pages): %.1f s\n\", a-b}'
  "
echo "END $(date)"
