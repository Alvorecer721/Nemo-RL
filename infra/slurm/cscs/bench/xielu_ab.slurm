#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=xielu-ab
#SBATCH --output=logs/imgbench/xielu_ab_%j.out
#SBATCH --error=logs/imgbench/xielu_ab_%j.err
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --export=NONE
#SBATCH --no-requeue

# Paired same-node A/B: fused-Python xIELU vs the CUDA custom op in vLLM
# generation. Cross-run tok/s on 3-step probes spans 102-272 for the SAME
# config (node + day variance), so only a same-node paired design can resolve
# an activation-sized effect. 15 steps per arm; both arms trace fresh (the
# shared runner purges the compile cache first, since its key is blind to
# kernel presence — trap 8); python arm first so its graphs can never be
# loaded by the kernel arm. The kernel arm injects the kernel into generation
# via VLLM_XIELU_SITE, overriding the recipe-level kernel-free default.
# Verdict from the original runs (jobs 3045293/3045454): a tie — see
# docs/apertus-xielu.md.
set -euo pipefail
echo "START $(date) node=$(hostname)"
REPO_DIR=${REPO_DIR:-$PWD}
source "$REPO_DIR/infra/slurm/cscs/bench/arm_lib.sh"

run_arm python "" MAX_STEPS=15
run_arm kernel /capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current MAX_STEPS=15
purge_vllm_compile_cache
echo "END $(date)"
