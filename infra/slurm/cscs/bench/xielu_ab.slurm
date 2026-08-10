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
# config (node + day variance), so only a same-node paired design can
# resolve an activation-sized effect. 15 steps per arm, compile cache purged
# before each so both arms trace fresh; python arm first so its graphs can
# never be loaded by the kernel arm (the cache key is blind to kernel
# presence).
set -euo pipefail
echo "START $(date) node=$(hostname)"
REPO_DIR=${REPO_DIR:-$PWD}
EMPTY_SITE=$REPO_DIR/logs/imgbench/empty-site
mkdir -p "$EMPTY_SITE"

run_arm () {
  local arm=$1 site=$2
  local log=$REPO_DIR/logs/imgbench/arm_${arm}_${SLURM_JOB_ID}.log
  rm -rf $HOME/.cache/vllm*/torch_compile_cache
  echo "=== ARM $arm (XIELU_SITE=$site) $(date) ==="
  ( export REPO_DIR XIELU_SITE=$site MAX_STEPS=15
    source "$REPO_DIR/infra/slurm/cscs/probe_grpo_fixgate.slurm" ) > "$log" 2>&1
  echo "--- ARM $arm results (log: $log)"
  grep -oE 'Generation KL Error: [0-9.]+' "$log" | tail -3
  grep -oE '[0-9]+\.?[0-9]* tokens/s' "$log" | tr '\n' ' '; echo
  echo "kernel-detected: $(grep -ic 'using experimental xielu cuda' "$log" || true)"
}

run_arm python "$EMPTY_SITE"
run_arm kernel /capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/xielu-site-current
rm -rf $HOME/.cache/vllm*/torch_compile_cache
echo "END $(date)"
