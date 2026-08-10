#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=stall-probe
#SBATCH --output=logs/imgbench/stall_probe_%j.out
#SBATCH --error=logs/imgbench/stall_probe_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --export=NONE
#SBATCH --no-requeue

# Stall-step diagnosis: both paired A/Bs showed sporadic generation bursts at
# ~2x per-token cost with normal token counts, normal wake/sleep edges, no
# preemptions and no recompiles (~4-5% of end-to-end step time). Candidate
# causes: GPU clock/power transients from the colocated train->generate duty
# cycle vs workload-linked scheduling effects (steps 2,3,7 stalled in BOTH
# arms; same seed => same prompts). This run collects host-side GPU telemetry
# at 2 Hz alongside a 30-step probe so slow bursts can be correlated with
# clock/power/thermal state on the wall clock.
set -euo pipefail
echo "START $(date) node=$(hostname)"
REPO_DIR=${REPO_DIR:-$PWD}
TEL=$REPO_DIR/logs/imgbench/telemetry_${SLURM_JOB_ID}.csv

nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory \
  --format=csv,noheader -lms 500 > "$TEL" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

rm -rf /users/xyixuan/.cache/vllm*/torch_compile_cache
( export REPO_DIR MAX_STEPS=30
  source "$REPO_DIR/infra/slurm/cscs/probe_grpo_fixgate.slurm" ) \
  > "$REPO_DIR/logs/imgbench/stall_arm_${SLURM_JOB_ID}.log" 2>&1

kill $SMI_PID 2>/dev/null || true
echo "telemetry: $TEL ($(wc -l < "$TEL") samples)"
echo "--- per-step generation seconds:"
grep -oE 'generation: [0-9.]+s' "$REPO_DIR/logs/imgbench/stall_arm_${SLURM_JOB_ID}.log" | sed 's/generation: //' | tr '\n' ' '; echo
echo "--- kernel-detected: $(grep -ic 'using experimental xielu cuda' "$REPO_DIR/logs/imgbench/stall_arm_${SLURM_JOB_ID}.log" || true)"
echo "END $(date)"
