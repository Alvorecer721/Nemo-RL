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

# Stall-step probe: 30 GRPO steps with host-side GPU telemetry at 2 Hz so slow
# generation bursts can be correlated with clock/power/thermal state.
#
# VERDICT (job 3045982, nid006290, 2026-08-10): on a healthy node the stalls
# do not exist — 29/29 post-warmup bursts at full SM clocks (1980 MHz), the
# one mildly long burst was extra work at full speed (longer decode tail),
# and warmup settled in one step. The 2x stalls seen in the paired A/Bs were
# node-associated (every affected arm ran on nid007126; both arms equally, so
# the A/B conclusion is unaffected). Systemic recoverable time: ~zero. Ops
# heuristic: a node showing 2x bursts at normal token counts is a node to
# report — rerun this probe there to confirm with telemetry.
set -euo pipefail
echo "START $(date) node=$(hostname)"
REPO_DIR=${REPO_DIR:-$PWD}
source "$REPO_DIR/infra/slurm/cscs/bench/arm_lib.sh"
TEL=$REPO_DIR/logs/imgbench/telemetry_${SLURM_JOB_ID}.csv

nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory \
  --format=csv,noheader -lms 500 > "$TEL" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

run_arm stall "" MAX_STEPS=30

kill $SMI_PID 2>/dev/null || true
echo "telemetry: $TEL ($(wc -l < "$TEL") samples)"
echo "END $(date)"
