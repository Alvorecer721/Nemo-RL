# Shared arm-runner for the bench harnesses: the purge -> run -> attest sequence
# is the experiment-validity recipe (traps 8 and 10), so it lives in exactly one
# place. Source this from a bench sbatch script.

# The compile-cache purge must hit the cache vLLM will actually use: workers
# honor an inherited VLLM_CACHE_ROOT, so purging only $HOME would silently miss
# a redirected cache and void the fresh-compile guarantee.
purge_vllm_compile_cache() {
  rm -rf "${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"*/torch_compile_cache \
         "$HOME"/.cache/vllm*/torch_compile_cache 2>/dev/null || true
}

# run_arm <arm-name> <vllm-xielu-site-or-empty> [extra exported VAR=VAL ...]
# Runs one fixgate-based arm with a purged compile cache, logs to
# logs/imgbench/arm_<name>_<jobid>.log, and prints the validity attestations:
# per-step generation phase timers (never vLLM's windowed tok/s snapshots —
# trap 9) and the kernel-presence count (trap 10: logs attest actual config).
run_arm() {
  local arm=$1 vllm_site=$2; shift 2
  local log=$REPO_DIR/logs/imgbench/arm_${arm}_${SLURM_JOB_ID}.log
  purge_vllm_compile_cache
  echo "=== ARM $arm (VLLM_XIELU_SITE=${vllm_site:-<kernel-free>}) $(date) ==="
  ( export REPO_DIR VLLM_XIELU_SITE=$vllm_site
    for kv in "$@"; do export "${kv?}"; done
    source "$REPO_DIR/infra/slurm/cscs/probe_grpo_fixgate.slurm" ) > "$log" 2>&1
  echo "--- ARM $arm results (log: $log)"
  grep -oE 'Generation KL Error: [0-9.]+' "$log" | tail -3
  echo "gen seconds/step: $(grep -oE 'generation: [0-9.]+s' "$log" | sed 's/generation: //' | tr '\n' ' ')"
  echo "kernel-detected: $(grep -ic 'using experimental xielu cuda' "$log" || true)"
}
