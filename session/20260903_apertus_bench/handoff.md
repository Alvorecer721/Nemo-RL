# Handoff

## Resume From Here
Branch `autoresearch/2026-09-03-apertus70b-bench` (worktree `nemo-rl-worktrees/apertus70b-bench`, off `integration/glm51-on-upstream-a952` @ 6e2a68965) builds the NeMo RL side of the Apertus RL framework benchmark (verl vs NeMo RL, GSM8K, Apertus 1.5 70B). Starting checkpoint: frozen `/capstor/store/cscs/swissai/infra01/users/xyixuan/rl-bench/models/ap1p5-70b-sft-262k-2700_corr/` (weights = final 70B long-context SFT step 2700; RoPE factor 8 per the reported Megatron bug; release tokenizer/template; MANIFEST.json). verl reference runs (5x, jobs 3262739..3269456) used the public release (== Alignment/Apertus-1.5-70B-SFT-RL-DPO-FINAL, factor 32), 16 nodes (4 rollout SGLang TP4 + 12 trainer TP8), deliberation off, 512/512 tokens.

## Next Actions
- Read the config probe result (`.tmp/apertus_bench_config_probe.jobid`, logs `.tmp/slurm-logs/apertus-bench-probe/`); fix anything the SC preflight rejects.
- Probe ladder: `AP_VARIANT=8b-smoke infra/slurm/cscs/autoresearch/submit_apertus_bench.sh` (3 nodes, PP2 refit path) -> `70b-smoke` (3 nodes) -> `70b-bench` (16 nodes, 46 steps). Launches need the user's go.
- Eval before/after: SC has no in-run validation. Plan: `examples/run_eval.py` with a GSM8K-test jsonl via LocalMathDataset, `env_name: bracket_math`, greedy, on the start checkpoint and on a final checkpoint (enable a single final save for the bench run). Not written yet.
- Thinking-on row (DAPO/AIME, 8k-16k budget) is a later variant: chat_template_kwargs enable_thinking true, reward on the text after <|inner_suffix|>, overlong_filtering true.

## Watch Outs
- Freeze the tree while a probe/job runs (HEAD pin + dirty check in the launcher).
- Do not modify the frozen rl-bench folder; add siblings.
- `bracket_math` is registered in nemo_rl/environments/utils.py (one upstream line); the reward module is pure and unit-tested (16 tests).
- Parity gaps documented for the doc: old_log_probs source (verl bypass vs NeMo RL IS-correction), engine (SGLang vs vLLM), RoPE factor (8 here vs 32 in verl's release), memory policy (verl: full recompute + CPU offload).
