# Handoff

## Resume From Here
Branch `autoresearch/2026-09-03-apertus70b-bench` (worktree `nemo-rl-worktrees/apertus70b-bench`, off `integration/glm51-on-upstream-a952` @ 6e2a68965) builds the NeMo RL side of the Apertus RL framework benchmark (verl vs NeMo RL, GSM8K, Apertus 1.5 70B). Starting checkpoint: frozen `/capstor/store/cscs/swissai/infra01/users/xyixuan/rl-bench/models/ap1p5-70b-sft-262k-2700_corr/` (weights = final 70B long-context SFT step 2700; RoPE factor 8 per the reported Megatron bug; release tokenizer/template; MANIFEST.json). verl reference runs (5x, jobs 3262739..3269456) used the public release (== Alignment/Apertus-1.5-70B-SFT-RL-DPO-FINAL, factor 32), 16 nodes (4 rollout SGLang TP4 + 12 trainer TP8), deliberation off, 512/512 tokens.

## Next Actions
- Read the config probe result (`.tmp/apertus_bench_config_probe.jobid`, logs `.tmp/slurm-logs/apertus-bench-probe/`); fix anything the SC preflight rejects.
- Probe ladder: `AP_VARIANT=8b-smoke infra/slurm/cscs/autoresearch/submit_apertus_bench.sh` (3 nodes, PP2 refit path) -> `70b-smoke` (3 nodes) -> `70b-bench` (16 nodes, 92 steps of 48 x 16). Launches need the user's go.
- Eval before/after: SC has no in-run validation. Plan: `examples/run_eval.py` with a GSM8K-test jsonl via LocalMathDataset, `env_name: bracket_math`, greedy, on the start checkpoint and on a final checkpoint (enable a single final save for the bench run). Not written yet.
- Thinking-on row (DAPO/AIME, 8k-16k budget) is a later variant: chat_template_kwargs enable_thinking true, reward on the text after <|inner_suffix|>, overlong_filtering true.

## Watch Outs
- Freeze the tree while a probe/job runs (HEAD pin + dirty check in the launcher).
- Do not modify the frozen rl-bench folder; add siblings.
- `bracket_math` is registered in nemo_rl/environments/utils.py (one upstream line); the reward module is pure and unit-tested (16 tests).
- Parity gaps documented for the doc: old_log_probs source (verl bypass vs NeMo RL IS-correction), engine (SGLang vs vLLM), RoPE factor (8 here vs 32 in verl's release), memory policy (verl: full recompute + CPU offload).
- 2026-09-03 16:04: config probe 3280261 failed on two setup errors (actor env not in ACTOR_ENVIRONMENT_REGISTRY; recipe defaults path one level short); unit tests 18/18, prompt render and offline GSM8K were green. Fixed in 55148ea798139ab7a52f9ba88344df352a4f2650. Sampler switched to windowed (max_staleness_versions 8) at the user's request.
- 2026-09-03 16:30: probe 3280309 failed on stall_timeout_s == generation_timeout_s (fixed, 4200 s) and an exact-float assertion in the probe itself. Probe 3281005 then passed everything except the SC one-step-one-update rule (96 x 16 = 1536 != gbs 768). Decision: 48 x 16 per step, 92 steps, gbs 768, staleness 16 versions (same samples and optimizer updates as verl; twice the refits; two NeMo RL steps = one verl step in per-step plots). The 96 x 16 / gbs 1536 / 46-step alternative keeps verl's sync cadence but halves the updates and was rejected for that reason. Eval config + one-node eval launcher committed (53758d285); the probe now preflights the eval config too.
- 2026-09-03 17:28: probe 3281031 passed end to end (18 unit tests, env round trip, 3 SC preflights, eval preflight, prompt render, offline GSM8K) at 895b610e0. User gave the go for the infra comparison. Launched: baseline greedy eval of the frozen 70B checkpoint (job 3281177, 1 node, tag start). Enabled a weights-only final checkpoint in the bench recipe (save_period 100000 so only the last step saves; save_data_plane true is mandatory with the windowed sampler) so the trained model can be exported with tools/export_megatron_to_hf.py and evaluated with the same protocol.
- 2026-09-03 17:41: 8B smoke 3281230 failed at cluster build: `num_nodes (2) must be divisible by segment_size (4)`; cluster.segment_size 4 was copied from GLM. On Alps every node is its own NVLink domain, so the segment only constrains divisibility; reset to the upstream default null for bench and smokes. Launcher fixed earlier to resolve cluster.num_nodes through the defaults chain and to expect 92 bench steps.

