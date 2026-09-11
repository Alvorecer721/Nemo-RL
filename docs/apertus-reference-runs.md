# Apertus and GLM reference runs

Resolved configurations of the runs we cite externally, exported with
`tools/export_shared_config.py`. Each file is exactly the configuration one job ran with,
after replacing site paths with `<site>/<basename>` and listing the keys whose names
never occur in the upstream source at the pinned ref (name-based: a fork key that
reuses an upstream name is not listed). They document runs; they are not loadable by
upstream.

| Run | Snapshot | Source commit | Job |
|---|---|---|---|
| Apertus 1.5 70B GSM8K, 16 nodes (12 trainer TP2/PP4, 4 rollout), 48x16, 2k context | `reference-configs/2026-09-10-apertus-70b-gsm8k-job3352055.yaml` | `7197ac71505b` | 3352055 |
| GLM-5.1 GRPO ready-first, 136 nodes (72 trainer TP2/PP18/EP16, 64 rollout), 10 steps | `reference-configs/2026-08-29-glm51-ready-first-job3217663.yaml` | `b6ea7c17daf2` | 3217663 |
| Apertus 1.5 70B DAPO thinking-12k, 40 nodes (24 trainer TP2/PP4, 16 rollout), 48x16, GBS 768, 92 updates | `reference-configs/2026-09-06-apertus-70b-dapo-thinking12k-gbs768-job3308421.yaml` (recipe expansion; run and checkpoint directories are placeholders) | `e82118fb7659` | 3308421 |

Regenerate a snapshot from a checkpoint:

```bash
python tools/export_shared_config.py --config <checkpoints>/step_N/config.yaml \
  --upstream-ref <upstream commit from the sync ledger> --source-commit <sha> --job-id <id> \
  --reference-run "<one line>" --resolved-from "checkpoint step_N/config.yaml of job <id>" \
  --output docs/reference-configs/<date>-<run>-job<id>.yaml
```

Or from a recipe, with the launcher's `AP_*` / `GLM_*` variables exported, using `--recipe`.
The recipes themselves live under `examples/configs/recipes/llm/` and inherit through
`defaults:` chains; the snapshots are the flattened form.
