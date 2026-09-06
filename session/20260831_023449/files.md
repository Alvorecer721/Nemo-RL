# Files

- `nemo_rl/algorithms/single_controller.py`: planned vLLM speculative-counter step window.
- `tests/unit/single_controller/test_single_controller_actor.py`: planned focused counter emission/no-op tests.
- `infra/slurm/cscs/autoresearch/recipes/grpo-glm5.1-136n4g-megatron-tp2pp18ep16-ready-first-mtp3.yaml`: planned fused-logprob challenger overrides.
- `infra/slurm/cscs/autoresearch/validate_glm51_sc_scale.py`: planned fail-closed fused-setting certification.
- `tests/unit/infra/test_glm51_sc_scale.py`: planned certification coverage.
- `infra/slurm/cscs/autoresearch/submit_glm51_sc_mtp3.sh`: planned expected fused-profile export.
- `infra/slurm/cscs/autoresearch/run_glm51_sc_scale.sh`: planned controlled source-overlay environment propagation.

## Generated (2026-09-02 bubble analysis, slide assets in this directory)

- `fleet_bubble.png` — fleet-scaling grouped bars (mean warm step vs trainer wait, 2/4/8 engines + MTP3 pair).
- `fleet_bubble_table.png` — the matching table (job ids, steps, mean step, wait share, peak bubble).
- `step_time_breakdown.png` — stacked per-phase step decomposition, baseline vs MTP3+fused.
- `stall_gauge.png` — rollout/idle_s sawtooth small-multiples, 4 runs (fused-only excluded).
- Interactive page with all data + captions: https://claude.ai/code/artifact/f86315e8-0466-43d3-8c7f-f485a4840546
