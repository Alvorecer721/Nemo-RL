# Files

- `examples/configs/recipes/llm/autoresearch/grpo-glm5.1-136n4g-megatron-tp2pp18ep16-ready-first.yaml` — three-step 72+64 topology.
- `infra/slurm/cscs/autoresearch/run_glm51_sc_scale.sh` — fail-closed SingleController runner and terminal artifact writer.
- `infra/slurm/cscs/autoresearch/submit_glm51_sc_scale.sh` — reservation-aware single-srun Ray submission.
- `infra/slurm/cscs/autoresearch/validate_glm51_sc_scale.py` — correctness and throughput summary for the short scale gate.
- `tests/unit/infra/test_glm51_sc_scale.py` — focused validator gates.
- PR #27 file diff — staged locally so the experiment runs the exact Router Replay/runtime-contract code under review.
