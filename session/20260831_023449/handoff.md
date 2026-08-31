# Handoff

Continue on `autoresearch/2026-08-31-glm51-mtp3-fused-counters` at `e1af50b5a` (base `2e75f5612`). Do not edit or commit the older dirty checkout at `/capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl/v0.7.0`.

The source/config-only challenger should reuse the certified `ad878283417c-3f68ccb671f5` SQSH. Before launch, detach the branch from `/tmp/nemo-rl-glm51-mtp3-fused`, attach it to the clean shared copy at `/capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl-worktrees/glm51-mtp3-fused`, and export that path first on `PYTHONPATH`. Keep the dependency fingerprint enabled: it covers `pyproject.toml`, `uv.lock`, and submodule SHAs, so this source-only overlay should match without `NRL_IGNORE_VERSION_MISMATCH`. Keep exact source HEAD, clean tracked tree, recipe, and topology checks fail closed.

The speculative metrics are a wall-clock delta over generation activity concurrent with one trainer step. They are not exact counters for only the rollout groups consumed by that step; document and test that boundary.

Pre-launch gates are green in compute-image job `3238849`: dependency fingerprint and overlay import passed, 15/15 focused tests passed, and recipe resolution certified the fused profile. The run is accepted only after Slurm terminal success, 10/10 training steps, Router Replay trace, nonzero speculative draft/acceptance counters, finite generation KL, and no OOM.
