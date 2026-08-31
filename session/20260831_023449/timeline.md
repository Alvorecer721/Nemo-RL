# Timeline

- 2026-08-31 02:34 CEST: user confirmed that the already certified dependency SQSH should be reused for Python/YAML changes. Rebuild removed from the challenger plan.
- 2026-08-31 02:36 CEST: created a clean shared-storage copy at `/capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl-worktrees/glm51-mtp3-fused`; implementation remains in the equivalent clean `/tmp` worktree until committed.
- 2026-08-31 02:38 CEST: checkpointed baseline, failed MTP3 evidence, accepted change bundle, and preservation boundaries before editing.
- 2026-08-31 02:47 CEST: compute-image probe `3238845` proved the shared overlay import and dependency fingerprint, then exposed a test-fixture mismatch: the fake sampler emitted one prompt group while the helper expected two. Production code had not completed the test step. Corrected the test's expected prompt count to one.
- 2026-08-31 02:51 CEST: corrected probe `3238849` completed. The new SQSH imported commit `e1af50b5a` from shared storage without a fingerprint bypass; 15/15 focused tests passed and the exact MTP3 recipe resolved with fused logprobs enabled, chunk 256, and parameter-gather overlap disabled.
