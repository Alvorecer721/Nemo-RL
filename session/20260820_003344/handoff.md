# Handoff

## Resume From Here

The Bridge-upgrade candidate image is built and checksum-verified. The exact
artifact is listed in `session_state.md`. Its GH200 vLLM generation, four-GPU
DPO/XIELU, sync/async GRPO, two-node DPO, and ten-step two-node GRPO/refit gates
passed. Cross-allocation testing exposed and fixed stale PyTorch-DCP optimizer
detection in NeMo-RL; the fix now needs an exact-head source-only image rebuild.

## Next Actions

- Monitor async endurance `3130279` and DTensor-v2 SGLang `3130467`.
- Build an exact-head image from the pinned hermetic cache, then repeat
  `CHECKPOINT_MODE=resume` against `3130280`'s shared checkpoint on a new node.
- Record results, commit the launcher/evidence updates, push the rebased
  Bridge branch, open its NeMo-RL PR with `CI:L1`, and trigger CI for the final
  full SHA.
- Do not count restore `3130309` as optimizer-resume evidence: it completed
  weights-only and is the regression reproducer.

## Watch Outs

- `HERMETIC_CACHE_TAG=rebuild` is intentionally hermetic-only. Do not bypass that boundary or assemble release in the same allocation-local Podman graph.
- Do not rebuild XIELU unless Python, Torch, CUDA ABI, architecture, or kernel source changes; still require a real CUDA forward/backward smoke in the candidate image.
- The container fingerprint warns that embedded submodule Git directories are
  missing. The baked immutable SHAs are printed correctly and the warning is
  bypassed by the EDF; do not mistake that packaging limitation for a runtime
  dependency mismatch.
- Multi-node attempt `3130080` failed before NeMo-RL because fully static Ray
  service ports delayed raylet registration. The passing launcher retains only
  the established low worker-port range and raises
  `RAY_raylet_start_wait_time_s` to 120 seconds.
- Bridge PR #3 and Gym PR #22 are merged. Async failure hardening is already in
  this branch/image as landed commit `cc904664e`; do not reapply the old
  `d0d97df4c` SHA.
