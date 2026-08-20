# Handoff

## Resume From Here

The Bridge-upgrade candidate image is built and checksum-verified. The exact
artifact is listed in `session_state.md`. Its GH200 vLLM generation, four-GPU
DPO/XIELU, sync GRPO, async GRPO, async checkpoint/resume, and two-node/eight-GPU
Apertus Megatron DPO probes all passed. The final multi-node gate ran ten
optimizer steps, not only a startup step.

## Next Actions

- Publish local Bridge lint-only commit `535b7aa7` before publishing the corresponding NeMo-RL submodule pointer.
- Integrate `origin/fix/async-grpo-reliability` before treating async engine
  crashes, finite-loader EOF, and distributed finalization as production-safe.
- Later gates: cross-allocation checkpoint restore, text-only SGLang, and a
  many-hour endurance run.
- Build an exact-head image only after the PR head is final if release
  provenance must match the merge commit; the existing candidate contains the
  semantic Bridge upgrade and differs only by later formatting/build/docs/probe
  changes.

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
