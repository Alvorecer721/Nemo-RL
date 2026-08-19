# Handoff

## Resume From Here

The Bridge-upgrade candidate image is built and checksum-verified. The exact artifact is listed in `session_state.md`. Its GH200 vLLM generation smoke and four-GPU DPO/XIELU probe passed; sync and async GRPO refit probes remain before production certification.

## Next Actions

- Run the sync-GRPO refit and async-GRPO probes using `.tmp/nemo_rl_bridge_3b116bb.toml`.
- Build an exact-head source-only candidate after the builder-safety commit is finalized, then repeat the release gate as appropriate.

## Watch Outs

- `HERMETIC_CACHE_TAG=rebuild` is intentionally hermetic-only. Do not bypass that boundary or assemble release in the same allocation-local Podman graph.
- Do not rebuild XIELU unless Python, Torch, CUDA ABI, architecture, or kernel source changes; still require a real CUDA forward/backward smoke in the candidate image.
- The first GRPO submissions returned no job IDs while the Slurm controller was down. Confirm controller health before resubmitting to avoid assuming a job exists.
