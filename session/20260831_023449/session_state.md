# Session State

- Goal: rerun the matched 136-node GLM-5.1 DeepSeekMTP-3 challenger after removing the final-pipeline-stage logprob OOM and making speculative-decoding counters a certified output.
- Branch: `autoresearch/2026-08-31-glm51-mtp3-fused-counters`
- Base: `2e75f5612f8ec5d301af32d0215c7efbb0eb8e70` (curated upstream main `ccbcd4cc5442784f6af2288dd99021560480b8f2`).
- Dependency image: `/capstor/store/cscs/swissai/infra01/MLLM/containers/nemo-rl-apertus-vllm-0.25.1-ad878283417c-3f68ccb671f5.aarch64.sqsh`.
- Execution model: reuse the certified image and import this exact clean checkout as a shared `PYTHONPATH` overlay. Do not rebuild for Python/YAML-only changes. Keep the dependency fingerprint guard enabled; it excludes ordinary source files and should match naturally.
- Reservation: `SD-69241-apertus-1-5-0`, expected to expire 2026-08-31 12:00 CEST. Do not cancel it.
- Preservation boundaries: retain the 8.946-TB GLM checkpoint and 1.488-TB conversion cache.

## Evidence entering this session

- Baseline `3217663`: completed 10/10 matched steps on 72 training plus 64 generation nodes; Router Replay trace passed and generation KL stayed in the accepted range.
- First MTP3 attempt `3219237`: rejected before weights because vLLM DeepSeekMTP does not support `use_local_argmax_reduction`.
- Corrected MTP3 `3219292`: completed 8/10 steps, proved native MTP weights and Router Replay, then CUDA-OOMed in step 9 backward on the heavy final PP stage. The run emitted no `train/vllm/spec_*` metrics because SingleController did not consume vLLM's existing step-counter API.

## Approved one-bundle challenger

- Enable fused linear logprobs with chunk size 256.
- Disable DDP parameter-gather overlap, which is required by the fused implementation.
- Wire vLLM speculative counters into SingleController using the same wall-clock training-step window used by the existing async GRPO loop.
- Preserve all matched topology, sampler, data, checkpoint, Router Replay, and speculative-decoding settings.
- First certify locally and in the existing image with a shared source overlay; then submit one 136-node ten-step run and monitor it to terminal artifacts.

## Pre-launch certification

- Static checks: Ruff check and format, `git diff --check`, Python compilation, and shell syntax are green.
- Compute-image probe `3238849`: `COMPLETED`, source overlay and dependency fingerprint green, 15/15 focused tests passed, and the resolved MTP3 profile certified fused logprobs, chunk 256, and parameter-gather overlap disabled.
- Ray worker propagation: `RayWorkerBuilder` copies the driver's environment and explicitly retains `PYTHONPATH` in both isolated initializers and worker runtime environments.

## Loaded skills

- `nemo-rl-auto-research`: committed hypothesis, one feature bundle, terminal validation, and experiment ledger.
- `nemo-rl-session-memory`: durable state before edits and launches.
- `contributing`, `config-conventions`, `testing`, `build-and-dependency`, `error-handling`, and `linting-and-formatting`: focused tests, fail-closed preflight, signed commits, and image/overlay decision.
