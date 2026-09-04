# Session State

- Goal: turn the GLM-5.1 scale prototype into production evidence for optimizer checkpoint completion, fresh-allocation resume, reference KL, and endurance.
- Branch: `autoresearch/glm51-r3-10step-20260824`
- Published merge: NeMo-RL PR #26, `c85af58d9aa815504e006e736df6dc16042ee76c`
- Certified image: `/capstor/store/cscs/swissai/infra01/MLLM/containers/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.aarch64.sqsh`
- Reservation: `SD-69241-apertus-1-5-0`, expires 2026-08-31 12:00 CEST. Do not cancel it.
- Historical incomplete checkpoint and the 1.488-TB conversion cache are preservation boundaries.

## Current diagnosis

Two independent memory faults are now isolated and runtime-proven. Save-side CPU pressure came from strict per-worker NUMA binding; preferred-local placement with fallback completed all 288 shards in `3164148`. Restore-side GPU pressure came from allocating a full optimizer load skeleton and then letting `torch.optim.Optimizer.load_state_dict` cast it to CUDA before Transformer Engine allocated its final scaled state. TP2 did not hide that duplication: unpatched fresh-allocation job `3168898` still CUDA-OOMed on final-stage ranks 272-287 at about 92.6-93.0 GiB allocated with only 12-277 MiB free. The clean MCore candidate `23ae88370` initializes TE's final optimizer representation as the DCP target and restores group metadata without the casting loader. Job `3169314` restored all 288 ranks in a fresh 80-node allocation, including ranks 272-287, at about 63.0 GiB allocated with about 26.5 GiB free; it then completed refit and training step 2 with loss `0.0295`, reward range `0-1`, generation KL `0.0025`, and exit code 0. This closes checkpoint save and cross-allocation optimizer recovery at GLM-5.1 scale. Multi-hour endurance remains a separate production gate.

## Plan

- [x] Rebase the investigation onto current certified main.
- [x] Expose Bridge checkpoint save/load controls and a Ray object-store cap.
- [x] Pass current-source sync and NVRx save plus fresh-allocation next-step controls; both red Slurm exits were post-proof harness assertions, not checkpoint failures.
- [x] Port the GLM harness without a custom dataset adapter. Source-overlay diagnostics use an empty head-scoped actor venv because mutating baked packages in place fails on Enroot's writable overlay; production remains image-owned.
- [x] Runtime-attribute the checkpoint loss to strict per-worker NUMA memory binding and replace it with preferred-local placement that allows fallback.
- [x] Complete a 288-rank GLM optimizer save and fresh-allocation resume. TP2 Phase A `3168499` wrote a complete 8.946-TB DCP checkpoint; patched Phase B `3169314` restored it and trained the next step.
- [x] Runtime-prove nonzero reference KL (`0.0025` after restore/refit in `3169314`).
- [x] Publish the runtime-proven checkpoint and integration changes through NeMo-RL PR #26.
- [ ] Run representative endurance as a separate production gate.
- [x] Characterize GLM train-vs-rollout mismatch over ten R3-enabled steps, including tail probability error and route integrity.
- [x] Close the false-red legacy-async versus TransferQueue trace contract and reject unsupported mixed entrypoint configurations before setup.
- [ ] Repeat ten R3-enabled steps with a 2048-token sequence / 1536-token response envelope to obtain representative learning-signal evidence.

## Current subtask (2026-08-24 09:28 CEST)

- Run a fresh ten-step TP2/PP18/EP16 async-GRPO experiment with TP32/EP32 vLLM and `policy.router_replay.enabled=true`.
- Compare all ten KL and token-probability-error values with the preserved R3-off ten-step body from job `3147936`, whose KL mean was about `0.00250` but whose 3,858,221 valid training tokens included 7,873 `abs(delta log p) > 0.5`, 570 above `1.0`, and a maximum of `37.7`.
- Require strict route validation, a verified route trace, no missing-route fallback, at least eight learning-signal steps, and a clean terminal artifact. Do not write a new checkpoint or mutate the valid TP2 checkpoint.

## SingleController characterization (2026-08-24 18:28 CEST)

- The ten-step legacy-async control completed training but failed its final learning-quality gate because 94.77% of responses were truncated; its R3 transport evidence itself is complete and clean.
- The next experiment changes orchestration to the current SingleController + TransferQueue path and extends the response envelope to 3072 total / 2560 generated tokens. It preserves the exact 80-node TP2/PP18/EP16 Megatron and TP32/EP32 vLLM topology.
- Start with a one-step runtime characterization. Require in-image Pydantic validation, TransferQueue producer/fetch integrity, verified R3 forward replay and CP identity, generation KL below 0.001, token multiplier below 1.02, positive valid-token count, and a clean terminal artifact before launching the ten-step gate.
- Host-side Pydantic validation is unavailable because the login Python does not include Ray; this is expected and the same validation runs inside the certified image before Ray setup.

## SingleController runtime result (2026-08-24 23:10 CEST)

- One-step job `3175340` passed in-image config validation, built all 160 role environments, completed the initial Megatron-to-vLLM NCCL reshard refit in 5.817 seconds, generated all 16 prompt groups, and delivered one batch through TransferQueue.
- The run failed before prev-logprob computation and training step 1. `TQWorkerMixin._broadcast_batched_data_dict` passed the Router Replay `routed_experts` tensor directly to NCCL as `torch.int16`; NCCL rejected `Short` with `TypeError: Input tensor data type is not supported for NCCL process group: Short`.
- The R3 trace proves the failing payload field is `routed_experts`, shaped `[8, 2171, 78, 8]` and stored as `torch.int16`. This is an upstream-combination defect: compact R3 route storage and TransferQueue leader-broadcast are both inherited from upstream, but their combined NCCL path has no dtype-compatibility coverage.
- Current fix contract: preserve the compact logical dtype and exact bits outside the collective, carry unsupported `int16` tensors as raw NCCL-supported bytes only during broadcast, restore the original device/dtype for every recipient, add CPU plus real NCCL regression coverage, then repeat the one-step gate before any ten-step launch.
- The candidate implements that contract and adds Gloo plus two-GPU NCCL round trips. Exact-image job `3178396` showed Torch 2.11 Gloo rejects `Short` too, so the candidate uses the exact-byte wire for int16 on every process-group backend. Corrected job `3178422` passed Ruff, formatting, Gloo and real two-GPU NCCL coverage: 3 tests passed. The 80-node one-step gate is next.

## PR #27 exact-image gate (2026-08-24 15:32 CEST)

- Exact-image job `3173736` ran the broad changed-path selection from `5143b429d`: 242 tests passed before `test_sc_checkpointing.py::TestSetupResumeWiring::test_setup_forwards_latest_resume_paths` failed.
- The failure is a fixture regression from the new fail-fast runtime contract. `_setup_master_config` constructs a SingleController config but omitted `async_grpo=None`, so it inherited the legacy async block that production now correctly rejects. Keep the production guard and make the fixture explicit, then rerun the complete gate before submitting the 80-node experiment.
- Rerun `3173779` from `dfc9e50bb` passed all 420 selected tests and Ruff check, then failed only because Ruff format identified one mechanical reflow in `test_resiliency_config.py`. Exact-image formatter job `3173814` applied that one-line reflow; the final full gate remains required from the next committed head.

## R3 characterization result (2026-08-24 11:16 CEST)

- Job `3171492` completed all ten legacy-async GRPO training steps. Router Replay reduced generation KL from the historical R3-off mean `0.00250` to `0.000388`; all ten steps were below `0.000407`. Across 1,291,712 valid tokens only four had `abs(delta log p) > 0.5`, none exceeded `1.0`, and the maximum was `0.676`.
- The Slurm job was false-red after successful training: `tools/check_r3_trace.py` unconditionally required SingleController/TransferQueue producer and fetch events, while this recipe deliberately runs `examples.run_grpo` with `data_plane.enabled=false` and the legacy in-memory ReplayBuffer. Its 269,952 route records included assignments, actions, forward-verifier matches and CP identity, but TransferQueue records cannot exist on this path.
- Learning evidence remains insufficient: only five of ten steps had nonzero reward/advantage/loss because 94.5-100% of responses were truncated at the 1024-token generation cap. The next rung extends the response envelope rather than weakening the eight-of-ten learning-signal gate.
- The entrypoint audit found a related fail-open seam: `examples.run_grpo` can construct a TransferQueue policy for `async_grpo.enabled=true` plus `data_plane.enabled=true`, then call the legacy async trainer whose in-memory ReplayBuffer does not consume that transport. The unsupported mixed configuration must fail before Ray and actor setup.
- The harness now selects an explicit `legacy-async` or `transfer-queue` trace contract, isolates every Slurm job's artifacts, and centralizes suite completion checks so a zero exit without `train/loss` at the configured final step cannot be reported as complete. Entrypoints reject incompatible transport and unsupported SingleController knobs before Ray setup; configured GRPO advantage clipping is now applied instead of accepted as a no-op.
- Focused current-source validation is green: Python compilation, shell syntax, `git diff --check`, and 13 pure trace/completion tests passed. The dependency-bearing changed-path suite still needs the exact image because this host Python does not provide Ray.

## Published checkpoint status

- MCore PR #1 is merged at `9c82d4cad8c2aba345903cefee56989ba46f7013`; Bridge PR #5 is merged at `6b24b9e7944300a2a908c4e710841a679d435b95`; NeMo-RL PR #26 is merged at `c85af58d9aa815504e006e736df6dc16042ee76c` and pins that Bridge commit.
- Keep the valid TP2 checkpoint for endurance and future scale validation even though publication is complete.
- Preserve reservation `SD-69241-apertus-1-5-0` and the reusable 1.488-TB model-conversion cache.
- Main is level with `origin/main`; only the main NeMo worktree remains. The source overlay remains validation machinery, not the production install contract.

## Loaded skills

- `nemo-rl-auto-research` - one-variable experiment, committed hypothesis, terminal artifacts, and TSV ledger.
- `nemo-rl-session-memory` - checkpoint state before edits and long-running launches.
- `config-conventions` and `testing` - recipe inheritance, naming, and focused harness coverage.

## TP2 experiment plan

- [x] Add `TP2/PP18/ETP1/EP16` save/resume recipes on the existing 80-node layout. Dense DP is 8; expert DP remains 1.
- [x] Generalize only the harness topology preflight and pass focused tests/lint (9/9 tests; Ruff clean; `bash -n` clean).
- [x] Commit the hypothesis before launch.
- [x] Submit Phase A using the existing reservation and write a complete TP2 checkpoint (`3168499`).
- [x] Prove unpatched TP2 still fails during optimizer restoration (`3168898`).
- [x] Prove the final MCore patch restores in a fresh allocation and completes the next training step (`3169314`).
- [x] Add a launcher preflight so uninitialized or mismatched recursive submodules fail before node allocation (`558dd58b5` after DCO replay; focused tests 10/10, Ruff and `bash -n` clean).
- [x] Merge the clean MCore fix through fork PR #1, merge its Bridge gitlink through fork PR #5, and pin the merged Bridge SHA in NeMo-RL.
- [x] Relock with image-owned uv 0.11.28 in job `3169750`; 549 packages resolved in 1.04 seconds and `uv.lock` remained byte-identical.
- [x] Publish NeMo-RL PR #26, trigger CI at the full commit SHA, fix the copyright failure, and merge after every substantive check passed. The lone red check was the fork-only PR-comment publisher; its underlying submodule check passed.
