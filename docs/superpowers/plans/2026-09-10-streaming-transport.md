# Streaming and Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Complete whole-group streaming alignment, smaller int16 broadcast, and reusable refit regression coverage on the upstream sync base.

**Architecture:** Three separately reviewable changes. Backend-derived selection constraints feed existing samplers before upstream training claims; byte transport changes only the tensor collective representation; refit tests exercise the current implementation.

**Tech Stack:** Python, PyTorch, Ray, TransferQueue, pytest, Slurm/CSCS.

**Spec:** `docs/superpowers/specs/2026-09-10-streaming-transport.md`.

## Global Constraints

- Work only in `.worktrees/streaming-transport-fixes`, branch `fix/2026-09-10-streaming-transport`, base `7197ac715`.
- Keep dependency pins and lockfiles unchanged. Do not modify Claude's upstream-sync tree.
- Keep rollout PP1. Trainer PP2/PP4 in refit tests is permitted. No M2N/NCCL-EP activation.
- Preserve whole prompt-occurrence groups, full optimizer-batch normalization, upstream training claims and checkpoint/resume ownership.
- Use exact source and dependency fingerprints for GPU qualification; never use `NRL_IGNORE_VERSION_MISMATCH`.
- Run meaningful regression tests before and after runtime changes. Commit separately with sign-off. No main merge.

### Task 1: Streaming group alignment

**Files:** `nemo_rl/algorithms/async_utils/staleness_sampler.py`, `nemo_rl/algorithms/single_controller.py`, `nemo_rl/models/policy/tq_policy.py`, and relevant focused tests in `tests/unit/single_controller/` and `tests/unit/models/policy/`. Add a small helper module only if existing abstractions require it; add new typed modules to the pyrefly allow-list.

**Interfaces:** Extend sampler selection with keyword-only `group_count_multiple: int = 1`; backend supplies the required whole-group multiple from actual sharding/batching configuration. Determine enabled logprob constraints from their actual dispatch contracts, not a guessed GPU count. Preserve existing sampler/claim return contracts.

- [x] Inspect current sampler tests, policy train/logprob dispatch, controller update lifecycle and checkpoint tests. The old WIP at `dapo-main-stream24` is reference only; it uses obsolete immediate buffer removal.
- [x] Add regression cases for 25 ready groups at DP6/G16: select 24, retain the remaining group's rows/metrics, then complete 48 groups across aligned chunks exactly once. Assert training claims survive an open-step checkpoint and are released after success.
- [x] Add backend-constraint tests covering packed/dynamic and fixed batching, distinct active logprob constraints, invalid group counts, impossible min/max/target, and a shortfall tail. Establish which microbatch paths genuinely require divisibility from their implementations.
- [x] Run the new regressions and capture the expected failure on the unmodified base.
- [x] Derive the minimal group quantum as `sample_multiple // gcd(sample_multiple, generations_per_prompt)`, with `sample_multiple` the LCM of applicable dispatch requirements. Round selection down before concat/claim; never discard or split groups. Preserve existing error behavior for unsupported custom contracts, and make invalid target/bounds fail clearly before consumption.
- [x] Wire the controller to the backend constraint and account explicitly for dropped-group target changes. Keep unchanged default behavior for quantum 1. Avoid repeated per-step remote geometry lookups and side-channel attributes.
- [x] Run focused sampler, split-dispatch, train-pump and claim/checkpoint regression coverage. Confirm one optimizer update and normalization are preserved across chunks; self-review and commit only task-owned files.

### Task 2: Byte-view router broadcast

**Files:** `nemo_rl/data_plane/worker_mixin.py`, `tests/unit/data_plane/test_leader_broadcast.py`; optional focused distributed functional test if GPU coverage cannot use the existing harness.

**Interfaces:** Keep `_broadcast_batched_data_dict`'s signature and descriptor unchanged. The `tensor` branch broadcasts uint8 views only for int16 logical tensors.

- [x] Add a two-rank Gloo regression observing uint8 wire dtype and exactly `2 * logical.numel()` bytes while asserting exact reconstructed int16 values, shape, and untouched source. Cover negative sentinels/extremes, strided/transposed inputs, scalar and empty tensors.
- [x] Confirm the new wire-size regression fails against int32 transport while existing correctness tests pass.
- [x] Allocate the receiver at logical dtype; normalize source placement/contiguity. Use `tensor.reshape(-1).view(torch.uint8)` for int16, broadcasting into the receiver's logical allocation. Preserve source-device restoration, error envelopes and PackedTensor handling; no extra descriptor fields or generic packing protocol.
- [x] Run the full focused broadcast suite including source-error propagation and non-int16/PackedTensor cases. Add reusable NCCL parity/timing coverage with no speedup assertion; report GPU tests as pending until actually run.
- [x] Self-review, lint and commit only task-owned files.

### Task 3: Current refit regression probes

**Files:** Recover/adapt `tests/functional/refit_nccl_ordering_repro.py` from `.tmp/worktrees/glm51-stream8-common`; site launchers under `infra/slurm/cscs/`; focused tests alongside existing refit/watchdog/generation-pause suites.

**Interfaces:** Use current `StatelessProcessGroup`, `xferdtensor`, watchdog and vLLM refit APIs verified against source. Reuse current EDF/source-fingerprint launch contracts. No new dependency or alternative production synchronizer.

- [x] Inspect the historical three-rank/multinode reproduction and current upstream refit/pause/watchdog test analogs. Preserve the smallest useful coverage; remove stale environment/monkeypatch machinery when current tooling covers it.
- [x] Add focused tests for probe configuration/result validation and relevant current refit behavior. A smoke checks repeated transfer equality and bounded failure/teardown, not merely process exit.
- [x] Adapt the probe to current API signatures, explicit transport selection and node/rank placement. Test 1 versus 2 existing refit streams with ordinary NCCL, trainer PP2/PP4 analogs, rollout PP1. Do not invent a synchronization fix absent a reproduced failure.
- [x] Validate CPU-testable parts, shell syntax, Ruff and focused existing pause/watchdog tests. Prepare a bounded GPU launch against a matching image and record exact source/image/runtime.
- [x] Self-review and commit task-owned files. Record any GPU qualification dependency without claiming it passed.

### Task 4: Combined qualification and review

- [x] Review each task against its specification and fix substantive findings.
- [x] Run the union of relevant tests once on the combined source; run lint, format and diff checks. Re-run only after changes/failures justify it.
- [ ] Qualify Gloo broadcast and the bounded NCCL/refit tests when infrastructure is available. Match dependency fingerprint before larger 70B streaming/GLM parity jobs; keep unrun gates explicit.
- [ ] Obtain independent whole-branch review. Record commits, source/image identity, tests and limitations. Update existing issues #32/#39 with actual evidence; keep them open until their qualification criteria are met.

## Validation recorded September 10

Implementation commits: `d987cc6c6` (streaming), `b56199d73` (strict FIFO
preflight), `b36dcb1f7` (byte transport), `2c8939cfc` (refit probes).
Each task passed independent review. No production refit synchronization changed.

- Combined source at `2c8939cfc`: 468 tests passed, one skip, 74.94 seconds.
  The skipped Megatron split-state module needs `megatron.bridge`, absent from
  the scratch venv; this is not numerical model-normalization qualification.
- The 18 warnings are dependency deprecations: PyTorch/pynvml, the inherited
  HF transfer environment variable, and Ray Serve/Pydantic. Ray child imports
  also log upstream Transformers Qwen2VL output-docstring messages.
- Changed-file Ruff lint/format, per-task Pyrefly and whitespace checks passed.
  Lockfiles, actor environments and submodule pins match base `7197ac715`.
- Two-GPU NCCL broadcast job `3351386` completed: three tests passed on GH200.
  Scratch runtime: Python 3.13.14, Torch 2.11.0+cu130, Ray 2.56.1, NCCL 2.28.9.
  Exact values and byte payloads passed for CPU and CUDA sources. One 2 MiB
  transfer measured 9.743 ms; no matched baseline or throughput gain is established.
- Standalone refit job `3351750` was submitted from immutable `2c8939cfc`:
  two nodes, trainer stages 2/4, one/two streams, 40 iterations, eight transfers
  per stage, 2 MiB payloads. Results are pending.
- Native release-image assembly, native preflight, the 70B streaming/resume
  gate, GLM Router Replay/logprob parity and representative performance remain open.

## Decisions and limits

Strict WeightFifo rejects a group quantum above one before consumption.
An oldest-version remainder can otherwise stall; mixing newer versions would
change FIFO semantics. Quantum-one FIFO is unchanged, and ready-first/windowed/
in-order retain aligned leftovers. The cost is an explicit unsupported-layout
error for strict FIFO until version cohorts have a safe completion contract.

The scratch environment is read-only and tests standalone behavior. Native
model checks require matching dependency fingerprints and baked workers;
there is no mismatch bypass. Gloo and async tests require execution outside
the sandbox, whose network/executor restrictions were reproduced independently.
Failed launch allocations were released before retrying with the existing
explicit Slurm export pattern; unrelated jobs and hooks were left alone.

The feature worktree is isolated from the active sync and historical WIP trees.
Keep the branch and qualification evidence until native gates complete; keep
issues #32/#39 open and do not retire historical WIP copies yet.
