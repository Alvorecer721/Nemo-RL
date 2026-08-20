# Timeline

## 2026-08-20 00:33 CEST

- Build 3126533 completed the hermetic stage but exhausted the 334 GiB node-local Podman workspace while committing release step 34.
- The completed hermetic cache was pinned by signed commit `3b116bb38`.
- Fresh-node build 3127636 resumed from cache `ccab8a76...`, crossed step 34, completed all 47 release steps, exported the SquashFS, and passed the baked vLLM import check.
- Decision: never treat pre-build pruning as sufficient for a dependency-changing build; enforce a clean storage boundary between hermetic and release phases.

## 2026-08-20 00:40 CEST

- Added the enforced two-job boundary to `build_nemo_rl_image.slurm` and documented it in the CSCS README.
- Job 3128457 passed the exact-SquashFS GH200 vLLM smoke: model load, generation, and Apertus tool parser all reported `OK`.
- Job 3128458 passed the shared XIELU CUDA forward/backward check and a four-GPU one-step Megatron DPO run; `train/loss=0.6931471824645996` passed the metric gate.
- Sync and async GRPO submissions did not receive job IDs because `clariden-slurmctl` became unreachable; `scontrol ping` reported the primary controller `DOWN`.

## 2026-08-20 01:12 CEST

- Full Ruff check and format-check passed for all 528 Python files tracked directly by NeMo-RL.
- Bridge's own mandatory tracked-file lint exposed eight import-order and seven formatting failures in the merged fork.
- Applied only Ruff's mechanical fixes and created signed local Bridge commit `535b7aa7`; full Bridge Ruff now passes across 1,706 tracked Python files and the changed files compile under Python 3.12.
- The existing candidate image remains valid functional evidence because `535b7aa7` changes formatting/import ordering only; exact-SHA provenance would require publishing the Bridge commit and producing a new image.

## 2026-08-20 10:14 CEST

- Corrected the earlier controller diagnosis: sandboxed Slurm calls could not
  create controller sockets, while escalated calls showed the controller was
  healthy.
- Candidate-image sync GRPO job `3128582` passed one real rollout, Megatron
  update, generation-KL check, and vLLM refit (`baked_grpo_refit=OK`).
- Candidate-image async GRPO job `3128583` passed two steps with CUDA graphs,
  async scheduling, two generation-KL checks, and repeated refit
  (`baked_async_grpo_tp2=OK`).
- DPO job `3128587` asynchronously saved step 1, started a fresh Python
  process, resumed to step 2, and verified `training_info.total_steps == 2`
  (`baked_dpo_async_checkpoint_resume=OK`).
- Added a two-node/eight-GPU Apertus DPO probe. Attempt `3130080` failed in Ray
  startup before NeMo-RL after an over-constrained static port layout. The
  corrected attempt `3130089` registered two fresh GH200 nodes/eight GPUs,
  initialized eight Megatron workers across both hosts, completed two TP2 DPO
  optimizer steps, passed the finite-loss gate at `0.7125550508499146`, and
  emitted `baked_dpo_multinode_training=OK` with Slurm exit `0:0`.
- Verified that async failure-path hardening commit `d0d97df4c` is on
  `origin/fix/async-grpo-reliability`, not in the Bridge branch/image.

## 2026-08-20 10:21 CEST

- Promoted the multi-node probe from a two-step startup smoke to a configurable
  bounded gate and launched job `3130139` with `MAX_STEPS=10` in a new Slurm
  allocation.
- The job again registered two nodes/eight GPUs and placed four TP2 data
  replicas across eight Megatron workers. It completed all ten DPO optimizer
  steps with finite loss (`0.601646` to `0.763633`), eight valid samples and
  sixteen global valid sequences at every step, `2.3791` seconds mean step
  time, and final loss `0.6937193870544434`.
- The step-10 metric gate passed and emitted
  `baked_dpo_multinode_training=OK`.

## 2026-08-20 10:47 CEST

- Opened and merged lint-only Bridge PR #3. Commit `535b7aa7` is now an
  ancestor of the Bridge fork's `main`.
- Corrected Gym PR #22's label to `CI:L1` and merged it as `53cf1c038`. Its
  only red workflow step was a comment-post permission failure; all substantive
  lightweight checks and prior GH200/image gates passed.
- Rebased the six unpublished Bridge/build/probe commits onto the merged Gym
  mainline without conflicts.
- Added explicit cross-allocation checkpoint modes and configurable GRPO probe
  lengths. Submitted async endurance `3130279`, checkpoint save `3130280`,
  multi-node GRPO `3130301`, and Apertus SGLang `3130299`.
- Multi-node attempt `3130283` proved Ray registered two nodes/eight GPUs but
  failed in an embedded preflight print because shell quoting stripped the
  Python `"GPU"` string. Corrected the harness and resubmitted as `3130301`.

## 2026-08-20 11:00 CEST

- Job `3130301` completed ten two-node/eight-GPU GRPO steps with ten zero-KL
  refits and emitted `baked_grpo_multinode_refit=OK`.
- Cross-allocation save `3130280` and restore `3130309` ran on different nodes.
  The restore completed step 2 but warned that the optimizer was missing. The
  checkpoint's real `.metadata` contains optimizer shards; NeMo-RL simply did
  not recognize the current PyTorch-DCP layout and disabled optimizer loading.
- Added official DCP metadata inspection and positive/weights-only regression
  tests. All 54 checkpoint tests pass, and the detector recognizes optimizer
  state in job `3130280`'s real checkpoint.
- Baked-image preflight `3130348` reproduced the old detector because the
  candidate image predates the fix, establishing the need for an exact-head
  source-only rebuild.
- Apertus SGLang job `3130299` loaded four SGLang engines, then hit the explicit
  unsupported Megatron policy refit method. Started the supported DTensor-v2
  variant as job `3130467`. Async endurance `3130279` passed step 100.

## 2026-08-20 15:00 CEST

- Invalidated jobs `3130301` and `3130279` as learning-signal evidence after
  confirming their absolute `0.02` GRPO error threshold masked all valid
  nonzero batches. Added a multiplicative `1.02` threshold and regression gate.
- Hermetic build `3130524` and fresh-allocation release build `3131131`
  produced the exact runtime-source SquashFS. It completed all 47 build steps,
  passed the baked import smoke, and has SHA-256
  `4f4602919f6b982df135abf8dd5c0ae9bc69509e83b90a857b190ce6fb725e3d`.
- Exact-image focused tests passed: 54 checkpoint tests, the threshold
  regression, and three zero-valid-batch GRPO tests.
- Corrected multi-node job `3132023` completed ten Apertus GRPO steps across
  two nodes/eight GPUs with real rewards and repeated vLLM refits.
- Cross-allocation save `3132149` on `nid005395` and restore `3132165` on
  `nid006056` proved optimizer-aware PyTorch-DCP recovery. The restored step-2
  loss exactly matched the original (`0.7125550508499146`).
- DTensor-v2 SGLang job `3132159` completed five Qwen2.5-Math-1.5B GRPO steps
  on four GH200 GPUs with real learning signal, low Generation-KL, and repeated
  successful weight-update/cache-flush responses.
- SGLang logs exposed a separate Automodel bug: lexicographic comparison of
  Torch `2.11` against `2.9` disables DTensor async checkpointing. The probe had
  checkpointing disabled, so this is a follow-up rather than Bridge-PR scope.
- Corrected async Apertus/vLLM job `3132025` completed 500 optimizer steps in
  29:41 on `nid007400` with Slurm exit `0:0`. It recorded 4,000 trajectories,
  167 positive rewards, 95,952 nonzero advantage entries, and 1,000,157 valid
  token-mask entries, then emitted `baked_async_grpo_tp2=OK` without a
  traceback or OOM.
- Pushed `codex/sync-megatron-bridge-d9212902` and opened NeMo-RL PR #23 with
  exactly the `CI:L1` label.
- All substantive PR checks passed. The post-check comment job then failed with
  HTTP 403 because its reusable workflow lacked `issues: write`; added
  least-privilege caller permissions so future PR comments succeed after this
  branch lands on `main`.
