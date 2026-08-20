# Handoff

## Resume From Here

The Bridge-upgrade runtime is built and its bounded release gates are green.
The exact runtime-source artifact is:

`/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-0e73bdb8367e-f8a605afd716.sqsh`

SHA-256:
`4f4602919f6b982df135abf8dd5c0ae9bc69509e83b90a857b190ce6fb725e3d`.

The image was assembled in a fresh allocation from hermetic cache fingerprint
`508e7c3083af7ce63ece6885650b95dd70e66cc10f576b3ed8b41de6b7727d26`.
Commits after the image source only harden probe launchers and record evidence;
they do not change the installed runtime, dependency graph, or image contents.

## Closed Production Gates

- Exact-image focused checkpoint, GRPO, rollout, and validation tests passed.
- Two-node/eight-GPU Apertus GRPO completed ten optimizer steps with real
  learning signal and repeated Megatron-to-vLLM refits (job `3132023`).
- Cross-allocation DPO save and optimizer-aware restore passed on different
  nodes: save `3132149` on `nid005395`, restore `3132165` on `nid006056`.
  The restored step-2 loss exactly matched the original step-2 loss
  (`0.7125550508499146`).
- DTensor-v2 text-only SGLang GRPO completed five steps on four GH200 GPUs with
  real rewards/advantages and repeated successful weight-update/cache-flush
  HTTP responses (job `3132159`). This is Qwen2.5-Math-1.5B evidence, not
  Apertus or multimodal evidence.
- Corrected async Apertus/vLLM endurance job `3132025` completed 500 optimizer
  steps in 29:41 with Slurm exit `0:0`: 4,000 trajectories, 167 positive
  rewards, 95,952 nonzero advantage entries, and 1,000,157 valid token-mask
  entries. It emitted `baked_async_grpo_tp2=OK` without a traceback or OOM.

## Publication

- NeMo-RL PR: `https://github.com/Alvorecer721/Nemo-RL/pull/23`
- Branch: `codex/sync-megatron-bridge-d9212902`
- Label: exactly `CI:L1`

CI was triggered from the final branch SHA recorded in PR #23's
`/ok to test <full-final-sha>` comment.

## Next Actions

- Monitor PR #23 CI/review.
- Fix Automodel's Torch-version comparison as a separate SGLang checkpointing
  follow-up; do not mix it into the Bridge-upgrade PR.

## Watch Outs

- Do not count jobs `3130279` or `3130301` as learning-signal evidence. Their
  `0.02` absolute GRPO error threshold masked valid nonzero batches. The
  corrected probes use the multiplicative `1.02` threshold.
- Do not count restore `3130309` as optimizer-resume evidence; it reproduced
  the stale optimizer detector and restored weights only. Jobs `3132149` and
  `3132165` are the corrected exact-image evidence.
- Megatron-backed Apertus SGLang remains unsupported. The validated SGLang
  path is DTensor-v2 with Qwen2.5-Math-1.5B.
- Automodel currently compares `torch.__version__` to `"2.9.0"` as strings,
  so Torch 2.11 incorrectly disables DTensor async checkpointing. The SGLang
  probe intentionally had global checkpointing disabled; fix this separately
  before calling the entire SGLang backend production-stable.
- `HERMETIC_CACHE_TAG=rebuild` is hermetic-only. Release assembly must use a
  fresh allocation-local Podman graph; carrying the dependency graph into
  release assembly previously exhausted the 334 GiB node-local workspace.
- Do not rebuild XIELU unless Python, Torch, CUDA ABI, architecture, or kernel
  source changes. The candidate image still requires and passed a real CUDA
  forward/backward smoke.
- Bridge PR #3 and Gym PR #22 are merged. Async failure hardening is already in
  this branch as landed commit `cc904664e`; do not reapply `d0d97df4c`.
- Test-only cross-allocation checkpoint trees remain under `.tmp/` because a
  safety gate declined irreversible deletion without a separate explicit user
  approval. They are ignored validation artifacts, not source changes.
