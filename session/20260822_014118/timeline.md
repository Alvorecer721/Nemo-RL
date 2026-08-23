# Timeline

## 2026-08-23 02:42 CEST

- Resumed the final NeMo-RL sync/certification lane in `.tmp/nemo-upstream-7ea-curated` at clean exact-image commit `084ade845`.
- Verified hermetic job `3153709` completed with fingerprint `ae440a7e8b2ab39353e47d5d879e9cfcbb3f6bdda5ca3133015592611772141d` and release job `3155280` is scheduler-authoritatively `RUNNING` on `nid007642`.
- Release assembly reached layer 49/56; all production worker wrapper environments through ReplayBuffer were created from the locked dependency graph. Remote Podman was at about 185% CPU; the 429 GiB node-local workspace was not near exhaustion.
- Preserved stop order: exact image first, then exact-image probe ladder and Apertus 70B production evidence, GLM scale prototype second, latest upstream delta only after evidence is stable.

## 2026-08-23 02:54 CEST

- Created committed, source-isolated Apertus 70B certification branch `autoresearch/2026-08-23-apertus70b-cert/exact-image` at `a196ee1a5`. Static config preflight passed against the immutable 145.6-GB HF checkpoint, 16-shard Megatron cache, deterministic real-DAPO slice, TP4/PP4 trainer, TP4 async vLLM, and NCCL-reshard topology; focused harness tests passed 9/9.
- Created committed GLM scale-prototype branch `autoresearch/2026-08-23-glm51-prototype/exact-image` at `315b89b53`. Its eight-node real-weight TP32/EP32 vLLM probe compiles, is Ruff/Bash clean, uses one shared Slurm step and ports below 9000, and fails closed unless the image reports baked commit `084ade845`.
- Refreshed old GLM evidence: benchmark `3147936` reached Step 10/10 but scheduler state is `FAILED 1:0` after distributed finalization errors; checkpoint Phase A `3148504` was cancelled after stalling at 272/288 shards. GLM therefore remains a scale prototype, not production-certified.

## 2026-08-22 14:00 CEST

- User approved implementing the robust Apertus refit fix after reviewing the PP2 reproduction and 70B KL evidence.
- Verified the exact code boundary: `ApertusBridge.maybe_modify_converted_hf_weight` emits `beta`/`eps` only when `task.megatron_module` is locally owned; vLLM dummy loading randomizes every persistent `state_dict` value; Python xIELU reads the buffers while CUDA xIELU reads `_beta_scalar`/`_eps_scalar` cached at construction.
- Decision: remove constants from recurring weight transfer and initialize them under the Apertus inference architecture; keep NeMo-RL core changes strictly model-agnostic (PP manifest consistency and fail-closed IPC/NCCL completion).
- Created clean isolated worktree `.tmp/apertus-refit-fix` on `fix/apertus-pp-refit-static-state` from exact tested head `58ffe9ae1`. Initialized exact submodule pins; nested Megatron-LM used the existing validated local clone because sandbox DNS cannot reach GitHub. No code edits or jobs yet.

## 2026-08-22 11:40 CEST

- Refreshed the four user completion gates from Slurm, artifacts, isolated worktrees, agents, and GitHub PR #24. The stop condition is not met.
- GLM benchmark `3147936` completed three valid 256-sample steps with nonzero signal, repeated refits, and age at most 1; Step 4 is waiting on active target-3 generation. No hard error is present.
- KL job `3147942` failed at first reference logprob from 450-GiB host-memory cgroup OOM. Retry `3148084` used the 850000M ceiling but exposed a missing per-rank DCP load setting before reference setup. A focused recipe/test fix is awaiting a fresh exact-image gate and immutable rerun.
- Checkpoint Phase A `3147961` failed after writing about 7.7 TiB of incomplete shards because an immediate Bridge barrier raced the async saver. The signed Bridge/superproject fix exists, but the repaired exact gate and Phase A have not passed; Phase B remains unlaunched.
- No corrected representative-batch Apertus-70B async learning rerun has been launched. PR #24 remains open and unchanged at `a2de6675`; runtime fixes are not yet folded or merged.

## 2026-08-22 10:14 CEST

- Final Apertus-70B split-environment gate `3147963` completed `0:0` on exact image and clean head `58ffe9ae1`; CUDA xIELU forward/backward, config isolation, four focused tests, Ruff, and formatting passed.
- Matched DAPO-prompt raw-HF control `3147953` completed `0:0`, producing 3,774 tokens at 509.13 generated tok/s. The same four prompts were mostly incorrect or length-truncated on raw HF weights, weakening the transfer-corruption hypothesis and showing that the previous all-zero reward batch was too hard/small for learning evidence.
- Refreshed live state: benchmark `3147936`, KL reference smoke `3147942`, and checkpoint Phase A retry `3147961` are running without current hard errors. Phase B remains gated on Phase A terminal checkpoint evidence.
- GitHub PR #24 remains open. Substantive CI is green; the only failed check is PR-comment posting, so publication is not yet complete even though the code checks passed.

## 2026-08-22 01:41 CEST

- User asked to optimize NeMo-RL throughput on GH200 in parallel for Apertus-1.5 70B and GLM-5.1 async GRPO with vLLM, including MFU.
- Verified base branch `codex/sync-upstream-36b5999a7-replay` at `a2de667546e760134ca50cd27fed5d556208e7be`; preserved two existing uncommitted probe edits.
- Verified Apertus 70B vLLM TP4 real-checkpoint job `3146040` completed `0:0` and generated 256 tokens at 260.26 tokens/s.
- Found the native NeMo-RL GLM-5.1 recipe and release test on the synced branch.
- Found shared GLM-5.1 BF16 and FP8 checkpoints under `infra01/hf_models/models/zai-org/`.
- Confirmed the active 600-node reservation through August 31.
- Split work into independent Apertus training, GLM async topology, and GLM vLLM preflight tracks.

## 2026-08-22 01:46 CEST

- Verified the allocation's exact CUDA device name is `NVIDIA GH200 120GB`.
- Found NeMo-RL's MFU denominator table lacked that device string. Added the GH200 dense BF16/TF32 peaks on isolated auto-research branch `autoresearch/2026-08-22-gh200-throughput/baseline`.
- Focused `tests/unit/utils/test_flops_tracker.py` run passed all 21 tests inside the exact image on job `3145210`; Ruff check and format-check passed.
- Signed measurement-baseline commit: `6bb1dfe49` (`metrics: recognize GH200 peak throughput`).
- Confirmed Megatron GLM does not need a new NeMo-local FLOPs formula: the Megatron worker uses Bridge's GLM-aware FLOP estimator and the policy aggregates that result; the missing GH200 peak mapping was the only MFU denominator gap.

## 2026-08-22 01:50 CEST

- Apertus 70B config-only preflight passed inside the exact container.
- Submitted TP4/PP4 baseline as Slurm job `3146197` on four nodes/16 GH200.
- Benchmark holds model, effective sequence count, 2048-token maximum length, BF16, sequence parallelism, and activation checkpointing fixed. TP2/PP8 will run after the baseline to avoid checkpoint-conversion, filesystem, and fabric contention.
- Logs: `.tmp/apertus70b_perf/runs/tp4pp4_3146197/` and `.tmp/apertus70b_perf/slurm_3146197.out` in the experiment worktree reported by the Apertus track.

## 2026-08-22 01:51 CEST

- Apertus attempt `3146197` failed in 42 seconds before model setup (`FAILED 15:0`; inner step `1:0`).
- Root cause: nested quoting in an embedded Python f-string produced a `SyntaxError` while printing Ray GPU count. This is a launcher-only crash and yields no throughput evidence.
- Recorded attempt 1 as `crash` in `.tmp/nemo-upstream-replay/.tmp/apertus70b_perf/experiments.tsv` and required exact-container execution of the corrected preflight before resubmission.

## 2026-08-22 01:52 CEST

- GLM-5.1 exact-image CPU preflight passed: Transformers 5.8.1 and vLLM 0.25.1 resolve `GlmMoeDsaForCausalLM`; tokenizer and chat template loaded.
- Submitted real-weight vLLM async-engine TP32/EP32 preflight as Slurm job `3146203` on eight nodes/32 GH200.
- The bounded probe uses 32 concurrent prompts, 64 generated tokens, two-second per-node GPU telemetry, and a 60-minute timeout.
- Logs: `.tmp/nemo-upstream-replay/.tmp/glm5p1_vllm_preflight/run_3146203/` and `slurm_3146203.*`.

## 2026-08-22 01:53 CEST

- Replaced the Apertus launcher's broken embedded Python diagnostic with standalone `.tmp/apertus70b_perf/check_ray_cluster.py` and aligned Ray ports with the cluster's low-port convention.
- Exact-container execution of the standalone preflight exited zero on allocation `3145210`.
- Submitted corrected Apertus TP4/PP4 baseline as job `3146221`; it formed the four-node/16-GPU Ray cluster and began NeMo-RL policy setup.
- GLM TP32/EP32 job `3146203` formed its eight-node/32-GPU Ray placement group and initialized all distributed ranks without an OOM or traceback as of this checkpoint.

## 2026-08-22 01:55 CEST

- Apertus job `3146221` failed after 91 seconds before Megatron model creation. Worker-side `uv sync` treated the isolated MFU worktree as a full project, but its Gym workspace submodule was absent. This is an experiment-overlay failure, not evidence against TP4/PP4 or the 70B checkpoint.
- Tightened the next Apertus gate: it must create the real Megatron worker environment before any four-node training allocation is resubmitted.
- GLM job `3146203` initialized the TP32/EP32 engine in 171.577 seconds and generated 2048 tokens for 32 concurrent prompts in 29.248 seconds: 70.021 output tokens/s. NVML reported about 76.3 GiB used on the sampled GH200, leaving about 21.6 GiB free.
- The GLM async topology audit found TP4/PP3/EP8 unsafe without offload because tensor parallelism does not shard routed-expert weights at ETP1. Selected TP1/PP18/EP16 on 72 training nodes plus TP32/EP32 on eight rollout nodes as the first defensible no-offload baseline candidate.
- GLM job `3146203` subsequently completed `0:0` in 4:43. Generation-window telemetry averaged 67.53% GPU utilization across all 32 GPUs and reached 100%; sampled telemetry peaked at 75,792 MiB while NeMo's immediate post-load NVML diagnostic reported 76,296 MiB.

## 2026-08-22 01:58 CEST

- Audited the probe against `configure_generation_config` after seeing degenerate repeated-token output. Because the probe did not pass `is_eval=True`, NeMo-RL correctly replaced its requested `load_format=auto` with `load_format=dummy` for a training generator awaiting Megatron refit.
- Reclassified `3146203` as a successful TP32/EP32 architecture-fit, HBM, async-engine, and throughput systems gate with dummy weights. It does not prove disk checkpoint weights or semantic model output.
- Warmup repeat `3146238` was cancelled during Pyxis startup after nine seconds and produced no model evidence. Required the next dedicated gate to assert post-configuration `load_format=auto` and semantic output, or defer real-weight proof to an actual Megatron-to-vLLM refit.

## 2026-08-22 02:04 CEST

- Submitted corrected real-weight GLM vLLM gate `3146245` on eight nodes/32 GH200. The probe asserts `load_format=auto`, the engine reports `load_format=auto`, and rank zero is reading all 282 safetensor shards. A deterministic 137+286=423 semantic assertion and final Slurm exit remain pending.
- Authored the isolated 80-node async-GRPO candidate under `.tmp/glm51-async-topology`: 72 training nodes at TP1/PP18/EP16 and eight rollout nodes at TP32/EP32, three steps, n=2, max trajectory age one, in-flight updates, importance-sampling correction, and NCCL reshard.
- Pre-submit review found and returned four harness issues for correction: export the four-GPU node size to `ray.sub`, make command resolution independent of caller cwd, isolate each job's logs, and count the exact initial plus per-step refit markers.

## 2026-08-22 02:07 CEST

- Exact-image GLM config preflight `3146293` completed `0:0` in 46 seconds on one GH200 node.
- Resolved topology: 72 training nodes/288 GPUs at TP1/PP18/EP16 with dense DP16 and expert DP1; eight rollout nodes/32 GPUs at TP32/PP1/EP32 with one rollout replica.
- `check_nccl_reshard_refit_support` passed against the full `MasterConfig`, and the checkpoint index reported 1,507,728,316,928 bytes.
- Held the 80-node submission until the isolated experiment files are committed and the concurrent real-weight evaluation load releases the checkpoint filesystem.
- Froze the GLM experiment on factual branch `autoresearch/2026-08-22-glm51-async/tp32-baseline` at signed commit `965e68e1e0949e42f989cfb57eb5aee0651387b8`; the isolated worktree is clean and the commit contains only the experiment and its session record.

## 2026-08-22 02:19 CEST

- Real-checkpoint GLM-5.1 vLLM TP32/EP32 gate `3146245` completed `0:0` in 19:46. All 282 shards loaded in 1,082.137 seconds; a deterministic arithmetic prompt returned `423`, and the measured sampled batch reached 98.741 output tokens/s with the explicit success marker.
- Submitted the committed 80-node async-GRPO smoke as job `3146357`: 72 training nodes/288 GH200 at TP1/PP18/EP16 and eight rollout nodes/32 GH200 at TP32/EP32.
- Prepared a separate meaningful 10-step hard-math throughput benchmark on branch `autoresearch/2026-08-22-glm51-async/tp32-throughput-10step` at signed commit `008725773c50d7d158785a31bfebd9aa169a6b29`. It uses a local 1,791,700-row DAPO Arrow cache, n=8, 32 prompts, GBS256, and explicit nonzero-advantage/loss/gradient gates.
- Submitted that benchmark's exact-image config/data/parser preflight as job `3146355`; the 10-step 80-node run remains gated on preflight and smoke evidence.
- Benchmark preflight `3146355` completed `0:0` in 45 seconds. It verified the local DAPO cache checksum and 1,791,700 rows, config/topology resolution, NCCL-reshard support, TensorBoard parser, and post-run gate logic.
- Async smoke `3146357` acquired all 80 requested nodes immediately and brought up the Ray head; cluster assembly and driver startup are in progress.
- Smoke attempt `3146357` reached all 320/320 Ray GPU actors, then failed `1:0` after 1:12 before NeMo-RL import. The experiment runner required `SLURM_JOB_ID`, while `ray.sub` deliberately clears MPI/PMIx/Slurm variables before starting Ray; this is a launcher-only failure with no GLM, refit, topology, memory, or throughput evidence.
- Required fix: both experiment drivers must use a stable non-Slurm run identifier, with a regression execution where `SLURM_JOB_ID` is absent, before resubmission.
- Added an experiment-owned `GLM_RUN_ID` to both smoke and benchmark launch paths and a regression that executes both drivers with `SLURM_JOB_ID` explicitly unset. The regression, Bash syntax, and diff checks passed; signed fix commit is `3d1b203470ab19a69a1d473b47e2f67d30b3f05a`.
- Resubmitted the corrected 80-node async smoke as job `3146379`; all requested nodes were allocated immediately.
- Apertus exact-image worker-environment preflight completed successfully on allocation `3145210`: one Ray node, current-root mcore command assertion, and persistent Megatron worker Python all passed.
- Submitted the matched Apertus 70B TP4/PP4 baseline as job `3146385` on four nodes/16 GH200. The lower-TP TP2/PP8 challenger remains sequentially gated behind a valid baseline.
- Corrected GLM attempt `3146379` passed all earlier gates: 320/320 Ray GPUs, non-Slurm run identity, exact config/refit preflight, and the intended 72-train/8-rollout partition. During actor setup, seven of 80 Megatron builders failed while rewriting the image-prewarmed `/opt/ray_venvs` from `/opt/nemo-rl` to the source-overlay checkout (`File exists` in stale package directories).
- Cancelled `3146379` after 5:48 because the aggregate `ray.get` could not return with seven failed required builders. This is a source-overlay/prewarm mismatch before model setup, not a dependency-resolution, GLM, memory, NCCL, refit, or topology failure.
- Required rerun change: point both experiment drivers at a clean job-local actor-venv namespace. Production exact-image runs do not have the source/prewarm fingerprint mismatch; the isolation is specific to validating an overlaid checkout without rebuilding the image.
- Added allocation-local `/opt/ray_venvs/glm51_async_e9416845542a` isolation to both GLM experiment drivers. The no-Slurm launcher regression passed again, the branch is clean at signed commit `5c53ef91f7c47420ae490ab392ab5b3bd0486a96`, and no production NeMo-RL source was changed.
- Resubmitted the clean-venv 80-node smoke as job `3146404`.
- Apertus baseline `3146385` spent 393.05 seconds importing the shared current-root worker environment; process wait channels and near-zero HBM proved this was Capstor/Lustre metadata and page-cache I/O, not checkpoint load or NCCL. All 16 workers then initialized and real Megatron actors began CUDA startup.
- The driver-level Transformers FLOPS tracker does not recognize `ApertusConfig`; this does not remove the planned external MFU path, because the Megatron worker independently reports training FLOPs and synchronized elapsed time once a step completes.
- GLM smoke `3146404` passed 320/320 Ray, config, topology and clean-namespace selection. Allocation-local actor venv construction is active with no stale `File exists` failures at this checkpoint.
- Independent topology/instrumentation audit confirmed the 72/8 baseline is memory-coherent and ranked three follow-ups: batch/refit amortization first, PP9/EP32 second, and two rollout replicas only if baseline queue/starvation evidence proves rollout-bound.
- The audit rejected BF16 TP16 rollout replicas: real TP32 used 47.53 GiB model plus 22.11 GiB KV per GPU, while halving TP would put model weights alone near the GH200 physical limit.

## 2026-08-22 02:45 CEST

- GLM smoke `3146404` validated the clean allocation-local runtime namespace: all 80 policy and 80 vLLM actor environments built without the earlier stale-directory race; Ray held all 320 GPUs, all 32 rollout ranks initialized, and all 288 Megatron ranks entered real GLM model conversion.
- The run failed `1:0` after 14:15 during the first HF-to-Megatron `torch_dist` checkpoint save, before initial policy-to-vLLM refit. The distributed sharding-integrity path called a 288-rank WORLD `all_gather_object`; many ranks then raised `_pickle.UnpicklingError: invalid load key, '\\xba'` while decoding gathered metadata.
- This is the first model-path failure in the GLM smoke, not another launcher or dependency failure. The 10-step benchmark remains held. Bridge already exposes `ckpt_load_validate_sharding_integrity=false` specifically to skip the WORLD `determine_global_metadata` gather on save; require a narrow NeMo-RL pass-through, unit coverage, and config preflight before the next 80-node attempt rather than retrying the same collective.

## 2026-08-22 02:52 CEST

- Apertus TP4/PP4 job `3146385` completed HF-to-Megatron conversion and atomically published a reusable 136 GiB torch_dist cache with 16 shards. It then loaded iteration 0 into the target topology and passed full model setup.
- Step 1/10 completed in 33.82 seconds with 27 valid preference pairs, 54 sequences, and 29,071 valid tokens. Megatron reported 1,673.54 aggregate TFLOP/s, or 104.60 TFLOP/s/GPU; against the exact GH200 BF16 peak this is about 10.57% external MFU, with about 53.7 valid tokens/s/GPU. Treat this as warm-up rather than the final steady-state score.
- Peak observed HBM during the first step ranged from about 66.7 to 79.9 GiB/GPU. The GPU telemetry shows pipeline-stage imbalance/bubble, which is the intended signal for comparing the lower-TP challenger after the baseline finishes.

- Apertus TP4/PP4 job `3146385` completed all 10 steps and exited `0:0` in 32:40, with the explicit `baked_apertus70b_tp4pp4_training=OK` marker.
- Steady steps 3-10 averaged 19.349 seconds/step, 183.35 TFLOP/s/GPU, 18.53% external GH200 BF16 MFU, and 85.70 valid tokens/s/GPU. Peak HBM was 78.72 GiB; mean per-GPU peak was 73.03 GiB. Final DPO loss was finite at 0.8130.
- The TP2/PP8 challenger may now reuse the complete conversion cache, removing the baseline's one-time 24-minute HF conversion from the matched steady-state comparison.
- Submitted matched TP2/PP8 challenger as Slurm job `3146572` on four nodes/16 GH200; it is active against the complete shared conversion cache.

## 2026-08-22 03:08 CEST

- Final exact-image gate for the narrow GLM conversion fix, job `3146592`, completed `0:0` in 2:43. The Megatron actor interpreter imported Transformer Engine; the 72-rank TP1/PP18/EP4 topology and memory preflight passed; the NeMo incomplete-cache suite passed 4/4; and the Bridge save/load propagation suite passed 2/2, including validation disabled in both the fully-parallel wrapper and distributed save/load paths.
- The GLM track is now committing the separately scoped Bridge and NeMo changes before launching the isolated 18-node/72-GPU conversion. The 80-node async retry remains gated on a complete conversion checkpoint and a fresh target-topology load.
- Apertus TP2/PP8 challenger `3146572` passed its exact source/image/worker-environment gate, initialized all 16 workers, and entered real Megatron actor CUDA/NCCL setup without an error as of this checkpoint.

## 2026-08-22 03:12 CEST

- GLM conversion attempt `3146603` reached a complete 72/72-GPU Ray cluster and passed the topology preflight, then failed `1:0` after 1:41 before any model actor or checkpoint I/O. The exact driver used `/opt/nemo_rl_venv` with only the NeMo source root on `PYTHONPATH`, so importing `nemo_rl.models.megatron.setup` failed at `from megatron.bridge import AutoBridge`.
- This is a narrow experiment-launcher omission, not a package conflict or a failure of the Bridge checkpoint fix. The already-green focused gate used the correct Bridge `src` plus NeMo source pair. Required rerun gate: import `AutoBridge` under the exact launch interpreter before requesting the conversion workflow; the fresh cache remains absent.

## 2026-08-22 03:15 CEST

- First corrected GLM driver-import gate `3146609` failed `1:0` in 31 seconds, again before model work. Adding Bridge `src` exposed its transitive import of `megatron.core`, whose matching nested Megatron-LM source was not yet on `/opt/nemo_rl_venv`'s path. The next gate must assert both MCore and Bridge imports from the same committed source tree; no conversion cache was written.
- Apertus TP2/PP8 completed its first matched warm-up step in 37.96 seconds at 93.21 TFLOP/s/GPU, 9.42% external GH200 BF16 MFU, and 47.87 valid tokens/s/GPU. The identical TP4/PP4 warm-up was 33.82 seconds and 10.57% MFU, so TP2/PP8 begins behind; the decision remains gated on steady steps 3-10 and terminal `0:0`.

## 2026-08-22 03:18 CEST

- Apertus TP2/PP8 challenger `3146572` completed all 10 steps and exited `0:0` in 21:50. Steady steps 3-10 averaged 28.143 seconds, 125.98 TFLOP/s/GPU, 12.73% external GH200 BF16 MFU, and 58.89 valid tokens/s/GPU; peak HBM reached 90.60 GiB.
- The matched TP4/PP4 baseline is decisively better: 19.349 seconds, 183.35 TFLOP/s/GPU, 18.53% MFU, 85.70 valid tokens/s/GPU, and 78.72 GiB peak. TP4/PP4 improves step throughput by about 45.5% and uses about 11.9 GiB less peak HBM.
- TP4/PP2/DP2 was rejected without launch: MCore's own BF16 model-state estimator gives about 101.7 GiB on the largest stage before activations, already above the GH200's approximately 95.6 GiB usable HBM. Changing offload or precision would no longer be a matched topology comparison.
- GLM exact actor-interpreter gate `3146621` completed `0:0` in 3:16, covering TE, MCore, Bridge and full conversion-driver imports plus the topology/memory and six focused tests. The committed 72-GPU conversion retry is job `3146627`.

## 2026-08-22 03:25 CEST

- GLM conversion attempt `3146627` formed the 72/72-GPU Ray cluster and passed the exact actor-interpreter and conversion-driver import gates, then failed `1:0` after 5:57 before policy actor creation. Passing the long shared run path to `init_ray(log_dir=...)` made Ray's local `plasma_store` Unix socket exceed the platform's 107-byte limit.
- The smoke and benchmark entrypoints do not pass this long temp root. The narrow correction is to use Ray's short allocation-local temp directory for the conversion driver while retaining durable stdout/stderr and experiment artifacts on shared storage; require a socket-length regression before another 72-GPU retry. The fresh conversion cache remains empty.

## 2026-08-22 03:28 CEST

- Cancelled initial TP8/PP2 job `3146631` after 10 seconds because its launcher case was not committed before launch; it produced no topology evidence.
- Recreated the final Apertus comparison in an isolated clean worktree on branch `autoresearch/2026-08-22-gh200-throughput/tp8pp2`, signed-off commit `021368473d982eb06638f1687c47f1d41b1b8091` directly on frozen NeMo-RL `a2de667546e760134ca50cd27fed5d556208e7be`. The seven-artifact diff passed Bash, Python, Ruff, formatting and exact-image TP8/PP2/DP1 config preflights.
- Submitted provenance-correct TP8/PP2 job `3146827` on four nodes/16 GH200. This is the last matched factorization likely to fit; it tests whether PP2's lower bubble cost can outweigh cross-node TP8 collectives.

## 2026-08-22 03:35 CEST

- Exact GLM short-Ray-path gate `3146828` completed `0:0` in 3:10. It measured a worst-case representative plasma socket at 75/107 bytes and re-passed the actor interpreter, full driver import, topology/memory, NeMo 4/4, and Bridge 2/2 gates.
- Submitted the validated fresh 18-node/72-GPU GLM conversion as job `3146889`; a duplicate check confirmed it is the only active job with that name.

## 2026-08-22 03:43 CEST

- GLM conversion attempt `3146889` failed `1:0` after 7:49, after all 72 actors reached NCCL but before HF checkpoint import/save. The conversion driver treated `Policy(...)` as a remote-constructor barrier; Ray worker-group creation returns actor handles before every actor constructor finishes, so the driver immediately asserted missing metadata and its `finally` block force-terminated the still-initializing actors.
- The cache remains empty, and there was no OOM, NCCL failure, checkpoint write, or sharding-integrity gather. The narrow harness fix is an explicit all-worker readiness barrier after `Policy(...)` (the public `policy.get_free_memory_bytes()` waits on a method queued behind every actor constructor), followed by metadata validation and minimum-free-HBM capture.
- Benchmark analysis/hardening is frozen clean at signed-off commit `b86829201aecdb80bd076de0f35dbedc84c568af`; its one-node exact gate is job `3146962`. Diff since the conversion fix is limited to autoresearch benchmark scripts and focused tests, with no conversion, production NeMo, Bridge, or submodule path changes.

## 2026-08-22 03:52 CEST

- Apertus TP8/PP2 job `3146827` completed all 10 steps and exited `0:0` in 22:58. Steady steps 3-10 averaged 24.301 seconds, 145.78 TFLOP/s/GPU, 14.73% external GH200 BF16 MFU, 68.22 valid tokens/s/GPU, and 70.17 GiB peak HBM.
- Final matched ranking is TP4/PP4 first, TP8/PP2 second, TP2/PP8 third. TP4/PP4 is about 25.6% faster than TP8/PP2 and about 45.5% faster than TP2/PP8 in normalized steady throughput; TP8/PP2 saves about 8.5 GiB peak HBM but cross-node TP collectives outweigh its lower PP bubble cost.
- GLM benchmark evidence gate `3146985` completed `0:0` in 1:44 after exact formatting, covering the DAPO data hash/count, topology/refit config, 8/8 focused metric tests, Ruff, and formatting. Final conversion barrier gate `3146988` is running from clean commit `f1a5258cef5ed058fd07e1ce2ad0c5b1dce88041` with its regression now included in the selected mcore shard.

## 2026-08-22 04:02 CEST

- Final GLM conversion barrier gate `3146988` completed `0:0` in 3:14 from clean commit `f1a5258cef5ed058fd07e1ce2ad0c5b1dce88041`. The selected exact-image shard explicitly passed the all-policy-worker readiness regression (NeMo 5/5) and Bridge save/load validation tests (2/2), in addition to topology and import preflights.
- Both launch gates are now accepted: benchmark-evidence gate `3146985` and conversion gate `3146988` are terminal-green. The next authorized action is one fresh 18-node/72-GPU conversion, followed only on a complete reloadable checkpoint by the 80-node/320-GPU three-step async smoke and then the ten-step benchmark.
- Duplicate check returned no active conversion, then the clean committed launcher submitted fresh 18-node/72-GPU conversion job `3147000` from `f1a5258cef5ed058fd07e1ce2ad0c5b1dce88041` into the empty dedicated cache. This is the only authorized active conversion.

## 2026-08-22 04:06 CEST

- Conversion `3147000` is healthy through all infrastructure gates: Ray reports 72/72 GPUs, all 18 isolated actor environments completed, all 72 actor handles initialized, and 72 distinct Megatron workers reached `after_nccl_init` on the intended shared cache. Config, actor-import, driver-import, topology, memory-headroom and short-socket gates are green. No checkpoint file, HF import marker or error has appeared yet; continue monitoring through model import, save, readiness barrier and reload evidence.

## 2026-08-22 04:11 CEST

- Conversion `3147000` built all 18 GLM pipeline stages and successfully wrote a complete-looking `torch_dist` cache: 72 rank files plus `metadata.json`, 76 files and 1,487,992,817,079 bytes total. It then failed terminal `1:0` in 13:20 during runtime-config validation, before readiness/reload acceptance.
- Root cause is an experiment/image contract mismatch, not checkpoint corruption: the experiment forced `dsa_kernel_backend=cudnn`, but the exact actor environment has neither `flash_mla` nor the `cudnn.DSA` API from `nvidia-cudnn-frontend[cutedsl]`. MCore explicitly supports `dsa_kernel_backend=none` as the PyTorch fallback. Hold the smoke; commit and gate that no-rebuild fallback, reuse the metadata-complete cache, then prove reload in a fresh conversion allocation.

## 2026-08-22 04:18 CEST

- Committed the documented no-rebuild PyTorch DSA fallback and fail-closed cache reuse at `882fc69d0a75442991a9a15fe216f5694199f247`, then fixed the exact test selector at clean HEAD `04a7d42a8bca95a8c30a34507a82d1e64e4bf0d1`. The conversion retry now refuses to create actors unless the existing metadata-complete checkpoint is detected and emits an explicit reuse marker; all three cache/barrier regressions are selected.
- Submitted three one-node exact-image gates in parallel from that clean HEAD: benchmark evidence `3147048`, conversion/cache `3147049`, and smoke config `3147050`. No large follow-on is authorized until all three are terminal `COMPLETED 0:0`.

## 2026-08-22 04:22 CEST

- All three fallback gates completed terminal `0:0`: benchmark `3147048` in 1:44 with 8/8 evidence tests and Ruff, conversion/cache `3147049` in 3:13 with all 7 selected NeMo regressions plus 2/2 Bridge tests, and smoke config `3147050` in 0:44. Resolved configs explicitly report `dsa_kernel_backend=none`; topology and NCCL refit checks remain green.
- The next gated action is one 18-node/72-GPU fresh-process load of the existing checkpoint. It must emit `existing_converted_checkpoint`, load all shards without HF conversion/save, pass the all-worker barrier, report minimum free HBM and `glm51_megatron_conversion=OK`, then exit `0:0` before the 80-node smoke.
- Duplicate check was clear; submitted that checkpoint-load proof as job `3147064` from clean HEAD `04a7d42a8bca95a8c30a34507a82d1e64e4bf0d1`.

## 2026-08-22 04:46 CEST

- Cache-load proof `3147064` emitted the required existing-checkpoint marker and initialized all 72 actor/NCCL ranks, then hung after Megatron-Core's dataset-helper Makefile resolved bare `python3` and `python3-config` from the inherited host Python 3.12 rather than the actor's Python 3.13 environment. The resulting `cpython-312` build lacked the actor venv's pybind11 headers. The job was cancelled at 04:32:40 CEST; it did not reach model-setup or reload acceptance and did not modify the complete cache.
- Added a generic launch-time Ray worker invariant at clean signed-off commit `5ca7029c1d3632b9b2c129caddd129d369e9aead`: actor subprocess PATH now places both the venv bin and its resolved base-interpreter bin before inherited host paths. A plain runtime-env PATH assignment was explicitly rejected by testing because Ray rewrites it after application; the final `/usr/bin/env PATH=... <actor-python>` launcher preserves the invariant. Exact-image Ruff passed, and both the pure ABI regression and a real nested Ray actor proved Python 3.13 `python3` plus matching `python3-config`.
- The broader worker-group module passed every executed case until its pre-existing Nsight integration case launched Ray's unpatched bare `python` and hung without `ray`; this is separate from the GLM path and is not counted as a full-module pass. Duplicate and 72-shard cache guards were clear, then fresh 18-node/72-GPU reload proof `3147363` was submitted from `5ca7029c1`.

## 2026-08-22 04:58 CEST

- Fresh-process reload proof `3147363` completed scheduler-confirmed `0:0` in 11:20. It reused the complete checkpoint on all 72 ranks with no HF conversion/save path, completed all 18 Python 3.13 dataset-helper gates, and reached 72/72 `after_model_setup` plus 72/72 `init_complete` markers. Minimum free HBM after reload was 37,905,301,504 bytes (about 35.3 GiB); both driver and shell success markers were present.
- This closes the conversion/reload and actor-Python blockers. The next gate is the committed 80-node/320-GPU three-step async GRPO smoke; the ten-step benchmark remains held until the smoke exits `0:0` with valid trajectories, nonzero learning signal, successful NCCL refits, and bounded trajectory age.
- Duplicate and clean-worktree guards passed, then the committed smoke was submitted as Slurm job `3147380` on 80 nodes/320 GH200 from `5ca7029c1d3632b9b2c129caddd129d369e9aead`.

## 2026-08-22 05:16 CEST

- Smoke `3147380` failed with exit code 1 after 12:56, before training or refit. Infrastructure, all 160 node-local actor environments, the 288-rank primary policy DCP load, and model construction succeeded. The failure occurred only when the recipe's inherited `reference_policy_kl_penalty=0.01` caused a second 288-rank reference policy load; Megatron-Core's fully-parallel loaded-object exchange returned invalid serialized payloads (`UnicodeDecodeError` / `_pickle.UnpicklingError`, unsupported operand 95) on the WORLD group.
- This is not checkpoint corruption: the same cache loaded completely for the primary policy, and the earlier fresh allocation reload proof remains valid. The benchmark remains held. The narrow throughput-recipe correction is to make the intended large-model DAPO/GRPO mode explicit: zero reference-policy KL penalty and skip reference-policy logprob calculation, matching upstream large-model Qwen, Nemotron and DAPO recipes. Configuration and preflight drift gates must pass from a clean commit before retrying the smoke.
- Committed that configuration-only correction and its two-recipe regression at clean signed-off HEAD `e9c225aeba224b9f95ff07625aab747009c5d779`. Submitted exact-image smoke-config gate `3147400` and benchmark-evidence gate `3147401` in parallel; the 80-node smoke retry remains held until both are terminal-green.
- Both exact-image gates completed terminal `0:0`: smoke config `3147400` in 0:48 and benchmark evidence `3147401` in 1:52. The resolved recipes explicitly disable the reference policy; the benchmark gate also passed its DAPO hash/count, exact 288+32 topology, 10/10 focused tests, Ruff, formatting and MCore analyzer import. The clean frozen HEAD is now authorized for one 80-node smoke retry.
- Duplicate, cache-readability and clean-worktree checks passed, then the gated 80-node/320-GPU smoke retry was submitted as job `3147405` from `e9c225aeba224b9f95ff07625aab747009c5d779`.
- Independent root-cause review also found the production-preserving KL-reference fix: pinned MCore already implements per-rank ShardedObject loads, but pinned Bridge fails to wire `ckpt_fully_parallel_load_per_rank_objects` into `FullyParallelLoadStrategyWrapper`. Do not weaken `weights_only=True` or add safe-globals. The KL-free throughput run is valid on its own; preserving KL=0.01 should be a separate narrow Bridge-wiring follow-up with regression tests.
- Implemented that production-preserving fix in isolated clean worktree `.tmp/glm-reference-object-load`: Bridge commit `504875cb73b4fb0463066a94f4a72c4a8445d784`, superproject behavior/config commit `58436eb548387f8f7238c55fb9aad46efffd3690`, and exact-container test-gate commit `1308a77c3e4bc11e05cd4d1a84236aea6fe94107`, all signed off. The Bridge diff contains no unrelated formatting. Submitted one-node focused unit gate `3147416`; it is independent of active throughput smoke `3147405`.
- Reference-fix gate `3147416` proved the Bridge wiring test 1/1 and NeMo setup tests 2/2, then failed its shell count because `--mcore-only` correctly deselected the two unmarked recipe tests. Split those into the normal NeMo pytest shard at clean signed-off commit `383cf0e1068235ad8d1d05db71ee0ce6947dfc05` and submitted corrected gate `3147419`; no production behavior changed.
- Corrected reference-fix gate `3147419` completed scheduler-confirmed `0:0` in 2:34: Bridge forwarding 1/1, NeMo checkpoint/reference setup 2/2, and explicit KL=0.01 recipe contracts 2/2 all passed in the exact image. The isolated fix is unit-gated and requires a real 288-rank reference-policy smoke before publication; it remains separate from the active KL-free throughput campaign.

## 2026-08-22 05:49 CEST

- KL-free GLM smoke `3147405` failed terminal `1:0` in 23:27 after all 288 primary-policy ranks loaded the converted checkpoint and the TP32/EP32 vLLM worker initialized. Reference-policy setup remained absent as intended, so the previous KL-mode correction is confirmed.
- The failure was node-local environment provisioning, not model, checkpoint, NCCL or dependency resolution: one `AsyncTrajectoryCollector` builder on the Ray head failed `uv run --exact --locked --extra vllm` while removing `flashinfer_jit_cache-0.6.13+cu130.data` with `Directory not empty (os error 39)`. The other per-node builders converged. No refit, trajectory, reward or training step occurred, so the ten-step benchmark remains held.
- Investigate a narrow, tested retry/convergence fix for transient uv filesystem races, gate it in the exact image, and only then repeat the 80-node smoke from a clean commit.

## 2026-08-22 06:12 CEST

- Committed bounded actor-venv convergence and early prebuild protection at clean GLM head `cc282701bfa51ba8ed5eb0df7378c9cf8e790753`. Production venv setup serializes uv wheel installs by default while preserving an operator override and retries each convergent sync/run once; the GLM launchers prebuild both late async control-actor venvs before model loading.
- Initial exact-image gate `3147445` failed only because a unit fixture used absent `/usr/bin/python3`; the production path validation behaved correctly. After the fixture switched to `sys.executable`, gate `3147449` completed `0:0` with 38 focused tests, Ruff, formatting, config/data/topology checks and MCore analyzer import.
- Two-node real-Ray integration `3147462` completed `0:0` in 8:33. ReplayBuffer and AsyncTrajectoryCollector each transitioned from `0/2` to `2/2` ready markers and a second driver invocation reused `2/2` on both nodes. There were no retry, `ENOTEMPTY`, error or traceback markers. The 80-node smoke is authorized from clean `cc282701b`.

## 2026-08-22 07:10 CEST

- GLM smoke `3147466` finished its application and allocation wrapper cleanly: `glm51_async_e2e_smoke=OK`, `Async GRPO training complete!`, and `ray.sub exiting (exit_code=0)`. All three train steps completed with finite nonzero loss and advantages, trajectory ages bounded to 0/1, initial plus per-step NCCL refits, and generation weight version 3. No traceback, Ray actor/task error, or failed marker is present.
- Authoritative external accounting confirms `3147466` as `COMPLETED 0:0` on 80 nodes in 00:43:46. The earlier controller/database errors came from the command sandbox's DNS/network isolation, not an actual Slurm outage.
- Integrated the final evidence stack at clean signed-off HEAD `46c6b263991ce947813add22d75658b3a1caf0c5`: stable distributed telemetry, per-node PP attribution, HBM/MFU analysis, exact-SHA terminal smoke artifacts, strict TensorBoard signal checks, and a fixed hash-witnessed 64-row DAPO smoke sample.
- Direct exact-image validation caught a harness-only environment defect before tests: the base NeMo interpreter imported Bridge even though Bridge belongs to the Megatron actor venv. Commit `5805ab97c` moves the rank-order assertion into the existing actor-interpreter check. The corrected exact Python 3.13.14/CUDA 13.2 gate passed config/data/topology checks, 49 focused tests, Ruff, formatting, shell syntax, and the real actor Bridge/MFU import. A scheduler preflight is ledgered as experiment 44 before launch.
- The first scheduler gate `3147550` completed `0:0` with those 49 tests. A launch-surface audit then found that three newly added test files were not selected by the persistent preflight even though their author had run them separately. Commit `14ff1f430` adds smoke-artifact, fixed-DAPO and smoke-signal validation plus their full Ruff/format surfaces.
- Superseding exact-image job `3147551` completed `0:0` in 2:01: config/data/topology checks, all 82 selected tests, Ruff, 24-file formatting, shell syntax and Megatron actor Bridge/MFU import passed. This exact clean commit now authorizes the final 80-node real-DAPO smoke, ledgered as experiment 46 before launch.

## 2026-08-22 07:35 CEST

- Final real-DAPO smoke `3147552` failed `1:0` after 19:45, safely before any model/checkpoint allocation. Both late async actor environments converged from 0/80 to 80/80 without retry or `ENOTEMPTY`, then GRPO setup rejected the deliberate absence of a validation dataset because `val_period: 1000` still enables validation even when the run has only three steps.
- This is a recipe/preflight coverage defect, not model or runtime evidence. Commit `c2dc28262` makes the no-validation contract explicit with `val_period: 0` for the smoke and inherited benchmark, while keeping both validation flags false and `data.validation: null`. Exact-image smoke and benchmark config preflights, ten focused recipe/data tests, Ruff and formatting passed directly; the full scheduler gate is ledgered as experiment 47 before launch.

## 2026-08-22 09:52 CEST

- Began the Apertus-1.5 70B end-to-end async lane at clean signed-off commit `b5b03ea36`. The bounded topology keeps the selected Megatron TP4/PP4 trainer on four GH200 nodes and assigns a fifth node to one independent vLLM TP4 rollout replica. The three-step gate uses 32 real DAPO trajectories per step, age-1 async collection, importance correction, and one NCCL refit per step.
- The launch is fail-closed on the existing 136 GiB, 16-shard Megatron conversion cache; no HF reconversion is intended. Static Python, shell, Ruff, formatting, and a synthetic terminal-metrics validator gate passed. The exact-image one-node config/data/cache/topology gate is ledgered as experiment 53 before submission; the five-node model run remains held until it exits `COMPLETED 0:0`.

## 2026-08-22 14:39 CEST

- Isolated the Apertus PP>1 refit repair in `.tmp/apertus-refit-fix`. Bridge commit `cc3ada003` removes synthetic xIELU beta/eps emission and is published on the Bridge fork. NeMo-RL commit `abdfe0375` keeps the fixed constants as non-persistent vLLM buffers, validates legacy disk-load constants, and adds generic rank-manifest agreement before IPC or NCCL-reshard transfer.
- Focused gates passed: Bridge Apertus unit 1/1, vLLM source-patch tests 4/4, generic policy/refit tests 6/6, Ruff, formatting, shell syntax and diff hygiene. A permanent `MEGATRON_PP_SIZE=2` mode now preserves the four-GH200 TP2/PP2 regression.
- The exact-image dependency stack with the candidate checkout mounted at `/opt/nemo-rl` completed the PP2 reproducer on four GH200. Refit, generation, logprob recomputation and backward/train all completed; generation KL was `0.0002` versus the prior `0.58-0.73`, and the terminal marker was `apertus_pp2_refit_fix=OK`.
- Submitted five-node source-overlay Apertus 70B async-GRPO/NCCL-reshard proof `3149468` from clean `abdfe0375`. Acceptance still requires scheduler `COMPLETED 0:0`, three refits/steps, finite nonzero learning metrics and generation KL in the expected low-error range.

## 2026-08-22 15:16 CEST

- Apertus 70B job `3149468` terminated `FAILED 1:0` after 17:04, before Megatron model load or refit. The first hard failure was an NCCL/OFI config-broadcast failure with `PTLTE_NOT_FOUND` and `VNI_NOT_FOUND`; no 70B KL or training conclusion can be drawn from it. The same cache, actor-venv path and TP4/PP4 topology had loaded and trained in the prior run, so the generic old-checkpoint footer was rejected and the 136 GiB cache was preserved.
- The failed allocation left one zero-byte `core_nid005421_288209` in the Bridge worktree. It was removed under the standing core-dump cleanup rule; source remained byte-clean. Clean five-node retry `3149806` was submitted from `abdfe0375` with the source overlay and exact image dependencies.
- GLM reference-policy job `3148506` terminated `FAILED 1:0` after 48:38. Both primary/reference checkpoint loads and the initial NCCL refit completed, then reference-logprob computation timed out in distributed collectives and cascaded into actor loss; no optimizer step or KL result completed. Root-cause analysis remains separate from the Apertus fix lane.
- GLM checkpoint Phase A job `3148504` remains allocated after step-1 training/refit. Async DCP wrote 272/288 rank files totaling 7.6 TiB, but 16 rank files are absent and the driver has emitted nothing after `Saving checkpoint for step 1...`; Phase A is not accepted and Phase B remains held.

## 2026-08-22 15:43 CEST

- Apertus 70B retry `3149806` remains scheduler-authoritatively `RUNNING` after 26:41 on five nodes. Fresh actor venv creation completed and all four vLLM worker initializers came online. The run has not yet emitted Megatron checkpoint-load, refit, generation-KL, training-step or terminal markers, so neither the former OFI boundary nor the final acceptance gate is closed yet.

## 2026-08-22 15:48 CEST

- Apertus 70B retry `3149806` exited application code 1 after 29:14. Timeline: Ray GCS was ready at +0:36; config validation plus five-node ReplayBuffer and AsyncTrajectoryCollector prebuild occupied the run until the GRPO driver connected at +17:40; the full vLLM and Megatron actor venvs then installed 231 base packages plus 88/84 actor-specific packages per node, with the slowest pair taking 5:11; vLLM loaded its four TP shards in about 2.5 seconds and established a 537,488-token KV cache; all 16 Megatron ranks reached NCCL initialization.
- At 15:45:42 the first Bridge `read_run_config` object broadcast failed with NCCL/OFI `PTLTE_NOT_FOUND` and `VNI_NOT_FOUND`, followed by the connection-buffer freelist warning. This is the same boundary as fresh allocation `3149468`; no checkpoint/model load, refit, KL, rollout, reward, or optimizer step ran. The repeated fresh-allocation result upgrades the issue from a presumed transient allocation fault to a reproducible distributed startup/environment defect. Scheduler accounting was temporarily unavailable at checkpoint time, but the allocation wrapper emitted terminal `ray.sub exiting (exit_code=1)` and the job was absent from `squeue`.

## 2026-08-22 16:03 CEST

- Deep A/B found the successful `3147845` and failed `3149468`/`3149806` use the same TP4/PP4 distributed code, master-port range and worker construction. The discriminating input is submission environment: the successful actor environment lacks the parent allocation's fabric overrides; both failures contain `FI_CXI_*`, `NCCL_NET*`, OCI hook variables and invalid host `LD_PRELOAD` inherited by `sbatch --export=ALL`.
- The leaked profile is internally inconsistent with current CSCS guidance and this checkout's `docker/nemo_rl.toml`: CUDA-12 hook into a CUDA-13 image, `NCCL_NET_PLUGIN=ofi` plus `NCCL_NET=AWS Libfabric`, GDR level `0` rather than `PHB`, and TX size `32768` rather than `16384`. The two failed Slurm stderr logs contain 574 and 944 failed attempts to preload the parent container's `/usr/libexec/coreutils/libstdbuf.so`; the successful log contains none. The repository README already states never to use `sbatch --export=ALL` from a compute container.
- Added an ignored, syntax-checked two-node/eight-rank Ray NCCL reproducer at `.tmp/apertus-refit-fix/.tmp/ofi_probe/probe_ray_nccl_broadcast.py`. It creates two strict-spread four-GPU bundles and runs the same `broadcast_object_list`, followed by all-reduce and barrier, with full NCCL environment evidence. Next launch matched dirty and explicitly clean CSCS-OFI jobs; do not pay the five-node 70B setup cost until this boundary is closed.
- Submitted matched two-node Ray/NCCL probes from the same shell and exact image: explicit clean CSCS allowlist job `3150489` and inherited `sbatch --export=ALL` job `3150490`. Both run the same eight-rank object broadcast, all-reduce and barrier; only submission environment differs.

## 2026-08-23 03:08 CEST

- Exact-image release job `3155280` remains scheduler-authoritatively `RUNNING 0:0`. It completed every frozen worker environment and the NemoGym layer, then advanced through the optional Gym-prefetch layer to Dockerfile step 55/56. Podman sustained about 180% CPU while two overlay exporters each sustained roughly 28%, confirming slow layer export rather than a hang. The expected squashfs is not published yet.
- Re-ran the strict post-hoc validator against historical GLM ten-step job `3147936`. It rejected the run because at least one step lacked a nonzero GRPO advantage range. Combined with the authoritative `FAILED 1:0` terminal state, this run remains prototype/body evidence only and cannot be cited as a clean throughput benchmark.

## 2026-08-23 03:55 CEST

- Exact source image job `3155280` completed `0:0` for clean commit `084ade845b8421ab82dcda1849d913da517f194e`; the 50,049,769,112-byte SquashFS passed the built-in vLLM 0.25.1 renderer/tokenization/tool-parser smoke and has SHA-256 `d53ec42a2c770ccd0b78536708e3959aaa984048adfee201675d49df8e8d837c`.
- Cheap exact-image gates closed independently: Apertus 70B preflight `3156315` completed `0:0`; two-step async Megatron/vLLM GRPO `3156321` completed `0:0` with generation KL `0.0003` on both steps; SGLang text/refit `3156323` completed `0:0` with HTTP weight update, cache flush, one training step and generation KL `0.0003`.
- Cross-allocation DPO checkpointing is fully accepted. Phase A `3156322` completed `0:0` and isolated optimizer-aware `step_1`; fresh-allocation Phase B `3156330` detected optimizer state, restored it, trained step 2 with loss `0.7125550508499146`, saved the next checkpoint, emitted `baked_dpo_cross_allocation_resume=OK`, and completed `0:0`.
- PP2 job `3156324` failed before model load because the Apertus runtime guard assumed Bridge was importable in the base interpreter even though the image intentionally freezes it only in the Megatron worker venv. Commit `3ea1aa1ce` adds importable-or-bundled Bridge discovery with regression coverage; seven focused guard/build tests plus Ruff and formatting pass.
- Clean image ingress exposed a second launcher-only assumption: inherited interactive Pyxis had supplied a modern `python`, while a bare Clariden host maps `python3` to Python 3.6.15. Bare-host probes found `/usr/bin/python3.11`; job `3156358` proved it generates valid fingerprint JSON. Commits `e2a2b3a77` and `deedf0c8b` make host Python explicit and pin the compatible interpreter.
- Final exact-source release assembly `3156361` is running from clean commit `deedf0c8bb2462b9e6bc5c3bbef096758053deef` against hermetic cache `ae440a7e8b2ab39353e47d5d879e9cfcbb3f6bdda5ca3133015592611772141d`. It cleared both prior startup failures and is importing cached layers for target `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-deedf0c8bb24-fa7eee499679.sqsh`.
- Fresh upstream fetch confirms `upstream/main` remains exactly `7ea279abfbeb698b092856d0dab61e7cf39bc909`; there is no additional twelve-commit delta to absorb at this checkpoint.
- Publication audit caught that the last launcher commit lacked the mandatory DCO trailer. Assembly `3156361` was cancelled after 6:23, before source copy, rather than certifying an unpublishable SHA. The commit was amended with `-s`; all 19 commits above the frozen upstream base now carry `Signed-off-by`. Final cached assembly restarted as job `3156398` from exact source `2091e790e7e42317aabf10d415a7f897f94cfafa`, targeting `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-2091e790e7e4-582be00903a2.sqsh`.
- Existing PR `#24` uses the fork's canonical replacement convention: first parent is archived/current fork `main`, second parent is the curated replay, and the committed tree is exactly the replay tree. Assembly `3156398` was therefore cancelled at step 39/56 after 21:32; canonical join `e2f1bb6aeb53fd676c1ffdcde0faec1b2143626f` was created with tree `631a93751354e0bc4c60fcbade1aa7627c66ce88`, parents `8632389d5` and `2091e790e`, and DCO sign-off.
- Final PR-range diff hygiene exposed two trailing spaces inherited from upstream commit `e40aa046e5` in `docs/design-docs/nccl-reshard-refit.md`. Signed docs-only commit `ec97553b15c40fc633b6d4efa4008d63a9f66a15` removes them. The definitive exact-HEAD assembly is job `3156544`, targeting `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-ec97553b15c4-6076a02be659.sqsh`.

## 2026-08-23 04:37 CEST

- Hosted CI on PR #24 found two publication-only source issues before the image was allowed to continue: one missing test copyright header and two false-positive hermetic MD5 digests whose allowlist comments were attached to the wrong shell lines. Signed commits `a31c01a96` and `1c4093222` fix those findings. The exact detect-secrets 1.5.0 run also required a mechanical baseline refresh for a moved historical entry; signed commit `8f22e59195f547c5715ed250cd49d4776cda5d43` makes that scanner-stable.
- PR #24 now passes copyright, secrets, uv-lock, submodule, semantic-title, and triage checks. The lone red check is the known permission failure while posting a PR comment; it does not invalidate source or dependency state.
- Cancelled assembly `3156544` before expensive work because its source SHA was not publishable. Definitive exact-head assembly `3156560` is running on `nid007634` from `8f22e5919`; source, base-image, submodule, input SHA, TRT opt-out, and hermetic fingerprint checks passed, and cached layer transfer is active.
- Updated and amended the exact-image probe harnesses to fail closed on source `8f22e591...` and image `nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.sqsh`: Apertus harness `92db67ae0`, GLM harness `76217c755`. The GLM job explicitly sets Hugging Face datasets/hub offline so it can use only the already-staged shared checkpoint.
- Fresh upstream fetch still resolves main to frozen base `7ea279abf`; there are zero newer commits, so no post-certification sync is currently available.

## 2026-08-23 07:05 CEST

- Definitive release build `3156560` completed `0:0` in 2:07:40. It published the source-exact 50,049,777,664-byte SquashFS at `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.sqsh`, SHA-256 `4aaf2b1bba8613a1e515281d84ab9e330c41d2774ccd3992b5f0c0f81e9dd002`. SquashFS validation and the built-in CUDA/vLLM/renderer/tokenizer/tool-parser smoke passed.
- Final exact-image preflight `3156865` completed `0:0` in 1:29. It validated baked source, checkpoint/cache/data identity, TP4/PP4 training plus TP4 rollout arithmetic, NCCL-reshard configuration, CUDA xIELU forward/backward, nine harness tests, Ruff, formatting and shell/config gates.
- Final PP2 refit regression `3156866` completed `0:0` in 4:53. Four GH200 ran Megatron TP2/PP2 plus colocated vLLM; refit, generation, logprobs, backward and training passed with generation KL `0.0002` and `baked_grpo_refit=OK`.
- Final five-node Apertus-1.5 70B async GRPO job `3156886` completed `0:0` in 9:26. All three steps had reward range `0..1`, nonzero advantages, finite nonzero loss/gradient and 128 valid samples. Generation KL was `0.000982`, `0.000994`, and `0.000951`; trajectory ages were `0/1/1`; E2E throughput was `104.13/122.28/128.98` tokens/s/GPU; measured train MFU was `13.58/15.10/14.96%`. Three NCCL-reshard refits succeeded and the terminal-green artifact was written.
- Final eight-node/32-GPU GLM-5.1 vLLM TP32/EP32 prototype `3156867` completed `0:0` in 14:04 on the same image. It loaded all 282 real local shards with Hugging Face offline, passed the semantic arithmetic gate, measured `92.936` output tokens/s, and peaked at about `74.00 GiB` HBM/GPU. This remains a scale prototype, not a training-production certification.
- PR #24's stale body was replaced with the exact pins, image, scheduler evidence, 70B metrics and GLM boundary. The fork's `main` was fast-forwarded from `8632389d5` to certified SHA `8f22e5919`; GitHub marked PR #24 merged with merge commit exactly `8f22e5919`, preserving image-to-main identity.
- Published clean harness branches `autoresearch/2026-08-23-apertus70b-cert/exact-image` at `92db67ae0` and `autoresearch/2026-08-23-glm51-prototype/exact-image` at `76217c755`.
- A post-certification upstream refresh found one new commit after the frozen base: `f0557321e`, a two-line sequence-importance-ratio metric fix plus regression. The certified 70B configuration has `sequence_level_importance_ratios=false`; the commit is intentionally deferred so the exact-head image remains truthful.
