# Timeline

## 2026-08-23 14:01 CEST

- Created a clean current-main GLM production worktree.
- Re-read historical job `3148504`: 272/288 shards landed in under a minute, 16 ranks never completed, no metadata was written, and completed ranks waited in the post-save barrier until manual cancellation.
- Verified NVRx v0.6 persistent finalization blocks without checking writer-process liveness. The exact child failure is not preserved.
- Added explicit `fully_parallel_save` and load-integrity passthrough, plus an optional per-node Ray object-store cap.
- Staged matched one-node synchronous and NVRx asynchronous save / fresh-allocation resume controls using the certified TOML and image-owned `/opt/ray_venvs`.

## 2026-08-23 14:10 CEST

- Jobs `3161932` and `3161933` failed in the harness preflight before Ray or model startup: the driver Python was incorrectly given raw Bridge/MCore source paths and therefore lacked worker-only Transformer Engine dependencies.
- Cancelled dependent resume jobs `3161934` and `3161935`. The correction keeps the driver on the NeMo-RL source overlay and runs Megatron-specific preflight through the baked Megatron worker Python.

## 2026-08-23 15:48 CEST

- Current-source controls closed the small-topology question: jobs `3162831` and `3162855` each wrote two complete 4/4-rank optimizer checkpoints, about 103 GB per checkpoint, with synchronous and NVRx persistence respectively.
- Fresh-allocation jobs `3162832` and `3162856` both loaded `step_1` and completed the next training step. Their Slurm wrappers exited red only because the test harness attempted a permission-preserving cross-filesystem `mv` and later expected Megatron iteration 1 although the outer NeMo-RL step cursor was 2.
- Recovered the historical GLM contract and replaced its custom dataset adapter, raw squashfs launch, split Slurm steps and nested actor venv with the built-in `DAPOMath17K` loader, `docker/nemo_rl_vllm0251.toml`, single-step Ray and image-owned `/opt/ray_venvs`.
- Added the real 80-node Phase-A/Phase-B harness. It preserves the proven 288-rank TP1/PP18/EP16 training plus TP32/EP32 rollout topology, caps Ray's object store at 64 GiB per node, requests 850000M host memory, disables fully-parallel save, and fails after 20 minutes without shard-count progress while retaining diagnostics.
- Focused config tests passed 3/3 with the repository-wide Ray autouse fixture intentionally bypassed; the ordinary unit harness failed before test collection because Ray could not resolve the current container hostname.

## 2026-08-23 16:37 CEST

- Real-topology retry `3163157` failed after 4:08 before model initialization. The source overlay changed each actor's uv command root, so the worker builder correctly rejected the baked fingerprint and attempted an in-place update of the image-owned venv. On several nodes uv collided with lower-layer package directories (`flash_attn/layers`, `transformers/quantizers`, and `deep_ep/backend`: `File exists`). This is a launch-layer overlay failure, not a recurrence of the checkpoint failure.
- Historical job `3148504` is now correlated more tightly. The missing global ranks were `9,39,57,73,83,117,162,211,215,219,237,273,277,282,284,285`, spread across fourteen nodes; therefore one failed Slurm node cannot explain the loss. Five of sixteen final-PP-stage writers failed versus eleven of 272 other ranks.
- Slurm accounting reports the historical head step `OUT_OF_MEMORY`. The worker step's maximum RSS was 261.33 GiB on `nid006625` (`172.28.32.212`), the node whose rank 273 writer is missing. That node had already written roughly 162 GB of its four final-stage rank files; its missing fourth file was about 52 GB. Combined with the old Ray object-store allocation of about 135 GiB per node and other process overhead, the 450-GiB node cgroup had insufficient checkpoint-staging headroom.
- The exact hang mechanism is source-confirmed: NVRx persistent save performs D2H staging inside its child process while the training parent blocks on `preload_q.join()`. If that child is OOM-killed, it never calls `task_done()` and the parent waits forever. The historical driver never progressed beyond `policy.save_checkpoint`, matching the 16 absent files and lack of a Python traceback.
- The rerun now uses an empty source-head-scoped actor venv, retaining the warm image uv cache without mutating baked packages. It captures per-node `memory.events`, `memory.peak`, cgroup identity and process state once checkpointing begins and again on a bounded stall, so the next run can directly attribute any lost writer instead of inferring it from aggregate accounting.

## 2026-08-23 17:48 CEST

- Instrumented 80-node retry `3163625` initialized all 288 Megatron ranks, completed rollout, training and refit, then failed at the first optimizer checkpoint. It wrote 230 rank shards before the distributed save aborted.
- Slurm identified the initiating event as `nid006944: task 20: Out Of Memory`. That node hosted ranks 200-203; ranks 200, 201 and 203 each wrote a roughly 29.52-GB shard, while rank 202 wrote none.
- Rank 202 was hard-bound by `nemo_rl.distributed.numa_utils` to CPU NUMA node 2. GH200 exposes only about 120 GB of CPU DRAM in each such node. The outer job cgroup allowed 891.29 GB and peaked at only 334.78 GB on `nid006944`, proving that aggregate job memory was not exhausted.
- MCore's async checkpoint preload copies every tensor to CPU before persistence. The strict one-node `numa_set_membind` therefore converts a rank-local staging spike into an OOM despite free memory elsewhere. The historical sixteen missing writers across fourteen nodes are consistent with this code-side capacity bug, not bad hardware.
- Replaced the strict memory bind with a preferred-local policy that permits fallback while retaining GPU-local CPU affinity. Focused NUMA tests pass 21/21 (two GPU benchmarks skipped in the sandbox); Ruff and compile checks pass. The exact 80-node confirmation and fresh-allocation resume remain pending.

## 2026-08-23 18:36 CEST

- Exact-head 80-node confirmation `3164148` completed rollout, step-1 training, NCCL refit and the first full optimizer checkpoint with the preferred-local NUMA policy. All 288 rank shards landed, totaling 8,926,103,907,981 bytes, and both DCP `.metadata` and Megatron `metadata.json` were written. No Slurm OOM occurred.
- This closes the initiating checkpoint fault: the preceding strict-bind run lost rank 202 and stopped at 230 shards, while the otherwise-matched fallback-enabled run completed 288/288.
- Training quality remained finite and nontrivial in the checkpoint step: loss `0.0190`, average reward `0.0312`, and generation KL error `0.0028`.
- The Slurm wrapper exited 1 only after printing `glm51_cross_allocation_save=OK`: terminal-artifact creation read `SLURM_JOB_ID`, but `ray.sub` intentionally clears `SLURM_*` before launching Ray. Preserve the allocation ID as `NRL_SLURM_JOB_ID` and use that in the artifact writer. Focused harness tests pass 5/5; fresh-allocation restore remains pending.

## 2026-08-23 21:29 CEST

- Fresh-allocation Phase B job `3165089` read the complete checkpoint and entered Transformer Engine FusedAdam state initialization, then CUDA-OOMed only on global ranks 272-287. Those ranks are exactly the 16 DP replicas of PP stage 17, the heavy final pipeline stage. The failure is before the next training step and is fail-loud; no numerical corruption was observed.
- The existing optimizer checkpoint requires unchanged TP1/PP18. Merely adding unused nodes cannot help; the next useful compatible scale is DP32: 576 trainer GPUs on 144 nodes plus the unchanged eight-node rollout pool, 152 nodes total. This halves DP-sharded optimizer state per rank.
- Added an explicit scale/recipe/Phase-A-artifact contract to the launcher and a `152n4g` resume recipe. The existing reservation remains the 80-node default, while an explicitly empty reservation permits this larger ordinary-partition probe. Focused recipe and harness tests pass 7/7.
- Storage audit found three unusable partial checkpoints: 230 shards/6.4 TiB, 272 shards/7.6 TiB, and 273 shards/7.7 TiB, each without DCP metadata. The complete 288-shard checkpoint is 8.2 TiB and must remain until DP32 restoration is proven. The user authorized deleting the old failed sharded checkpoints; do not delete the complete checkpoint or 1.488-TB conversion cache.
- Deleted exactly the three metadata-less 230/272/273-shard namespaces and verified they no longer exist, reclaiming about 21.7 TiB. Re-verified the retained `526a5c6e...` checkpoint has DCP metadata and 288 shards; the conversion cache and reservation were untouched.

## 2026-08-23 23:45 CEST

- User requested a TP2 restore test without the CPU-placeholder fix and authorized removing the old complete Megatron optimizer checkpoint plus regenerating/resharding as needed.
- The existing TP1 optimizer checkpoint is not used as TP2 evidence. The controlled plan creates a fresh `TP2/PP18/ETP1/EP16` Phase-A checkpoint and restores it in Phase B with the same topology.
- On 72 trainer nodes (288 ranks), dense DP changes from 16 to 8 while expert DP remains 1. The final dense pipeline stage is tensor-sharded in half, directly testing whether extra model headroom is enough to survive the still-unfixed duplicate optimizer allocation.
- Reservation `SD-69241-apertus-1-5-0` remains a preservation boundary. The 1.488-TB model-conversion cache remains reusable and must not be removed.

## 2026-08-24 02:47 CEST

- Unpatched TP2 Phase A job `3168499` completed one learning step and wrote a complete 288-shard optimizer checkpoint: 8,946,006,731,964 bytes plus 35,338,056-byte DCP metadata. Loss was `0.0240`, reward range `0-1`, and generation KL `0.0027`.
- The same-topology fresh-allocation control `3168898` disproved TP2 as a sufficient workaround. Ranks 272-287, exactly TP2 times dense-DP8 replicas of the heavy last pipeline stage, CUDA-OOMed while Transformer Engine initialized restored optimizer state. PyTorch held roughly 92.6-93.0 GiB and only 12-277 MiB remained free.
- Job `3169053` showed that relocating only the empty DCP placeholder to CPU is not enough: `torch.optim.Optimizer.load_state_dict` automatically casts it back to the parameter device before TE creates final scaled state. The candidate therefore bypasses that casting loader only when precision-aware TE state has an identity-safe float32 representation.
- Direct-state job `3169260` exposed a dormant integration mismatch before tensor load: pinned MCore called `FusedAdam.initialize_state(p)`, while TE 2.15 requires `initialize_state(p, store_param_remainders)`. The final candidate carries the exact two-argument correction already present on upstream MCore development.
- Final job `3169314`, using clean MCore SHA `23ae88370`, restored all 288 optimizer ranks in a fresh 80-node allocation. Heavy ranks 272-287 completed at about 63.0 GiB PyTorch allocation and 26.5 GiB NVML free, roughly 30 GiB less allocation than the unpatched control. Replay state restored, vLLM refit completed, and outer step 2 trained with loss `0.0295`, reward range `0-1`, generation KL `0.0025`; the driver and Slurm job exited 0 with `terminal_green=true`.
- Added a submission preflight after job `3168998` revealed an uninitialized fresh-worktree submodule. The launcher now rejects recursive gitlink `-` or `+` states before requesting nodes. Focused tests pass 10/10; Ruff, formatting, `git diff --check`, and `bash -n` pass.
- Merged the clean MCore change through [fork PR #1](https://github.com/Alvorecer721/Megatron-LM/pull/1) at `9c82d4ca`, then merged the single-gitlink Bridge handoff through [fork PR #5](https://github.com/Alvorecer721/Megatron-Bridge/pull/5) at `6b24b9e7`. The only red automation on those personal-fork PRs was inherited NVIDIA/Claude infrastructure requiring NVIDIA SSO or an Anthropic credential; link and secret checks passed.
- Image-owned relock job `3169750` verified Bridge `6b24b9e7` and nested MCore `9c82d4ca`, ran uv 0.11.28 from the image in 1.04 seconds, and left `uv.lock` unchanged. Two preceding harness attempts exited before uv: one inherited compute-node Pyxis state and one asserted the future `/usr/local/bin/uv` location, while this certified image still bundles its unshadowed uv at `/root/.local/bin/uv`.
- Replayed only the unpublished post-main NeMo commits to add mandatory DCO trailers. The tested pre-replay tree `a65d5c5f34926790d774261eb0d43a5f5ae3cdcf` is bit-identical to the published-candidate tree; only commit identities changed.

## 2026-08-24 03:36 CEST

- Opened [NeMo-RL PR #26](https://github.com/Alvorecer721/Nemo-RL/pull/26) from the exact runtime-proven tree. The first CI pass found one substantive issue: the new GLM harness test lacked the standard NVIDIA copyright header. Added it in signed commit `12c16cb23`, reran the 10 harness tests plus Ruff/format checks, and retriggered CI at the full SHA.
- Copyright, lock freshness, secrets, semantic-title and recursive submodule checks passed. The only red job was the personal fork's PR-comment publisher; the producer submodule check and its artifact both passed. The NVIDIA heavy test queue is repository-gated and cannot launch from a personal fork; changed-path suites and the exact 80-node red/green runtime remain the applicable evidence.
- Merged PR #26 as `c85af58d9aa815504e006e736df6dc16042ee76c`. The merge tree is byte-identical to tested head `12c16cb23af3d1e1548a3d3bd22074558dd4ae33`; local `main` is clean and level with `origin/main`.
- Proved clean public MCore reachability by fetching `9c82d4ca` into an empty repository directly from the Bridge-recorded NVIDIA URL. Removed the two clean NeMo task worktrees and restored the exact submodule pins in the sole remaining main checkout.
- Reverified the preserved TP2 checkpoint under `step_1/policy/weights/iter_0000000`: 288 `.distcp` shards totaling 8,946,006,731,964 bytes, 35,338,056-byte `.metadata`, and three small control files. No reservation, checkpoint, or conversion-cache mutation was issued.

## 2026-08-24 09:28 CEST

- User requested deeper analysis of the `0.0027` GLM generation KL and a ten-step run rather than relying on one checkpoint step.
- Recovered the ten-step R3-off body from job `3147936`. Per-step KL ranged from `0.0022966` to `0.0027089` with mean about `0.00250`; however, token-multiplicative-error tails were severe. Applying both token and sample loss masks leaves 3,858,221 valid tokens: 7,873 had `abs(delta log p) > 0.5`, 570 exceeded `1.0`, and the maximum reached `37.7257`.
- Decision: run a fresh TP2 ten-step R3-on experiment from the merged stack, not from the route-less saved replay buffer. Gate average metrics, tail metrics, route trace integrity, learning signal and clean finalization; disable checkpointing and preserve the complete checkpoint and reservation.

## 2026-08-24 11:16 CEST

- Job `3171492` completed ten TP2/PP18/EP16 legacy-async GRPO steps with TP32/EP32 vLLM and Router Replay enabled. Per-step generation KL was `0.0003615-0.0004061` (mean `0.0003876`), versus the R3-off control mean `0.00250`.
- Direct masked-token evidence improved from 7,873 errors above `0.5`, 570 above `1.0`, and maximum `37.7` in the control to four above `0.5`, zero above `1.0`, and maximum `0.676` across 1,291,712 valid tokens.
- Only five steps carried learning signal. Response truncation was `94.5-100%` per step under the 1536-total / 1024-new-token envelope; the next experiment will use 2048 total / 1536 new tokens and retain the strict eight-of-ten signal gate.
- The training process printed `Async GRPO training complete!`; the wrapper failed afterward because its trace checker assumed the SingleController/TransferQueue event schema. The legacy async path correctly emitted Router Replay assignment/action/forward-verification and CP-identity records but no TransferQueue producer/fetch records. This is a harness contract bug, not a model or training failure.
- A broader path audit is in progress before the rerun: make the trace transport contract explicit and reject unsupported legacy-async plus data-plane configurations before expensive setup.

## 2026-08-24 publication preparation

- Implemented explicit `legacy-async` and `transfer-queue` Router Replay trace contracts. Legacy async now rejects TransferQueue records, while the SingleController contract requires producer/fetch integrity and every expected forward stage.
- Centralized test-suite completion handling so direct/final runs require `train/loss` through `MAX_STEPS`, chained intermediate runs remain valid, and original nonzero exit codes are preserved. GLM artifacts are now isolated by Slurm job ID to prevent stale evidence reuse.
- Strengthened the next GLM gate to 2048 total / 1536 generated tokens, at least eight learning-signal and nonzero-loss steps, mean truncation below 0.9, no valid-token error above 1.0, and fewer than 1e-4 above 0.5.
- Added fail-fast entrypoint and SingleController config validation, including unsupported transport/backend combinations and formerly accepted no-op controls. Applied configured GRPO advantage clipping in the SingleController training path.
- Current-source compilation, shell syntax, diff hygiene, and 13 pure trace/completion unit tests passed. Dependency-bearing tests remain an exact-image gate because host Python lacks Ray.

## 2026-08-24 15:32 CEST

- Submitted exact-image changed-path job `3173736` from PR head `5143b429d`. It started Ray and passed 242 tests before the broad SingleController suite exposed one stale fixture: `_setup_master_config` modeled the new SingleController path but inherited the default legacy `grpo.async_grpo` block.
- Decision: preserve the production fail-fast guard, set `async_grpo=None` in that fixture as the nearest SingleController fixture already does, and rerun the full exact-image gate from a new committed head. The 80-node GLM run remains held until the rerun exits green.
- Committed and pushed the fixture correction as `dfc9e50bb`. Exact-image rerun `3173779` passed all 420 tests and Ruff check, then failed closed on Ruff format for a single existing PR-side line in `test_resiliency_config.py`. Exact-image job `3173814` applied only Ruff's mechanical reflow and completed `0:0`; a new immutable-head full rerun is next.

## 2026-08-24 18:28 CEST

- Preserved the complete ten-step legacy-async run as the Router Replay control: all ten train steps completed, mean generation KL was `0.0003867`, and only one of 1,918,607 valid tokens exceeded absolute logprob delta `0.5`; the terminal failure was the independent 94.77% response-truncation gate.
- Decided to certify the current SingleController + TransferQueue path instead of rerunning legacy async. Prepared an ignored one-step probe at `.tmp/glm51-sc-probe/` with the same 80-node GLM topology, a 3072/2560-token envelope, R3 validation, and fail-closed terminal checks.
- Host config validation stopped at the known environment boundary (`ModuleNotFoundError: ray`). The probe repeats Pydantic construction and cross-section validation inside the certified image before actor setup.

## 2026-08-24 23:10 CEST

- SingleController/TransferQueue one-step job `3175340` reached the current runtime path: all 320 Ray worker units connected, 160 actor environments completed, initial NCCL reshard refit took 5.817 seconds, rollout produced all prompt groups, and the trainer fetched one TransferQueue batch.
- The first prev-logprob fetch failed in `_broadcast_batched_data_dict` because the R3 payload carries `routed_experts` as `torch.int16` and NCCL does not accept `Short`. The trace records the exact `[8, 2171, 78, 8]` int16 route tensor; no training step ran.
- Diff against `upstream/main` is empty for the broadcast helper and its existing Gloo-only test. Decision: fix the generic data-plane wire representation, not the GLM recipe or R3 storage dtype, and require a real NCCL regression before rerunning the 80-node gate.
- Implemented an exact-byte NCCL wire view for int16 tensors, preserving route dtype, values, and compact storage outside the collective. Added int16 coverage to the Gloo test and a two-GPU NCCL regression using the repository distributed-test fixture.
- Host Python compilation, Ruff check/format, `git diff --check`, and a standalone bit-exact int16 byte-view round trip passed. Attempts to submit the exact-image one-node gate did not create a job because `sbatch` and `squeue` timed out against the Slurm controller; the reservation was not mutated.
- The controller calls were sandbox-blocked rather than unhealthy. Exact-image job `3178396` started immediately and exposed that Torch 2.11 Gloo also rejects int16 with `RuntimeError: Invalid scalar type`. Generalized the byte wire from NCCL-only to every backend; this makes the existing CPU/Gloo test exercise the conversion as well as the dedicated NCCL test.
- Corrected exact-image job `3178422` passed all three focused tests, including the real two-GPU NCCL round trip, in 57.98 seconds; Ruff and formatting also passed. Online source review confirmed NCCL exposes 8-, 32- and 64-bit integer types but no 16-bit integer type, PyTorch's NCCL mapping omits `Short`, and established arbitrary-payload broadcasters use `uint8` wire buffers. Keep the exact byte view rather than widening route IDs to int32.

## 2026-08-25 08:46 CEST

- Refactored the int16 wire conversion through the existing packed-tensor abstraction and added non-contiguous plus scalar coverage in signed commit `070abefa6`.
- Rejected a 60-line producer-side padding candidate because the policy consumer already owns the topology and the candidate could preserve a stale `global_forward_pad_seqlen` after restore. Replaced it with a recomputed consumer invariant and one focused regression in signed commit `314e22e20`.
- Local pure policy tests passed 10/10; local Ray/Gloo startup was unavailable due hostname resolution, so exact-image job `3181797` ran the authoritative focused gate. It passed all 17 tests, Ruff, formatting, Gloo, and real two-GPU NCCL in 58.18 seconds.
- Submitted controlled 80-node one-step SingleController/TransferQueue job `3181802` from immutable head `314e22e20`. It uses the proven 72-train/8-inference split and 3072/2560 token envelope.
- User directed the next scale run to spend more nodes and tokens on inference. On a green one-step result, use 88 total nodes with 16 inference nodes (two TP32 vLLM replicas) and 4096/3584 tokens; do not relax the truncation or learning-signal gates.

## 2026-08-25 10:00 CEST

- Controlled 80-node job `3181802` completed one SingleController step: loss `0.0735495`, grad norm `0.305298`, reward `0.1484375`, advantage range `-2.26778..1`, generation KL `0.0003694103`, 306,986 valid tokens, and no stale/dropped/replaced/promoted groups. The terminal postcheck alone failed because the rollout actor and 288 policy ranks independently sampled different trace windows.
- Corrected the trace validator to retain the forward producer-to-`prev_lp`/`train` contract while allowing independently sampled worker fetches. The complete job trace then passed with 144,200 records, 38,400 route assignments, 57,600 replay actions, 38,400 forward verifications, and 576 CP records covering 1,382,544 token rows.
- Added compact SingleController valid-token logprob-tail metrics using the exact shifted GRPO loss mask. The fork validator now supports those scalar series as well as the legacy per-step JSONL mode, preserving the absolute-delta gates without large trajectory dumps.
- Exact-image job `3182189` passed 76/76 focused tests in 60.61 seconds, including the SingleController metrics, both validator modes, trace contract, padding, Gloo, and real two-GPU NCCL. Ruff, formatting, shell syntax, and diff checks are clean.

## 2026-08-25 ready-first matched experiment

- User approved a matched `ready_first` comparison to test whether retaining late prompt groups reduces the trainer wait and wasted rollout work shown by the `windowed` baseline.
- Baseline job `3182849` is running on 88 nodes from `6366a7599`; by step 6 it had logged completed training steps with strict KL/tail evidence and concrete stale-work waste, including 14 in-flight aborts after step 5.
- Created isolated branch `autoresearch/glm51-ready-first-20260825` from the exact baseline source. The experiment changes only the sampler policy; all topology, model, response envelope, Router Replay, and validators remain fixed.
- Committed the hypothesis and recipe as `d071be968`. Host YAML parsing and shell syntax passed; the in-container preflight remains fail-fast because the existing code allocation could not be reached through Slurm during validation.
- Submitted ready-first job `3184823` with `afterok:3182849`. It is pending on the baseline dependency and cannot compete for the 88 reserved nodes.

## 2026-08-25 14:08 CEST

- Live baseline `3182849` remains running on all 88 nodes. It has completed eight of ten steps; logged step times are `1694.29`, `883.35`, `1519.22`, `1323.29`, `1639.41`, `1267.55`, `1472.70`, and `1152.62` seconds, averaging `1369.05s` (`22.82min`). Step 9 is in progress; matched ready-first job `3184823` remains dependency-pending.
- The trainer's directly measured `exposed_generation` wait across those steps is `10044.23s`, averaging `1255.53s` (`20.93min`) per step and `91.71%` of step wall time. This is an upper bound on window-policy cost, not an attribution: it also includes unavoidable time to generate the 16 useful groups. The causal window penalty is the matched difference versus ready-first after job `3184823` runs.
- Corrected the rollout-delay explanation against `single_controller.py`, `staleness_sampler.py`, `rollout_manager.py`, and the live log. Step 1 creates policy `v1`; step 2 trains `v1` using eligible `v0`/`v1` trajectories, then creates `v2`. The 21 `v0` tasks aborted after step 2 were not directly awaited; they occupied concurrency slots during step 2 and therefore constrained the rate at which fresh `v1` work could launch.
- Replaced the misleading visualization with a preview that separates model-weight versions from rollout-data versions and explicitly shows `v0 -> v1 -> v2`: `/users/xyixuan/.codex/generated_images/01a014d5-9519-78d1-87a5-1decdc434e47/exec-d5dce044-1140-43e5-9c9b-1269dc1165ca.png`.
- Baseline step 9 completed in `1449.77s`, with `1359.12s` of exposed generation and one additional completed-group eviction. Across nine steps the windowed means are now `22.97min` total and `21.12min` exposed generation; cumulative sampler waste is 67 aborted in-flight groups and 5 evicted completed groups.
- Verified the experiment does not set sampler behavior through an OS variable. Commit `d071be968` carries a dedicated YAML recipe whose `async_rl.sampler.name` is `ready_first`; the launch environment only supplies output placement and Slurm dependency orchestration.

## 2026-08-25 14:40 CEST

- Windowed job `3182849` completed ten of ten training steps and printed `SC run complete`; final-step loss, grad norm, reward, and generation KL were finite and healthy. During the launcher's `finally` teardown, vLLM shut down its engine manager, Megatron ranks observed the closing TCPStore, and the driver returned exit code 2 without a training traceback. Because the shell remained in `GLM_PHASE=training`, the failure artifact over-broadly labels this as a training failure and the later trace/metric postchecks did not run.
- Slurm therefore left matched ready-first job `3184823` in `DependencyNeverSatisfied` under its original `afterok` contract. Updated the existing job dependency to `afterany:3182849`; the same already-approved 88-node job started immediately, with output rooted at `/iopsstor/scratch/cscs/xyixuan/nemo_rl_glm51_ready_first/d071be968`.
- Rebaseability reminder: the planned root `FORK_PATCHES.md` does not exist yet. Add it after this experiment to make nested MCore patches, upstream status, validation evidence, and absorption rules visible before future Bridge or NeMo-RL bumps.

## 2026-08-25 15:50 CEST

- Ready-first job `3184823` remains active with no Ray completion, exit, or fatal-error marker. It completed step 1 in `2456.12s`, of which `2154.79s` was exposed generation, and is now generating step 2.
- Step-1 correctness is healthy: loss `0.06967`, grad norm `0.32295`, reward `0.21875`, generation KL `0.0004083`, three valid-token logprob deltas above `0.5`, none above `1.0`, and zero stale evictions, aborts, drops, or masked sequences.
- The matched windowed step 1 was `1694.29s` total / `1430.13s` exposed generation. Do not attribute the ready-first step-1 slowdown to stale-work policy because no trajectory can yet be stale; evaluate the hypothesis from steps 2-10 and cumulative sampler waste. Slurm accounting was temporarily unavailable, so liveness was verified from the continuously updating submit-side log and absent Ray terminal markers.

## 2026-08-25 16:20 CEST

- Ready-first job `3184823` completed steps 2 and 3 in `852.25s` and `1559.06s`; step 4 is generating. All three completed steps remain correctness-green, with generation KL at or below `0.000521` and no stale eviction, abort, drop, or masking.
- Against matched windowed steps 1-3, ready-first is `770.57s` slower cumulatively. Steps 2-3 alone differ by only `8.74s` (`0.36%`), so essentially the entire gap is the first rollout batch (`2456.12s` versus `1694.29s`). Stale reuse cannot affect step 1, but sampler admission can: ready-first caps admitted lookahead while windowed can refill freed rollout slots. Continue through ten steps before separating that tradeoff from warm-up and stochastic tail variance.
- Direct timer/code audit confirms real trainer idleness: `exposed_generation` wraps the loop waiting for a selectable 16-group batch. Across ready-first steps 1-3, corrected idle time is about `90.4%`; the steady optimizer step is about `67s` versus `741-1447s` exposed generation. The 72-train/16-inference topology supplies 288 training GPUs but only two TP32 vLLM replicas on 64 inference GPUs, so rollout capacity is the dominant bottleneck.
- Found an inherited upstream instrumentation bug in `nemo_rl/algorithms/utils.py`: `training_worker_idle_time_ratio` returns zero when `exposed_generation_time > 0.1`, exactly reversing the intended small-wait clamp. Keep diagnosis separate from a future source fix.

## 2026-08-25 16:43 CEST

- User chose not to spend the remaining hours on the 88-node ready-first run and explicitly requested cancellation. Cancelled only Slurm job `3184823`; reservation `SD-69241-apertus-1-5-0` remains untouched.
- Preserve the three completed correctness-green steps as partial evidence. The next authorized experiment changes only inference capacity: 72 training nodes remain fixed, while inference increases from 16 to 32 nodes, producing four TP32 vLLM replicas in a 104-node allocation.
