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
