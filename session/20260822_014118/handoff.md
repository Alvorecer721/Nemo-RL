# Handoff

## Final Checkpoint — 2026-08-23 07:05 CEST

- PR #24 is merged. Public fork `main`, GitHub's recorded merge commit, and the certified image source are all the same commit: `8f22e59195f547c5715ed250cd49d4776cda5d43`.
- Exact release image job `3156560` completed `0:0`. Artifact: `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.sqsh`, 50,049,777,664 bytes, SHA-256 `4aaf2b1bba8613a1e515281d84ab9e330c41d2774ccd3992b5f0c0f81e9dd002`.
- Final Apertus preflight `3156865` and PP2 refit regression `3156866` completed `0:0`; PP2 generation KL was `0.0002` after refit, generation, logprobs, backward and training.
- Apertus-1.5 70B production smoke `3156886` completed `0:0` on five nodes. Three async GRPO steps and three NCCL-reshard refits passed; every step had reward range `0..1`, nonzero advantages, nonzero finite loss/gradients, 128 valid samples, and generation KL below `0.001`. Final E2E throughput was `128.98` tokens/s/GPU at `14.96%` measured training MFU. Terminal evidence is `.tmp/apertus70b-exact-cert/logs/apertus70b_async_smoke/terminal_green_20260823T044937Z_235193.json`.
- GLM-5.1 exact-image scale prototype `3156867` completed `0:0` on eight nodes/32 GH200. It loaded all 282 local shards offline, passed semantic arithmetic, measured `92.936` output tokens/s and about `74.00 GiB` peak HBM/GPU. This is a fit/throughput prototype only; 288-rank optimizer checkpoint completion, long endurance and the reference-KL path remain unresolved.
- The requested twelve upstream commits are included in frozen base `7ea279abf`. One newer upstream commit appeared after certification: `f0557321e`, which fixes only the sequence-level importance-ratio metric. The certified 70B recipe disables that mode, so defer it to a source-only follow-up rather than invalidating exact-image identity.
- Published harness branches: `autoresearch/2026-08-23-apertus70b-cert/exact-image` at `92db67ae0` and `autoresearch/2026-08-23-glm51-prototype/exact-image` at `76217c755`.
- Root checkout is on `main` at `8f22e5919`; Gym and Bridge worktrees are updated to their pinned commits. A pre-existing untracked compiled helper remains at `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets/helpers_cpp`; it was deliberately preserved because it is user/build data, not release source.

## Remaining Follow-ups

- Cherry-pick or absorb upstream `f0557321e` after deciding whether to retain exact image-to-HEAD identity or schedule the next source/image refresh. It needs no dependency rebuild by itself.
- GLM is not production-certified. Diagnose the historical 272/288-shard async optimizer checkpoint stall before another 80-node run, then prove a fresh-allocation resume and longer real-DAPO endurance.
- Runtime-prove the isolated nonzero-reference-KL GLM path before publishing it; current accepted GLM throughput evidence is intentionally KL-free.
- Automodel's Torch-version string comparison still disables SGLang async checkpointing under Torch 2.11; fix as a narrow Python follow-up. The SGLang DTensor-v2 text/refit path itself passed the prior exact-stack gate.

The core deliverable is complete. Do not reopen the Apertus PP/refit investigation unless a new run violates the permanent PP2 KL gate.

## Latest Checkpoint — 2026-08-23 02:42 CEST

- Active final-sync worktree: `.tmp/nemo-upstream-7ea-curated`, clean branch `chore/sync-upstream-7ea279abf`, exact release-image commit `084ade845b8421ab82dcda1849d913da517f194e` on frozen upstream base `7ea279abf`.
- Curated replay, refit static-state repair, model-agnostic refit-manifest fail-closed checks, Ray single-step CSCS launch, GRPO post-validation refit, changed-path tests, Ruff, Bash/YAML/lock checks, vLLM patch tests, and Bridge Apertus export tests are green.
- Hermetic dependency build job `3153709` completed and published cache fingerprint `ae440a7e8b2ab39353e47d5d879e9cfcbb3f6bdda5ca3133015592611772141d`; the exact guards are committed in `084ade845`.
- Fresh-allocation release assembly job `3155280` is running on `nid007642`. It resumed from that hermetic cache, verified vLLM 0.25.1 and lock MD5s, and has reached layer 49/56 after successfully materializing every vLLM, SGLang, DTensor, Megatron, quantized-policy, and async-trajectory worker environment. Podman remains CPU-active and the 429 GiB local workspace is healthy.
- Intended output is `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-084ade845b84-a14fb058fe83.sqsh`. Do not launch source-bound certification until the file exists, its labels match `084ade845`, and job `3155280` is scheduler-confirmed `COMPLETED 0:0`.
- After the image is accepted, run the exact-image ladder, ending with Apertus-1.5 70B TP4/PP4 Megatron plus TP4 vLLM async GRPO: low generation KL, nonzero reward/advantage variation, finite nonzero loss/gradients, repeated refits, and cross-allocation optimizer checkpoint resume. Keep GLM-5.1 explicitly as a scale prototype with throughput/MFU and failure evidence, not production certification.
- Only after current-image certification settles, fetch and curate the small newest upstream delta. Do not move the base or invalidate image evidence mid-run.
- Exact-image Apertus certification harness is committed at `a196ee1a5` in `.tmp/apertus70b-exact-cert`. It runs production imports only from `/opt/nemo-rl`, records baked source `084ade845` separately from the harness SHA, uses a deterministic 64-row real-DAPO overlay, and requires every one of three 16x8 grouped steps to have nonzero reward/advantage ranges plus generation KL below `0.002`.
- Exact-image GLM scale-prototype harness is committed at `315b89b53` in `.tmp/glm51-exact-prototype`. It is an eight-node/32-GH200 real-weight vLLM TP32/EP32 async generation and semantic-throughput probe with low non-ephemeral Ray ports and baked-source attestation.
- Historical GLM benchmark `3147936` is scheduler-confirmed `FAILED 1:0`, but reached Step 10/10 and emitted generation KL `0.0024` before distributed finalization failed. Treat its learning/throughput body as prototype evidence and the exit as an unresolved teardown failure, never as a production pass.

## Latest Checkpoint — 2026-08-22 14:00 CEST

- Apertus-1.5 PP>1 refit corruption is now the active fixing lane. The 70B TP4/PP4 trainer to vLLM TP4 NCCL-reshard run had generation KL `0.58–0.73`; an exact-image Apertus-8B TP2/PP2 colocated control failed loudly with rank-dependent synthetic xIELU `beta`/`eps` keys (`unexpected` stage-1 keys against rank-0 metadata).
- Root-cause boundary: the xIELU arithmetic/kernel is not implicated. The Bridge hook conditionally synthesizes untrained constants only on the owning PP stage, while vLLM dummy loading randomizes persistent buffers and its CUDA path separately caches scalar copies.
- User approved the robust fix: remove `beta`/`eps` from recurring refit, make them Apertus architecture-owned/config-backed static state with one buffer/cache initializer, and add only model-agnostic PP/transport fail-closed invariants to NeMo-RL core.
- Clean isolated superproject worktree: `.tmp/apertus-refit-fix`, branch `fix/apertus-pp-refit-static-state`, base `58ffe9ae1cd980b992db48a4e9f4d7dfae6864e2`. Submodules are at the exact tested pins; nested Megatron-LM was initialized from the validated local checkout after sandbox DNS blocked GitHub.
- No production edits or new jobs have been launched yet. Keep the active GLM lanes untouched. Definition of done is focused tests, permanent PP2 regression, and a 70B NCCL-reshard rerun with KL returning to the normal low range before folding changes into PR #24.

## Latest Checkpoint — 2026-08-22 11:40 CEST

- Campaign is not complete. GLM benchmark `3147936` is RUNNING and has completed 3/10 valid learning/refit steps. Step 3 had 256 samples, nonzero loss/reward/advantages, age 1, and no runtime error; it is rollout-starved while target 3 is actively decoding for Step 4.
- KL reference retry `3147942` FAILED in first reference logprob from a proven 450-GiB Slurm host-memory cgroup OOM after 288 primary plus 288 reference loads and initial refit. The 850000M retry `3148084` then FAILED earlier in primary DCP load because the recipe omitted `ckpt_fully_parallel_load_per_rank_objects: true`. The narrow recipe/test fix is implemented but awaits a fresh exact-image gate, commit, and rerun.
- Checkpoint Phase A `3147961` reached Step 1, optimizer update, refit, and wrote about 7.7 TiB of temporary shards, then FAILED because Bridge immediately entered a duplicate global barrier while the async saver still used collectives. Bridge `d069afb3` and superproject `4c758f73` contain the clean signed fix. A fresh exact-image gate and repaired Phase A are next; Phase B remains unlaunched.
- Apertus-70B final xIELU/config preflight is green, but no new representative-batch async learning rerun has been launched. Nonzero reward/advantage evidence remains open.
- PR #24 remains OPEN at `a2de6675`; substantive CI is green and only PR-comment posting is red. None of the newly isolated runtime fixes has been folded into the PR yet.

Do not merge PR #24 until the runtime proofs identify which fixes are accepted and only those proven commits are replayed onto the PR branch. Do not call Phase B complete until a fresh allocation restores a finalized Phase-A checkpoint and executes the next optimizer step.

## Latest Checkpoint — 2026-08-22 10:14 CEST

- NeMo-RL upstream-sync PR #24 is open and substantive CI is green. GitHub reports `UNSTABLE` only because the PR-comment posting job failed; lint, lockfile, unit, e2e, docs, copyright, secret, and submodule checks passed.
- Final Apertus-70B xIELU split-environment preflight `3147963` completed `0:0` from clean head `58ffe9ae1`: config boundary, CUDA BF16 forward/backward, four focused tests, Ruff, and formatting passed.
- Matched raw-HF DAPO-prompt control `3147953` completed `0:0`: 3,774 tokens at 509.13 generated tok/s. The exact hard prompts were mostly answered incorrectly or truncated even before Megatron conversion/refit, so the earlier all-zero reward batch is not evidence of transfer corruption. A full 70B async run with nonzero reward/advantage remains unproven.
- GLM 10-step throughput/MFU benchmark `3147936` is running from frozen head `5890dfa3`; actor-venv prefetch is nearly complete and no hard error has appeared.
- GLM KL=0.01 reference-policy smoke `3147942` is running from clean head `df824a6f`; actor-venv prefetch is progressing and no hard error has appeared.
- GLM cross-allocation checkpoint Phase A retry `3147961` is running from clean head `f71bd12b`; exact-image gate `3147952` passed. Phase B must be a new allocation after Phase A finalizes a valid checkpoint.

Do not call the campaign complete until the three running GLM jobs reach terminal evidence, checkpoint Phase B passes, and the Apertus-70B learning-signal decision is closed.

## Read First

The Apertus-1.5 70B track is complete and selected TP4/PP4 from three matched 10-step, terminal-green GH200 runs. The active work is GLM-5.1 async GRPO on 80 nodes/320 GH200 using Megatron TP1/PP18/EP16 training and one vLLM TP32/EP32 rollout replica.

Work only in `.tmp/glm51-async-topology`. Its clean source-bound head is `c2dc2826223788c319eb3ee9526d895c72462949`. Do not edit that worktree while job `3147587` is running because the terminal smoke artifact verifies exact Git head and clean status.

## Active Job

- Slurm: `3147587`
- Recipe: `examples/configs/recipes/llm/autoresearch/grpo-glm5.1-80n4g-megatron-async-vllm-tp32-smoke.yaml`
- Expected artifact: `logs/glm51_async_smoke/terminal_green_20260822T053757Z_95982.json`
- Exact image: `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-e9416845542a-6c7d469c3e2a.sqsh`
- At handoff checkpoint: full 80-node/320-GPU Ray cluster formed and all-node control-actor venv prebuild was running.

Require all of the following before accepting it:

- `sacct -X -j 3147587`: `COMPLETED 0:0`;
- expected artifact exists and validates against `c2dc28262` plus the exact image SHA;
- policy 288 ranks and vLLM TP32/EP32 initialize;
- initial NCCL refit and at least three training refits complete;
- steps 1-3 have valid tokens, positive finite grad norm/step time, finite nonzero loss, nonzero advantage range, and age in `[0,1]`;
- no traceback, Ray actor/task error, or failed marker.

## If Smoke Passes

Launch the ten-step benchmark only through:

`GLM_SMOKE_SUCCESS_ARTIFACT=/absolute/path/to/artifact infra/slurm/cscs/autoresearch/submit_glm51_async_benchmark.sh`

The launcher validates the artifact against exact source and image before `sbatch`. Record the job in `experiments.tsv` before launch and replace the pending row with its real job/artifact paths.

The benchmark launcher hard-codes 12 hours, which is not worst-case safe. At the measured real-weight TP32 rate of 98.741 output tokens/s, 10 full batches at 32 prompts x 8 generations x 1536 tokens require about 11.06 hours of generation alone; async lookahead can raise this to 12.17 hours. After submission, request an 18-hour limit using `scontrol update JobId=<id> TimeLimit=18:00:00` if cluster policy permits, then verify the new limit. This operational adjustment preserves the artifact's exact source identity. If the limit cannot be raised, do not claim a 12-hour timeout is a model/runtime failure.

After terminal success, run the committed analyzer/validator and record:

- steps 3-10 mean/median time;
- valid trained tokens/s and samples/s;
- actual-length-corrected TFLOPS/GPU and MFU;
- vLLM token volume and queue/KV telemetry;
- train/logprob/generation/refit breakdown;
- trajectory-age distribution;
- train versus rollout GPU utilization and peak HBM;
- all 18 pipeline-stage summaries.

## If Smoke Fails

Preserve exact terminal evidence and distinguish the phase. Do not broaden fixes without proving the root cause. The prior two failures are already understood:

- `3147405`: one transient uv filesystem race; fixed by serialized installs, bounded retry, and early all-node prebuild.
- `3147552`: positive `val_period` enabled validation with no validation dataset; fixed and exact-gated at `c2dc28262`. Job `3147583` passed `0:0` with 82 tests, Ruff, formatting and actor import.

## Durable Facts

- Apertus winner TP4/PP4 `3146385`: 19.349 s steady step, 183.35 TFLOPS/GPU, 18.53% external GH200 BF16 MFU, 85.70 valid tokens/s/GPU, 78.72 GiB peak HBM.
- GLM real vLLM proof `3146245`: 98.741 output tokens/s, about 73.98 GiB peak HBM/GPU.
- GLM cache reload `3147363`: `COMPLETED 0:0`, 72/72 ranks, about 35.3 GiB minimum free HBM.
- GLM toy full-pipeline smoke `3147466`: `COMPLETED 0:0`, three async train steps, repeated refits, age at most 1. This proves pipeline plumbing, not the stronger real-DAPO signal/artifact contract.
- KL-free throughput is intentional. The separate KL=0.01 Bridge/MCore fix in `.tmp/glm-reference-object-load` is unit-gated but lacks a real 288-rank reference-policy proof.

## Damage Boundaries

- Do not delete or reconvert `/iopsstor/scratch/cscs/xyixuan/.cache/huggingface/nemo_rl_glm51_tp1pp18ep4` (about 1.488 TB).
- Do not edit the active GLM worktree during a source-bound run.
- Do not call a run passed from wrapper text alone; require scheduler state plus semantic evidence.
- Use escalated Slurm queries. Sandboxed `scontrol`/`sacct` can report false controller/database outages because of isolated DNS/network.
- Do not relaunch Apertus; its matched topology decision is closed.

## Other Open Thread

The production-preserving KL=0.01 reference load fix is isolated in `.tmp/glm-reference-object-load`. It forwards per-rank ShardedObject loading from Bridge to MCore and passed exact unit/config gate `3147419`; it must not be published as runtime-proven until a real 288-rank reference-policy smoke succeeds.
