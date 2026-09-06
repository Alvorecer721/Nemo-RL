# PP foundation integration ledger

## Scope and status

The approved source integration, dependency rebuild, focused checks and bounded 70B optimizer-resume qualification are complete. The requested microbatch 2 trial is also complete and failed with OOM; the qualified recipe retains microbatch 1. Publish the dependency branches, create a PR to `Alvorecer721/Nemo-RL:main`, and perform the requested fresh local review, then stop. Do not merge main or implement rollout PP in this change.

The original dirty checkout, other worktrees, historical checkpoints and user allocations were preserved. All source work uses the isolated `integrate/2026-09-05-pp-foundation` branch and independent Bridge/MCore integration clones.

## Source and dependency identities

| Component | Integration input | Final pin / result |
|---|---|---|
| NeMo-RL | Fork `b7a6a2a61d8d9aa76becd156d123c36a58159ceb`; upstream `5368eff5f6bc7287f5a3d7fdc762530396610b1c` | Merge `440ae6a22` includes all ten remaining upstream commits; runtime qualification source is `b6a7903ae177f8f1213c5824c346a4607161b18e`. |
| Megatron-Bridge | Fork `3ce0cddefc84ba8f18515e410bb913978d27d63c`; upstream `75d7b9eb89b05b7edbef7daf931045c38c6ca9df` | `1d9a69fde22cb1e607f7ff84e4b62cc9e890dc75` |
| Megatron-Core | Fork `7851ef7cb9fbd261d592b58e3ede406aa151e60b`; upstream `b1fe7599e18a213292600177ad7ff290a1127160`, plus applicable PR7088 | `1e5025f8521cb5b6c3c9276429fc012f45b5c291`; Bridge `.main.commit` and gitlink agree. |
| Transformer Engine | Exact source `27486e03cfc1fa41f6932dcecdc47c71c47eac3e` | `2.18.0+27486e03` |
| vLLM | Existing upgraded pin retained | `0.26.0` |

The root lock resolves 562 packages with pinned uv `0.11.28`. Relative to the integration input, the package-identity delta is TE and its required `nvdlfw-inspect==0.2.2`; Bridge/MCore dependency metadata is reconciled. Unrelated backend pins remain fixed.

The PR also contains 16 preceding local commits absent from fork main: the vLLM 0.26 image upgrade, baked worker setup, wire-safe refit metadata, streaming-metric serialization and the 70B microbatch-one correction. Its scope is larger than the ten newly merged upstream commits.

## Integration decisions and corrections

- Preserve Apertus XIELU/conversion, local HF refit views, optimizer live-state reuse, direct-bulk shape guards, wire serialization, ALP configuration and streaming metrics.
- Correct the inherited optimizer-state dtype handling: BF16 remainder initialization follows TE's effective flag and parameter dtype instead of applying to FP32/FP16. The regression failed before the fix and passed afterward.
- Restore eager packed HybridEP padding using NeMo's packing setting after Bridge moved the safeguard to its dataset configuration. Keep graph/backend guards intact.
- Reconcile upstream split-training tests with the later public VLM guard. Internal packing/MTP flag forwarding remains tested; unsupported public VLM split execution remains rejected.
- Enable `logprob_chunk_size: 256` together with `defer_fp32_logits: true` in the optimized 70B recipe after numerical parity and the measured memory/time tradeoff were accepted. Minimize the recipe without changing its 356 inherited configuration values, interpolation strings, scalar types or header comments.
- Include `transformers_compat.py` in the Dockerfile's early source subset; fresh setuptools metadata loading otherwise fails before the full package COPY. Refresh the four dependency-cache constants rather than bypassing fingerprint checks.
- Use `grep -Fq` for launcher completion checks because the candidate image does not contain `rg`.
- Local commits carry DCO sign-off. GPG signing was disabled per command because the configured identity has no secret key; user signing configuration was preserved.

The [activation audit](docs/guides/pp-foundation-activation.md) records available but unqualified features and missing connections. Local training graphs and TP userbuffer initialization already have entrypoints; stable shapes, worker-package availability and numerical/performance gates determine whether they can be enabled.

## Built image

The 334 GiB allocation-local Podman graph required two fresh allocations. Job **3303718** rebuilt and published the hermetic dependencies, completing `0:0` in **1:20:19**. Its fingerprint and package-file digests matched the committed target pins. Job **3303932** assembled the release image/SquashFS and passed embedded version/API checks, completing `0:0` in **1:32:19**.

```text
image: /iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.26.0-b339b0aba11c-27896e220d80.sqsh
bytes: 43246944256
sha256: 10f24b38180f071525c3fe3708f86bf76486e636381e2f1aefcecb9cc361a113
image source: b339b0aba11c4d813db03a70cd03230e33dadf73
hermetic fingerprint: 679a0eb8c22b64929c8db02890c1d3c343b68c7ba3193b96713473a4e17033cb
pyproject MD5: 5bcd5feb94a97da7fe078f0ec1c1a902
lock MD5: e266ae4f7b43eefdaf328f5387dfc309
release input SHA256: 27896e220d80e03909227595e934987366bb339647c8364bdf22b952e8b4df11
```

The later source/config/launcher corrections run from a clean, explicit checkout overlay. They are not baked into this image; dependency inputs and fingerprints are unchanged. The focused worker probe reported TE `2.18.0+27486e03`, torch `2.11.0+cu130`, and PyTorch NCCL **2.28.9** on GH200 120GB. The 70B refit logs also contain native NCCL **2.30.7** banners; the PyTorch probe must not be generalized to every communicator in the process. `NVTE_WITH_NCCL_EP=0` remains set.

## Validation

| Check | Evidence |
|---|---|
| NeMo CPU regressions | Job3303593: 109 passed after reconciling two inherited padding/value fixtures. |
| MCore optimizer/MLA | Seven focused cases passed, including the optimizer dtype regression. |
| Bridge CPU/static | 27 cases and full Bridge pre-commit passed. |
| New image GPU suites | 29 Bridge, 29 vLLM refit/HTTP, four distributed-logprob and 156 split/refit/router/VLM-setup cases passed. |
| HybridEP correction | Baseline3304305 reproduced exactly two intended failures and eight guard passes; post-fix3304319 completed `0:0` with all ten regression cases and 49 surrounding checks passing. |
| MCore RNG | Job3304307 completed `0:0` in 2:08: eight tests passed on every one of eight ranks, with no skipped/failed cases. |
| Root static/config | Changed Python Ruff check/format passed; job3304381 completed `0:0` with zero Pyrefly errors under the configured allow-list/ignore rules in the candidate image. The 70B recipe passes minimize-check with independently verified inherited-config equality. |
| 70B initial training/save | Job3304323 completed updates 1–2 and saved the model/optimizer, then failed in launcher postflight with exit127 because `rg` was absent. |
| Initial postflight recovery | Job3304365 completed `0:0` in 31 seconds using the corrected checks against the existing artifacts. All 274 logged scalar metrics were finite; both updates contained 768 valid samples. Original failed Slurm/terminal records were preserved. |
| Fresh 70B optimizer resume | Job3304396 completed `0:0` in 10:55: updates 3–4, 274 finite scalar metrics, final checkpoint and direct Adam/scheduler counter continuity passed. |

The RNG harness correction detaches peer `TempNamedDir` deletion finalizers while preserving rank-0 cleanup and the original barriers. The uncorrected helper raced to delete a shared checkpoint directory, producing ESTALE and stalled rank teardown. This was a test-helper correction only; MCore production code and its pin did not change.

### 70B setup and artifact checks

The bounded check uses 16 nodes / 64 GPUs: 12 trainer nodes at TP2/PP4/DP6 and four rollout nodes at TP4/PP1. Model/tokenizer are the existing `ap1p5-70b-sft-262k-2700_corr` checkpoint. Dataset identities are unchanged: cached GSM8K with 7473 training and 1319 test examples. Inputs are unpacked with a 2048-total-token limit, thinking disabled in the template, 48 prompts × 16 responses, global batch 768 and train microbatch1. Sampling is windowed age1, TIS2, `force_on_policy_ratio=true`, LR1e-6 and weight decay0.01. W&B is disabled; TensorBoard is enabled.

The first two updates took 116.30 and 56.34 seconds inside the step timers; policy training took 62.05 and 40.69 seconds. These reflect startup/async scheduling differences and are not a matched performance comparison. Both had finite loss, gradient norm and generation-KL diagnostics.

The step-2 checkpoint contains all 48 DCP shards totaling 1,019,550,895,101 bytes, including optimizer master parameters and Adam moments, correctly located tracker/train-state files, replay/data-plane state and dataloader state. The completion log and serialized `checkpoint.save` show an earlier path because of a mutable asynchronous configuration field; captured write destinations and the original cache metadata were independently checked. NeMo rebuilds the resume path explicitly.

The first fresh resume attempt,3304367, failed before updates because extending the bounded run from two to four changed the derived weight-decay horizon from1536 to3072 while strict scheduler matching was enabled. The resume-only configuration now sets `policy.megatron_cfg.scheduler.use_checkpoint_opt_param_scheduler: true`, with override disabled. This restores the saved scheduler/counter along with the optimizer; constant LR and weight decay are preserved.

Fresh resume **3304396** completed `0:0` in **10:55**. All 48 trainer workers loaded the saved checkpoint; 87 replay groups were restored. Updates 3–4 took 88.57 and 53.90 seconds, with policy training taking 60.50 and 41.36 seconds. All 274 logged scalar metrics were finite, both updates used 768 valid samples, and final training state reports step/version4 and 192 consumed prompts. The final checkpoint again contains all 48 expected shards.

An independent metadata comparison found all 16 saved Adam group counters advanced **2 → 4**, and the scheduler counter advanced **1536 → 3072**. Every other saved optimizer-group and scheduler setting remained unchanged. This rules out a full optimizer-counter reset in the tested resume; it does not assert bitwise equality of all moment tensors or RNG state.

## Microbatch 2 trial

Both arms used source `9eba46778d01ed79a121160f78a7ff5c76ebd185`, the same dependency image, model/data/seed and global batch 768, with checkpoint writes disabled. The only independent configuration change was `policy.train_micro_batch_size: 1` versus `2`; disabled packing/dynamic-batching token budgets inherit this value. Both enabled existing per-rank allocator logging. These allocations exposed 95 GiB per GPU; the isolated logprob benchmark used a GH200 120GB device.

- **MB1 control 3304520:** completed `0:0` in 12:01; all four updates used 128 microbatches and 768 valid samples. All 273 scalar metric series were finite, and all 48 training ranks logged four completed steps. Warm policy times (steps 2–4) were 40.84/40.48/41.28 seconds, mean 40.87; full-step times 45.06/58.15/46.44 seconds, mean 49.88. Step 3 includes 13.14 seconds of exposed generation. Maximum recorded allocated/reserved memory was 78.20/87.80 GiB.
- **MB2 candidate 3304521:** failed `1:0` in 9:21 before its first optimizer update. Rank 10 exhausted memory in TE Linear weight-gradient GEMM: a 336 MiB allocation failed with 326.81 MiB free on a 95 GiB device; 89.79 GiB was allocated by PyTorch at the failing site. Other ranks recorded allocation peaks up to 90.12 GiB before termination. Partial interval logs are not a complete all-rank peak measurement.

Retain MB1. The trial establishes that chunked logprobs alone do not make this MB2 configuration fit. It provides no MB2 throughput or convergence comparison. A later MB2 attempt needs an additional measured activation-memory reduction; reduced recomputation would increase memory pressure. Raw logs, terminal states, resolved inputs and the analysis are in `session/20260905_114410/pp_foundation/microbatch-ab/`.

## Qualification limits and records

This is bounded stack validation, not a convergence or full-step speedup claim. It does not establish rollout PP2, GLM DSA/model-level MTP execution, all historical checkpoint formats, exact NeMo/Bridge RNG continuity, FA3, training CUDA graphs, FP8 training or TP userbuffer overlap. NeMo currently sets `load_rng=false`; passing MCore training RNG tests does not change that.

Detailed reports, source reviews, JUnit files, job logs, model/data hashes, image descriptor and qualification JSON live under `session/20260905_114410/pp_foundation/`. Run artifacts use `validation/20260906-foundation-94a9fdb0a-chunk256/`; this run ID remains stable across the documented retries. Checkpoints are isolated under `/iopsstor/scratch/cscs/xyixuan/nemo_rl_pp_foundation_validation/20260906-foundation-94a9fdb0a-chunk256/checkpoints`.
