# Upstream sync ledger (2026-09-10)

Carry/drop record for merging NVIDIA-NeMo/RL `c49d53e2` (with Megatron-Bridge
`4386117` and Megatron-Core `c9b53d0a`) into the CSCS build branch. The
behavior list follows the sync handoff; every row names the files or symbols
that carry it in the merged tree and how it was checked. Evidence sections are
filled in as qualification steps complete.

## Sources and pins

| Component | Before | After | Notes |
|---|---|---|---|
| NeMo-RL | `c00cb93cf` (build branch = main `6196eabe3` + 8) | merge of upstream `c49d53e2e46e95ba645870fead6d48e36c37259b` | 29 upstream commits since merge base `5368eff5`; 28 conflicted files |
| Megatron-Bridge | fork `1d9a69fde22cb1e607f7ff84e4b62cc9e890dc75` | `3880d9e020cc93bb676c21d5e3c9783d225d052c` = fork + upstream `4386117130f9e86024fc694e77ccc9c903899b9d` | conflicts only `.main.commit` and the Core gitlink; merged tree = upstream + the fork's 16-file Apertus/checkpoint delta (+1243/-16) |
| Megatron-Core | fork `1e5025f8521cb5b6c3c9276429fc012f45b5c291` | `c4df534e114a0f948fdf1c22993f56ce6e2dfe01` = fork + upstream `c9b53d0a87cb926f47115259593ecfeb351ca29f` | clean merge; merged tree = upstream + the fork's 5-file optimizer/MLA delta (+306/-9) |
| Gym | `c3bac96314a59f28b896f597eb9845d175bb0252` | `fd5e84d6b1c485c80e7ae61553bbd485611c03b4` | fast-forward; carries the token-capture support NeMo #3837 imports |
| Automodel / kernels | `1814c6c9` / `90b92d6b` | unchanged | |
| Transformer Engine | 2.18.0+27486e03 | unchanged | upstream still on release_v2.15 |
| vLLM / flashinfer / cutlass-dsl | 0.26.0 / 0.6.14 / 4.6.0 | unchanged | upstream 0.25.1 / 0.6.13 / 4.5.2 |
| Python | 3.13.14 | unchanged | already pinned by the Dockerfile and `.python-version` |
| uv.lock | 533 packages | 503 packages | regenerated with uv 0.11.28 against the new submodule trees; see "Dependency identity" |

Fork submodule pins are published as `integrate/2026-09-10-upstream-sync` in
`Alvorecer721/Megatron-LM` and `Alvorecer721/Megatron-Bridge` (Task 5).

## Retained fork behavior

| Behavior | Where it lives after the merge | Check |
|---|---|---|
| Apertus model/tokenizer/XIELU (Bridge provider, layer spec, engine-owned static state; NeMo tokenizer/BOS runtime) | Bridge `src/megatron/bridge/models/apertus/*`, `models/__init__.py`; NeMo `nemo_rl_apertus/`, Apertus topic files (0 of 19 touched by upstream) | Bridge diff vs upstream target is exactly the fork delta; every `megatron.bridge` symbol the Apertus package imports exists at the merged Bridge tip |
| GLM import/refit correctness | Bridge GLM-5 FP8 dequant (upstream #5851 is the same patch as fork `542cab9df`), NeMo MTP/fused-counter handling | present in both parents; no conflict |
| Worker-side serializable metadata (`nccl_reshard_refit_info`, finish-step metrics) | `megatron_policy_worker.py`, `data_plane/worker_mixin.py` (`tree_map(_metric_tensor_to_python, result)` kept together with upstream's `route_fallback_counts`) | conflict resolved as union |
| Bulk local-view refit and shape checks | `_iter_local_hf_param_shards` on `task.local_hf_param_specs()`, o_proj/vocab-parallel bulk whitelist, direct-target shape checks in `vllm_backend.py` | no conflict in these hunks; tests `test_nccl_reshard_backend.py`, `test_nccl_reshard_utils.py` |
| Optimizer restore memory/dtype fixes | Core `distrib_optimizer.py` (`_can_reuse_precision_aware_checkpoint_state`, `_load_optimizer_param_groups_without_state`), `optimizer/__init__.py` BF16 remainder guard, `test_distrib_optimizer_load_state.py` | grep after merge; upstream PR #7088 (MLA guard) still open, guard retained; import regression passes |
| Async checkpoint submission/finalization and refit manifest validation | `lm_policy.py`, algorithm callers | not conflicted |
| NUMA binding | unchanged since the fork's release backport equals upstream `9e01af64b3` | in common ancestry |
| Startup and failure propagation | trajectory collector first-failure traceback and `check_health`; `ray.sub` native-worker gate (build PR); venv readiness markers in `venvs.py` | not conflicted; `venvs.py` conflict resolved to upstream's `--inexact` base sync with the fork's pinned-uv resolution |
| Sequence-group occurrence alignment for GRPO/ALP | `single_controller_utils/rewards.py` (not conflicted), callers in `single_controller.py` | ALP log keys retained (`alp_shaped_rewards`, `alp_successes`, `alp_response_lengths`), raw-reward logging kept next to upstream's `sample_masks` |
| Boxed-answer parsing and prompt rendering | `data/processors.py` `render_single_turn_prompt`, environments | import-only conflict (union) |
| dtype/value-preserving integer transport | `worker_mixin.py` single-broadcast `wire_dtype` path (fork) kept over upstream's two-step int16 copy; `packed_tensor.restore_tensor_from_bytes` kept over upstream's inline closure (same alignment clone and scalar reshape) | tests `test_leader_broadcast.py`, `test_packed_tensor.py` |
| Fused/selected-token logprob guards and tail evidence | `resolve_fused_linear_logprobs` wired into `ClippedPGLossFn(..., opd_full=...)` in `single_controller_utils/setup.py`; `_reduce_token_logprob_error_tails` in `utils.py` | conflict resolved as union |
| Rollout quality and no-signal-group metrics | `_compute_generation_quality_metrics` and `cot_token_ids` in `rollout_manager.py`, kept next to upstream's receipt filtering | conflict resolved as union |
| Legacy GRPO safety guards, metric denylist, `_override_`-aware config minimizer, sleep-level staleness, multi-EOS boundaries, non-SP fused CE fix | `grpo.py`, `logger.py`, `tools/config_cli.py`, vLLM workers, `distributed/model_utils.py` | auto-merged; `config_cli.py` keeps the fork's guard and drops upstream's duplicate hunk |
| CSCS build/env/launch (hermetic image, builder/assembly split, receipts, fingerprint and inventory gates, profiles) | `docker/Dockerfile` (fork side kept for all five conflict blocks), `actor_environments.py --profile`, `tools/image_*`, `infra/slurm/cscs/*` | fork blocks kept; Gym venv override version follows Gym `fd5e84d6` (0.25.1) |
| DPO train status | `DPOTrainStatus` return kept; upstream's `stop_after_step` returns `COMPLETED`; upstream `metrics_cls`/`post_train_step` kwargs adopted | `examples/run_dpo.py` still keys on `TIMED_OUT` |
| GRPO group baseline over every rollout | `_advantage_stage` calls the estimator without `valid_mask`; rows the environment, overlong or logprob-error gates exclude from the loss (upstream #3766/#3671, in the common base) still shape their group's baseline, and the `reward` metric stays the unweighted mean | upstream #3837 now passes `valid_mask=final_sample_mask` (masked rollouts leave the baseline, advantages zeroed). Not adopted in this sync: the certified 70B/GLM runs trained with the older rule, and switching mid-line would confound their comparisons. One-line flip, pinned by `test_alp_advantage_stage_preserves_raw_rewards_and_uses_occurrence_groups` and the replay-invariance test. Token-capture placeholder rows (the #3837 motivation) do not occur while token capture is off. |

## Upstream-covered or replaced fork code (dropped)

| Fork code | Replaced by | Why |
|---|---|---|
| `nemo_rl/modelopt/registry.py`, `_EXECUTABLES_BY_EXTRAS`, `VLLM_CONTROLLER_ACTORS` exception | upstream #4020: `PY_EXECUTABLES._resolve_system_overrides()`, `uv_py_executable(extras)`, `MODELOPT_*` constants | one place knows `NEMO_RL_PY_EXECUTABLES_SYSTEM`; the controllers-are-not-overridden exception was a base-era artifact upstream removed on purpose |
| `train_microbatch` multimodal `NotImplementedError` guard and its test | upstream #4009: `_train_microbatch_body` attaches the media-token validity mask and model-owned packing/CP flags | guard existed because the split path lacked the mask; upstream closed the gap |
| `prefetch_venvs --negative-filters` and in-function `RuntimeError` | upstream #4002: explicit actor FQNs and a returned failed list with `SystemExit(1)` in `__main__` | no fork caller used negative filters (Dockerfile and overlay pass FQNs / profiles); the `--prebuilt` finalizer stays |
| Gym as workspace member with package-qualified `conflicts` | upstream #4050: Gym as an editable path dependency | same effect, smaller lock (drops the Gym-extra vllm 0.24.0 tree) |
| `venvs.py` conditional `--inexact` base sync | upstream unconditional `uv sync --inexact` | equivalent; fork keeps the pinned `uv` resolution and `--exact` actor sync |
| `_prune_equal` `_override_` guard (fork copy) | upstream #3917 added the same guard | duplicate hunk dropped; fork tests remain |
| Bridge `542cab9df` (GLM-5 FP8 import), `ee1081c8b` (MTP MoE metric layers); Core `3eddc9786` (NCCL on idle ranks) | upstream `44c871b4`, `8eea7a95`, `af6d4a98` | exact patch-id duplicates; content identical in both parents |
| 75 historical NeMo patch replays (Sep 6 audit) | upstream originals in common ancestry | no conflicts; nothing to do |

## Adopted upstream changes with local impact

- vLLM workers now declare `["vllm", "nemo_gym"]` (upstream #4009/#4020): token capture imports `nemo_gym` inside the worker. The Apertus image profile still builds six workers, two of them now carrying the Gym package. Token capture stays off in our recipes.
- New actor `nemo_rl.data.energon.sft_worker.SFTMegatronPolicyWorker` (`mcore`) is registered; it is in the `full` profile only.
- `mcore` extra gains the `megatron-energon[av-decode]~=7.3` floor and `av` is no longer excluded (upstream #3917). The lock already resolved energon 7.4.0; `av 18.1.0` is the one added package.
- Upstream features present but not enabled by any CSCS recipe: TQ token capture (#3837), rollout checkpointing (#3923/#3924), MOPD (#3978), worker extension FQNs (#3809), vLLM `reload_weights` refit (#3651; still rejects `nccl_reshard`), Mooncake GDR (#3501), HybridEP recipes (#3438), lora warm-start (#3874), colocated MInf (#3730), TQ scalar schema patch at import (#4024).

## Deferred (not in this sync)

- Dockerfile #4038 (`libnvidia-ml-dev` during the dependency build, `uv cache clean deep-ep`) for multi-node HybridEP: DeepEP/HybridEP are not qualified on Slingshot; deferred with the HybridEP feature itself.
- Rollout PP>1, the 12k ablation, and every exclusive feature listed in `session/20260910_git_cleanup/feature-audit.md` (prompt-group metrics, export safeguard, fixed-manifest evaluation, RoPE verification, profiling hooks, streamed chunk divisibility, thinking penalty, refit-every-N, MFU): extracted once, separately, after this base lands.
- Bridge local BF16 export iterator (#6005) and CPU/native load fixes (#5977/#5978) as replacements for the export safeguard: candidates, evaluated with that feature.

## Dependency identity

`uv.lock` regenerated locally with uv 0.11.28 in the sync worktree (the in-image
lock phase would resolve against the base image's old submodule sources).

| Layer | Change |
|---|---|
| Native compile inputs (torch 2.11.0+cu130, TE 2.18.0+27486e03, FlashMLA, mamba-ssm, causal-conv1d, grouped-gemm, DeepEP, vLLM 0.26.0 wheel, flashinfer 0.6.14, cutlass-dsl 4.6.0, sglang, modelopt, nccl-extensions, nixl, mooncake) | unchanged |
| Added | `av 18.1.0` |
| Removed (31) | the Gym `vllm` extra tree (daytona*, opensandbox, openshell, socketio, engineio, websocket-client, httpx-ws, simple-websocket, aiohttp-retry, tenacity, bidict, deprecated, obstore, librt, ast-serialize, toml), test tooling that only Gym's extra pulled (mypy, mypy-extensions, pytest-xdist, execnet, requests-mock, gprof2dot), opentelemetry-instrumentation(-aiohttp-client), opentelemetry-util-http |
| Changed (9) | vllm loses the 0.24.0 variant; flashinfer-cubin loses 0.6.12; humming-kernels and tokenspeed-mla drop old variants; mlflow/mlflow-skinny/mlflow-tracing, openai and sqlparse gain a second resolved version (Gym `fd5e84d6` pins) |

Because `pyproject.toml` and `uv.lock` bytes changed, the hermetic dependency
cache identity changes and the builder will treat the dependency layer as new;
uv's wheel cache still serves every unchanged native wheel. Full report:
`session/20260910_152226/LOCK_DELTA.md`.

## Known issue resolved in this sync

TransferQueue restored-schema warmup: after a checkpoint restore, a fresh
adapter re-warmed integer fields with float32 placeholders and the controller
logged `dtype mismatch: existing=torch.int64, incoming=torch.float32`.
`register_partition` now asks the controller which fields the partition's rows
already carry and warms only the rest (`_tracked_fields`). Regressions:
`tests/unit/data_plane/test_tq_lifecycle.py::test_register_partition_skips_fields_the_controller_already_tracks`
and `tests/unit/data_plane/test_restored_schema_warmup.py`.

## GLM-5.1 qualification baseline

The committed records (`session/20260903_integration/`) show probe 3278193 was
the certification test probe for `33c1e8f5f` on the vLLM 0.25.1 image, not a
10-step run. The last GLM-5.1 10-step run that completed all ten steps is job
3217663: `submit_glm51_sc_scale.sh` with the ready-first recipe (72 trainer +
64 rollout nodes, TP2/PP18/EP16, vLLM TP32, rollout PP1) on the 0.25.1 image.
The MTP3 variants never completed (step-9 OOM in 3219292) and the vLLM 0.26
matched retry was cancelled before model load (3297983). The sync candidate is
therefore qualified with the same ready-first launcher and recipe, only the
container image changes; it is the first GLM 10-step run on vLLM 0.26 and is
not a matched throughput comparison.

## Test adaptations made for the merged tree

Each row is a test-only change (or a one-line recipe fix) needed because an
upstream test now exercises a fork behavior, or a fork test asserted a shape
upstream changed. Production code was not weakened for any of them.

| Test | Change | Why |
|---|---|---|
| `tests/unit/data_plane/test_tq_policy_routes.py`, `test_multimodal_wire_roundtrip.py` | stub policies carry `cfg["make_sequence_length_divisible_by"]` | the fork's `_stamp_pad_seqlen` rounds the cross-DP pad target to the TP multiple; upstream stubs built `TQPolicy` without a config |
| `tests/unit/utils/test_prefetch_venvs.py` | upstream's failed-actor test runs with `NRL_VENV_PREFETCH_MAX_ATTEMPTS=1`; fork failure tests assert the returned list | fork retries venv builds; upstream's return contract adopted |
| `tests/unit/utils/test_venvs.py` | `test_non_uv_worker_keeps_exact_base_sync` retired | upstream's unconditional `--inexact` base sync adopted |
| `tests/unit/tools/test_docker_hybridep_build.py` | module skipped with reason | #4038 deferred (see Deferred) |
| `tests/unit/experience/test_rollout_generation_failures.py` | Gym impl fixture sets `_cot_token_ids = None` | fork's generation-quality metrics read the reasoning delimiters |
| `tests/unit/single_controller/test_setup.py` | expected `ClippedPGLossFn(..., opd_full=None)`; recovery fixture sets `grpo.calculate_advantages_on_gpu=false` | upstream MOPD kwarg; upstream's Gym recipe enables a flag the fork's SC validator rejects on purpose |
| `tests/unit/single_controller/test_single_controller_actor.py` | fork ALP/occurrence fixtures add `sample_masks` and a `DataPlaneCheckpointBarrier` | upstream logs `sample_masks` and enters the rollout-checkpoint barrier inside the advantage stage |
| `tests/unit/models/policy/test_megatron_split_state.py` | `test_public_split_path_rejects_multimodal_models` removed | fork guard superseded by upstream #4009 |
| `examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-probe-gym.yaml` | `split_validation_size: null` removed | `ResponseDatasetConfig.split_validation_size` is `NotRequired[float]`; `null` never validated |

Pre-existing on the build branch, unchanged by this sync: the 13 GLM autoresearch
and Apertus bench recipes have no test-suite driver (`test_all_recipe_yamls_accounted_for_in_test_suites`
was already failing at `c00cb93cf`), and the CSCS launcher recipes interpolate
`AP_*` / `GLM_*` from the environment, so config validation needs those variables
set the same way upstream needs `HF_HOME`; the CPU test job exports placeholders.

## Evidence

Filled in as steps complete; job IDs and digests only after the runs exist.

| Step | Status |
|---|---|
| Core merge | done, `c4df534e`; MLA import regression passes locally |
| Bridge merge | done, `3880d9e0`; Bridge unit/pre-commit checks pending (container) |
| NeMo-RL merge + relock | done: merge `7bae96802` on build tip `28f599e9f` (re-anchored after the CSCS layout reorganization), followed by `fix(data_plane)`, `fix(sc)`, `test(sync)`, `docs(sync)`, `test(infra)` |
| Static checks | ruff check/format clean on every changed Python file |
| CPU unit tests | job 3350447 (venv from the new lock inside the qualified 0.26 image): data plane 290 passed / 14 skipped, build 326 passed, controller 2862 passed / 1 failed (pre-existing recipe accounting); TQ regression red against the merge commit, green with the fix |
| Pins published | Core `c4df534e` and Bridge `3880d9e0` on `integrate/2026-09-10-upstream-sync` in both forks (ls-remote verified); NeMo branch pushed |
| Image build / assembly / native gates | dependency rebuild job 3350563 submitted from `6b8d6d1cf`; release build, assembly and gates pending |
| 70B initial + resume | pending |
| GLM-5.1 10-step probe | pending |
