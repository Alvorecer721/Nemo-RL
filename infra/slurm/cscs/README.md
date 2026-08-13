# CSCS Slurm Probes

This directory contains the Clariden/GH200 Slurm wrappers used to build, probe, and train with the NeMo-RL `nvcr.io/nvidia/nemo-rl:v0.7.0` container on Slingshot.

The default container environment is `docker/nemo_rl.toml` in this checkout. The wrappers set `CUDA_CACHE_PATH` and Hugging Face cache paths in shell code because TOML values are not shell-expanded by Pyxis/EDF.
Its AWS OFI hook and NCCL/libfabric values follow the current
[CSCS NCCL guidance](https://docs.cscs.ch/software/communication/nccl/): the
portable `cuda-dl` hook, GPU Direct RDMA through `PHB`, and
`FI_CXI_DEFAULT_TX_SIZE=16384`.

The EDF points at the shared pre-pulled image in `MLLM/containers/`, so no user ever waits on an NGC pull. Measured cost of serving it from capstor instead of a per-user iopsstor imagestore (fresh nodes, same sqsh, paired runs sharing a page-cache pre-warm — so the values are comparative, not true cold-start absolutes): sequential read 2.2 vs 7.7 GB/s, container mount 11.8 vs 7.6 s, `import torch` 2.6 vs 2.2 s — about five seconds per job start, which the zero-pull onboarding more than repays. A paired same-node A/B additionally measured ~1.5–2% slower steady-state generation on the capstor-served image (~0.5% of end-to-end step time) — noted for honesty, not action. Point `image =` back at `nvcr.io#nvidia/nemo-rl:v0.7.0` if you prefer paging from your own imagestore.

## Submit from a login node (humans)

Submit these from a **login node** (e.g. `clariden-ln001`).
The wrappers follow CSCS guidance — `--environment` lives on each `srun`, never on `#SBATCH` — so the batch script and the Slurm client run on the **host**, which has the system libraries the Slingshot/pyxis plugins need (`libjson-c.so.5`); only the task is containerized.
A login node has those libraries natively, so nothing extra is required.

Run from the repository root after creating the log directory:

```bash
mkdir -p logs
sbatch infra/slurm/cscs/build_xielu.slurm                       # one-time CUDA xIELU kernel build
sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm                # 3-step online GRPO smoke (sync, colocated)
sbatch infra/slurm/cscs/probe_grpo_async.slurm                  # same gate on the async non-colocated path
sbatch infra/slurm/cscs/probe_nemo_rl_dpo_megatron_apertus.slurm  # DPO probe + provider gate
sbatch infra/slurm/cscs/submit_nemo_rl_dpo.slurm                # full MaxMin DPO run
```

Useful overrides:

```bash
CONTAINER_ENV=$HOME/.edf/nemo_rl.toml sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm
RECIPE=examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml sbatch infra/slurm/cscs/submit_nemo_rl_dpo.slurm
```

## Submit from inside a compute node (coding agents)

This subsection is specifically for a **coding agent** (e.g. Claude Code / VS Code) running **inside a compute-node Container Engine (CE) session** — it cannot reach a login node, and its CE container ships only `libjson-c.so.3`.
Two things break as a result:

- `sbatch`/`srun` can't load the Slingshot/pyxis plugins (`libjson-c.so.5: cannot open shared object file`) — the plugins are loaded by the *client*, which here runs inside the lib-less container.
- the CE session's inherited pyxis `--environment` collides with the launcher's per-`srun` one (`--environment specified multiple times`).

To submit from inside a compute node, feed the client the shared `libjson-c.so.5` from the wheelhouse and unset the inherited pyxis options:

```bash
env -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts \
    LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:$LD_LIBRARY_PATH \
    sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm
```

This path is **only** for agents/automation that must submit from a compute node; humans should submit from a login node (above), where none of this is needed.
See <https://docs.cscs.ch/software/container-engine/run/>.

**One containerized step per allocation at a time.** Launching a second
`srun --environment` step from the same EDF while another is live on the node
has externally SIGTERMed the first step's healthy processes (observed with a
GRPO probe killed seconds after a sibling test step's container came up; RAM
pressure from the sibling's venv build into tmpfs is the alternate suspect).
Serialize containerized steps — or give a concurrent step a different EDF —
until the mechanism is pinned down.

## Custom vLLM 0.25.1 GH200 image

The machine-local `docker/nemo_rl_vllm0251.toml` EDF selects the custom arm64
image built from this checkout. It runs the baked `/opt/nemo-rl` tree and
frozen environments under `/opt/ray_venvs`; it does not require checkout-local
`.venv` or `venvs` directories.

The current SquashFS is a clean, standalone build from `cef7e2ddb9d4`. It
contains the `packed_broadcast` stream joins, RayExecutorV2 TCPStore and
MessageQueue port patches, FP8 in-place refit fix, and dependency-aware frozen
environment markers. It supersedes the earlier
`336136c10490-dirty-fd360335e307` artifact, which required a checkout overlay
and must not be used for multi-node startup or refit certification.

The EDF is intentionally ignored because its
`image` field points at a user- or project-specific SquashFS path. Create it by
copying `docker/nemo_rl.toml` and replacing only the `image` value with the
builder's reported `BUILD COMPLETE` path.

| Field | Value |
| --- | --- |
| SquashFS | `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-cef7e2ddb9d4-f62a9972872b.sqsh` |
| Size | 48,752,754,688 bytes (about 45.4 GiB) |
| SHA-256 | `3d36df6b3ec654bf3691f29fd33f3759c09a0a2d437cfcca773ff234cbc8a592` |
| OCI tag | `nemo-rl-apertus:vllm-0.25.1-cef7e2ddb9d4-f62a9972872b` |
| OCI image ID | `824dff64d3d2bae6310fef29d8b24864544b35719875aa51cee9196a4616733e` |
| Persistent OCI data | `/iopsstor/scratch/cscs/xyixuan/podman-cache/nemo-rl` |

The image ID suffix is `<12-char-git-revision>-<12-char-build-input-hash>`.
Release builds reject dirty repositories and dirty recursive submodules. The
input hash records the full source commit, recursive submodule SHAs, the pinned
base-image manifest, and platform build arguments. The launcher copies only
Git-tracked files while preserving metadata needed for Podman's persistent
layer-cache identity, so ignored files and mutable `.git` metadata cannot
enter it.

### Build and cache lifecycle

Submit the builder from the repository root. The launcher already names the
current Apertus reservation; override the `#SBATCH` option at submission time
when the reservation changes.

```bash
mkdir -p logs
sbatch --chdir="$PWD" infra/slurm/cscs/build_nemo_rl_image.slurm
```

Useful environment overrides are `SCRATCH_ROOT`, `CACHE_DIR`, `OUTPUT_DIR`,
`REGISTRY_IMAGE_ARCHIVE`, `BASE_IMAGE`, `MAX_JOBS`, `NVTE_CUDA_ARCHS`, and
`PODMAN_STORAGE_BASE`. `HERMETIC_CACHE_TAG=rebuild` deliberately rebuilds all
dependencies instead of resuming the pinned hermetic image. The default
architecture is arm64/GH200, the NGC base is digest-pinned, and Transformer
Engine is limited to `NVTE_CUDA_ARCHS=90`.

The build has two different stores:

- `PODMAN_STORAGE_BASE` is allocation-local overlay storage. It disappears
  with the compute node and is only working space for Podman commits.
- `CACHE_DIR` is a Lustre-backed registry data directory. The registry process
  and `127.0.0.1:5000` endpoint exist only during the build, but its blobs,
  cached layers, and the final manifest survive node changes. The launcher
  takes an automatic `flock` before writing this shared cache. The `.sqsh` in
  `OUTPUT_DIR` is the delivered Container Engine image and also survives the
  allocation.

The launcher cleans interrupted Buildah containers before building, restores a
pinned `registry:3` bootstrap image, waits for registry readiness, pushes the
final OCI manifest, exports SquashFS, verifies its superblock, and checks the
vLLM renderer/tokenizer/tool-parser import boundary.

For a source-only release, the launcher verifies the current dependency
fingerprint and resumes from the content-addressed final hermetic cache image.
This is intentional: Podman models a named source context as a parent image,
so any source edit otherwise invalidates the earlier dependency COPY layers.
When dependencies change, run with `HERMETIC_CACHE_TAG=rebuild`, then replace
the pinned tag and fingerprint with the new completed hermetic-stage values.

### Failure and recovery ledger

Thirteen substantive build/recovery attempts were made. Three additional
18-second Slurm jobs failed during launcher bootstrap and did not begin image
construction.

| Attempt | Failure or action | Durable fix/result |
| --- | --- | --- |
| 1 | `/dev/shm` exhausted while committing several Megatron Ray environments | Bound large frozen environments to separate layers. |
| 2 | Podman overlay commit reported `lgetxattr ... no such file` | Moved to an explicit clean allocation-local Podman store; more memory was not the fix. |
| 3 | The 12-hour allocation expired during the long build | Persisted completed cache layers in the Lustre-backed local registry. |
| 4 | One layer containing five DTensor environments exceeded commit-time temporary space | Split every DTensor and remaining Ray actor into its own layer. |
| 5 | Transformer Engine tried to compile unsupported `compute_120` code on GH200 | Added `NVTE_CUDA_ARCHS` and set the CSCS default to `90`. |
| 6 | Apptainer cleanup failed on gocryptfs pathological long-name fixtures | Excluded only its test fixtures and left the build directory before cleanup. |
| 7 | Registry blobs existed, but a reconstructed `COPY --link` parent missed the descendant Podman cache identity | Stopped the wasteful traversal and recovered from the allocation-local step-39 parent. |
| 8 | Recovery reached NemoGym but ran out of space because about 70 interrupted Buildah containers retained overlays | Added `podman system migrate`, external prune, and build-container prune before each build. |
| 9 | Cleaned transient state, finalized the image, pushed its manifest, and exported SquashFS | Successful artifact listed above. |
| 10 | A clean hermetic resume reached the release vLLM/SGLang environments, but direct GitHub wheel fetches failed with transient HTTP/2 refused streams; the prefetch summary also returned success with failed workers | Retry each worker environment, propagate failures out of prefetch, and pass completed paths directly to wrapper generation. |
| 11 | Retries recovered the vLLM and SGLang environments and crossed four split DTensor layers, but the outer `uv run` auto-synced the already-complete main environment and failed before the Python retry loop | Invoke release helpers with `uv run --no-sync`; their inner worker `uv sync` remains explicit, retried, and fail-closed. |
| 12 | The first no-sync build invoked the helper by file path, so Python started in `nemo_rl/utils` and could not import `nemo_rl` | Invoke it as `python -m nemo_rl.utils.prefetch_venvs`, preserving the project import root without an outer sync. |
| 13 | The module-based build recovered two transient direct-wheel failures, committed every worker separately, pushed OCI image `824dff64d3d2`, exported SquashFS, and passed the baked import check | Successful standalone artifact listed above; job `3069254`. |

Historical build output is under `logs/nrl-vllm0251-image_*.{out,err}`. Direct
step-39 recovery was an allocation-specific rescue, not the supported rebuild
path; use `build_nemo_rl_image.slurm` for future builds.

A later attempt to reproduce the image from a committed tree added three more
launcher fixes:

- Batch steps have no logind session, so rootless Podman found no
  `/run/user/<uid>` and died on `pause.pid`. The launcher now provides
  `XDG_RUNTIME_DIR` itself, which is why earlier builds only ever worked from
  interactive allocations.
- Cache ownership now uses a process-held `flock`, so Slurm releases it even
  when an interrupted build cannot run cleanup.
- Podman lives on the compute node's host, not inside the Container Engine
  session, so a build cannot be driven from inside a CE container; run it via
  `sbatch`, or an `srun` step with no `--environment`.

Plain batch jobs have no logind session. The registry therefore uses host
networking and binds `127.0.0.1:5000` directly; it does not use rootless port
publishing, which was the remaining session-bus dependency. A missing session
bus may still produce a harmless cgroup warning, but no longer blocks registry
readiness.

The successful build log is `logs/nrl-vllm0251-image_3069254.out`. It records
all 47 release steps, final OCI publication, SquashFS export and checksum, and
the baked vLLM 0.25.1 renderer, tokenizer, and tool-parser import check.

### Image validation

These probes use `/opt/nemo-rl` plus the frozen worker environments, so a
passing run demonstrates the delivered image rather than checkout overlays.
The DPO probe's `PYTHONPATH` contains only the shared CUDA xIELU site; the vLLM
probe leaves it blank.

```bash
sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_dpo_vllm0251_image.slurm

sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_vllm0251_image.slurm

sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_grpo_vllm0251_image.slurm

sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_vllm0251_multinode_image.slurm
```

The builder's CPU-side standalone boundary check passed on job `3069254`:
the exported image reports vLLM `0.25.1`, OpenAI `2.6.1`, and xgrammar
`0.2.3`, and imports the renderer, tokenizer, and Apertus tool parser from its
baked tree. Run the four GPU probes above after changing the artifact; they
cover real-model generation and repeated FP8 storage-preserving refit, a
four-GH200 DPO step with CUDA xIELU, a four-GH200 GRPO refit/KL step, and a
two-node TP8 vLLM startup respectively.

The two-node probe carries a CSCS-specific vLLM 0.25.1 collective workaround:
it disables vLLM's direct PyNccl and Hopper symmetric-memory backends for the
node-spanning TP8 group, leaving collectives to PyTorch's NCCL process group
over the AWS OFI plugin. A standalone eight-rank PyTorch all-reduce and TP8
generation both pass on the same two-node Slingshot topology. Job `3072893`
also passed with NCCL's default lazy connection behavior, so
`NCCL_RUNTIME_CONNECT` is deliberately not overridden.

This workaround is intentionally scoped to the CSCS node-spanning launcher.
The generic `VllmGeneration` path remains unchanged: disabling these backends
globally would also remove optimized paths from single-node and other supported
platforms. Revisit the generic path only with a topology-aware upstream guard.

For historical comparison, the checkout-overlay validation on allocation
`3061315` on 2026-08-12 injected the ABI-matched
`xielu-site-current` into the frozen Megatron worker environment and passed a
CUDA xIELU forward/backward preflight. It then loaded the real Apertus 1.5 8B
policy and reference checkpoint on four GH200s, trained one MaxMin batch,
recorded loss `0.693147`, and passed its metric assertion without reporting the
eager xIELU fallback. The step took 4.96 seconds. Evidence is in
`logs/nrl_vllm0251_dpo_xielu_3061315_retry/`.

The vLLM probe then initialized vLLM 0.25.1 with the real 17.16 GiB Apertus
checkpoint on one GH200, generated 24 tokens from a chat-formatted prompt, and
validated the registered Apertus parser with a structured `get_weather` call.
Both `model_generation=OK` and `apertus_tool_parser=OK` were asserted before
exit 0. Evidence is in `logs/nrl_vllm0251_smoke_3061315_retry2/run.log`.
The first two model-probe invocations were harness corrections: one lacked the
Python multiprocessing main guard, and one asserted non-empty decoded text
after a raw prompt legitimately emitted an immediate stop token. Both reached
the expected image code; neither required an image change.

The model probe uses eager execution to isolate loading and generation from
`torch.compile`. The image does not bake the optional CUDA xIELU extension:
the DPO launcher injects it only into Megatron training, while vLLM generation
keeps an explicitly blank `PYTHONPATH` and uses its Python xIELU fallback. This
split is intentional; see `docs/apertus-xielu.md`.

## Troubleshooting first launches

To skip first-launch building entirely, pre-provision the checkout once with
`sbatch --chdir=<repo> infra/slurm/cscs/prepare_env.slurm` — it builds `.venv` plus the worker
venvs behind an flock (safe to rerun; concurrent submissions queue instead of deadlocking), and
every later training job starts hot.

- **Two cold-cache jobs deadlock on a source build** (`Failed to acquire lock on the distribution
  cache`, 300 s timeout): run the first launch alone (or use `prepare_env.slurm` above); every
  later job reuses the built wheels.
- **After editing `pyproject.toml` or `uv.lock`, settle the shared `.venv` before parallel
  submissions**: the next `uv run` re-syncs it, and simultaneous jobs racing that re-sync can
  degrade into source rebuilds and uv lock timeouts (two probes died this way). One solo job —
  or `prepare_env.slurm` — settles it; steady-state parallel submissions are unaffected.
- **Run the unit suite in its own venv, never the shared `.venv`**: the suite needs the
  `test` group, so `uv run` re-syncs `.venv` to a shape no training job wants, and a job
  starting alongside re-syncs it back — a suite run died mid-collection this way when a probe
  started. Use `UV_PROJECT_ENVIRONMENT=$PWD/.venv-test uv run --locked --extra mcore --group
  test bash tests/run_unit.sh unit/` (`.venv-test` is gitignored). `mcore` and `vllm` are
  declared conflicting extras, so ask for one.
- **Worker venvs rebuild automatically when their inputs change**: the readiness marker records
  `uv.lock`, `pyproject.toml`, and the normalized worker command that selects its extras. A later
  job repairs the environment in place when any of those inputs differ. This prevents a lock
  bump such as vLLM 0.20→0.25, or an actor moving between extras, from silently reusing the old
  packages. Let one job settle shared venvs before starting concurrent jobs across such a change.
- **`Pretrained run config not found ... iter_0000000/run_config.yaml` on rank>0**: a stale,
  half-written conversion cache. Delete the `$HF_HOME/nemo_rl/model__*` directory for that
  checkpoint and rerun — rank 0 reconverts cleanly.
- **Worker import errors like `module 'torch' has no attribute 'Tensor'` right after venv
  creation**: a crashed builder left a partial worker venv. Since the readiness-marker fix a
  venv without `NEMO_RL_VENV_READY` is repaired in place on the next run and a stale
  `STARTED_ENV_BUILDER` claim expires after `NRL_VENV_BUILD_TIMEOUT_SECS` (default 3600 s), so
  plain resubmission heals it; `rm -rf <repo>/venvs/<worker-name>` remains the manual override.
- **A personal `uv` in your dotfiles shadows the image's** (`/root/.local/bin/uv`) and its cache
  keys may miss the image-seeded wheel cache, causing one-time source rebuilds. Since the vLLM
  0.25 bump this is no longer just slow: `pyproject.toml` uses `[tool.uv]` fields (e.g.
  `exclude-dependencies`) that uv < 0.11 rejects, so an old personal uv hard-fails worker-venv
  builds. The launchers export `UV=/root/.local/bin/uv` and `venvs.py` honors it; keep that pin
  intact if you write a new launcher.
- **Never submit with `sbatch --export=ALL`**: it leaks the interactive session's environment
  (including a different `uv`) into the job and breaks dependency resolution. The launchers use
  `--export=NONE`; pass parameters the way `probe_grpo_async.slurm` does (a wrapper that exports
  them in the job body).
