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

Useful overrides below pass an explicit export list, overriding the scripts'
`#SBATCH --export=NONE` directive without importing the rest of the interactive
environment. Do not combine `NONE` with named variables; Slurm rejects that
form.

```bash
sbatch --export=CONTAINER_ENV=$HOME/.edf/nemo_rl.toml infra/slurm/cscs/probe_grpo_fixgate.slurm
sbatch --export=RECIPE=$PWD/examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml infra/slurm/cscs/submit_nemo_rl_dpo.slurm
```

The DPO launcher chains itself across the 12 h wall-clock limit: at
`CHECKPOINT_MUST_SAVE_BY` (default 11 h 30 m) the run saves a checkpoint and
exits `75` (`EX_TEMPFAIL`), and the launcher queues exactly one
`--dependency=singleton` successor that resumes from that checkpoint in a fresh
window; a run that finishes exits `0` and queues nothing, so the chain
terminates by itself. In the queue this looks like one `ap1p5-train` job at a
time, reappearing until the run completes. Set `AUTO_REQUEUE=false` to disable
the chaining, or `CHECKPOINT_MUST_SAVE_BY` to move the save deadline. A run
that completes exactly at the deadline exits `0`, not `75` — completion wins.

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

The current SquashFS is a clean, standalone build from `7c68228e4f09`. It
contains the `packed_broadcast` stream joins, RayExecutorV2 TCPStore and
MessageQueue port patches, the post-v0.25.1 invalid-MNNVL-workspace fix, FP8
in-place refit fix, and dependency-aware frozen environment markers. It
supersedes the earlier
`336136c10490-dirty-fd360335e307` artifact, which required a checkout overlay
and must not be used for multi-node startup or refit certification.

The EDF ships with the checkout and points at the shared certified copy below
(no build needed, readable by all of `infra01`). To run your own build instead,
replace only its `image` value with the builder's reported `BUILD COMPLETE`
path — and leave the checked-in default on the shared copy.

| Field | Value |
| --- | --- |
| Shared SquashFS | `/capstor/store/cscs/swissai/infra01/MLLM/containers/nemo-rl-apertus+v0.7.0+vllm-0.25.1.aarch64.sqsh` |
| Builder original | `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-7c68228e4f09-38c6b702948c.sqsh` |
| Size | 48,752,754,688 bytes (about 45.4 GiB) |
| SHA-256 | `d50f39e45f6104d13e12b9323dbe28cc91b0f13e3a250d029ce6cc7e7646742a` |
| OCI tag | `nemo-rl-apertus:vllm-0.25.1-7c68228e4f09-38c6b702948c` |
| OCI image ID | `18c36e6a31df01fc0370f65f9446c373d741198abf821b35a89be8214c11e79e` |
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

If the submitting shell is itself running inside a Container Engine or VS Code
session, clear inherited Pyxis options first. Otherwise `sbatch` can
containerize the batch script before it reaches the explicit
`srun --environment` command and change its working directory to
`/opt/nemo-rl`.

```bash
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts
```

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
When dependencies change, treat the rebuild as two allocations. First run with
`HERMETIC_CACHE_TAG=rebuild`. The launcher builds and publishes only the
hermetic target under its dependency fingerprint, prints the exact tag and
digests to pin, and exits successfully without entering release assembly.
Replace the pinned tag, fingerprint, and digests with those values, commit the
change, and start the release from a clean allocation-local Podman store. Do
not carry the dependency-stage graph into the release build on 334 GiB nodes:
rootless Podman briefly needs a second copy while committing a layer. Do not
move the overlay graph to Lustre either; that filesystem does not provide the
extended-attribute semantics Podman needs.

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
| 14 | A dependency-changing build published its hermetic cache but continued into release assembly with the dependency graph still occupying the 334 GiB Podman workspace; release step 34 exhausted `/tmp` | `HERMETIC_CACHE_TAG=rebuild` now targets and publishes only `hermetic`, prints the required cache pin, and exits before release; a fresh job performs release assembly. |

Historical build output is under `logs/nrl-vllm0251-image_*.{out,err}`. Direct
step-39 recovery was an allocation-specific rescue, not the supported rebuild
path; use `build_nemo_rl_image.slurm` for future builds.

The final clean rebuild on job `3077164` restored the persistent hermetic
manifest, committed every release actor as a bounded layer, exported the
SquashFS above, and passed the baked import check in 2:03:51. It crossed the
former five-DTensor temporary-space failure point with 235+ GiB of node-local
tmpfs still available.

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

The successful build log is `logs/nrl-vllm0251-image_3077164.out`. It records
all 47 release steps, final OCI publication, SquashFS export and checksum, and
the baked vLLM 0.25.1 renderer, tokenizer, and tool-parser import check.

### Image validation

These probes use `/opt/nemo-rl` plus the frozen worker environments, so a
passing run demonstrates the delivered image rather than checkout overlays.
The DPO probe's `PYTHONPATH` contains only the shared CUDA xIELU site; the vLLM
generation workers leave it blank.

```bash
sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_dpo_vllm0251_image.slurm

sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_vllm0251_image.slurm

sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_grpo_vllm0251_image.slurm

sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_grpo_async_vllm0251_image.slurm

sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_vllm0251_tp4_image.slurm

sbatch --reservation=SD-69241-apertus-1-5-0 --chdir="$PWD" \
  infra/slurm/cscs/probe_nemo_rl_vllm0251_multinode_image.slurm
```

The builder's CPU-side standalone boundary check passed on job `3077164`:
the exported image reports vLLM `0.25.1`, OpenAI `2.6.1`, and xgrammar
`0.2.3`, and imports the renderer, tokenizer, and Apertus tool parser from its
baked tree. Run the six GPU probes above after changing the artifact; they
cover real-model generation and repeated FP8 storage-preserving refit, a
four-GH200 DPO step with CUDA xIELU, synchronous and asynchronous four-GH200
GRPO refit/KL steps, a compiled single-node TP4 vLLM run, and a two-node TP8
vLLM startup respectively.

The GRPO probes' `RECIPE` defaults use the post-rename recipe names and
therefore match any image built from current `main`. The published
`7c68228e4f09` image predates the rename: to certify that specific artifact,
override the recipe to the old in-image name, e.g.
`RECIPE=/opt/nemo-rl/examples/configs/recipes/llm/probe-grpo-apertus1p5-8b-1n4g-megatron-async.yaml`
(and `.../probe-grpo-apertus1p5-8b-1n4g-megatron.yaml` for the sync probe).
Drop this note once the image is rebuilt from a post-rename revision.

The baked TP2 async probe passed on job `3077837` in 4:48. It ran one compiled
TP2 AsyncLLM engine on two GH200s and TP2 Megatron training on the other two,
rejected the invalid MNNVL workspace, captured CUDA graphs, completed two GRPO
steps with generation KL error `0.0000`, refit successfully, and used the CUDA
xIELU training kernel.

The baked single-node TP4 probe passed on job `3078831` in 3:27. It loaded the
real Apertus 1.5 8B checkpoint across all four GH200s with compilation enabled,
rejected the unusable MNNVL multicast workspace, initialized the compiled
TRT-LLM FlashInfer all-reduce backend, captured CUDA graphs, and generated four
64-token responses. This is the relevant full-node inference certification for
Clariden. The two-node TP8 probe is optional, non-blocking scalability evidence;
current Apertus 8B Async-GRPO uses one node with TP2 inference and TP2 training
and does not require TP8.

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

### Parallelism placement on quad-GH200 nodes

Upstream validates TP8 on 8-GPU nodes, where the whole TP group stays inside
NVLink. On Clariden's quad-GH200 nodes the same setting silently becomes
node-spanning: every per-layer all-reduce crosses Slingshot instead of NVLink,
which is both a throughput cliff and the reason the collective workaround above
exists. Upstream TP8 test results therefore do not transfer to this cluster.

Default placement here: keep TP inside the node (TP <= 4), scale generation
across nodes with additional engine replicas (data parallel), and cross nodes
with pipeline parallelism when a model outgrows the roughly 384 GB a TP4 node
provides (NeMo-RL requires the async engine whenever PP > 1). Node-spanning
TP8 is certified by the two-node probe but remains a last resort: it pays
Slingshot latency on every layer and depends on the scoped workaround. It is
not a release blocker for the current one-node Apertus 8B workloads; the
single-node TP4 probe is the maximum relevant tensor-parallel certification.

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
