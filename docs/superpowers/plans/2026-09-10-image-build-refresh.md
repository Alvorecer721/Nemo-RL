# Image build refresh implementation plan

Approved by the user's "fire" following the Docker audit. Execute in the isolated
`build/2026-09-10-image-refresh` branch. Preserve the current dependency pins,
including TE 2.18, vLLM 0.26, Bridge and MCore. No main-branch merge is requested.

## Design and constraints

Use upstream #4002's dependency-free actor manifest and two-phase hardlink
installation, adapted to our current workers and readiness markers. The full
image remains available; the CSCS builder selects an `apertus` profile for
Megatron training, vLLM/Megatron generation, and their controller/reward actors.
Keep backend environments separate. Create dependency venvs in the cache-owning
layer; final assembly adds source and checks existing installations offline.

Use a generated JSON build manifest rather than hand-maintained cache hashes.
Its digest includes dependencies, recursive submodule pins, actor selection,
dependency-stage Docker instructions, compiler choices and build helpers.
Dependency changes must reject stale caches. A source-only edit must retain the
dependency cache. Do not bypass runtime fingerprints or change platform support.

## Task 1: Actor environment manifest

- [x] Add `nemo_rl/distributed/actor_environments.py`, usable as a stdlib-only
  script, and derive the runtime registry from the same mappings.
- [x] Support `--profile full|apertus` plus upstream backend skip flags. Output
  TSV records `actor_fqn<TAB>stage<TAB>uv arguments`; reject invalid selections.
  `stage` is `deps` or `trtllm`. Keep system actors in the registry but omit them
  from venv build rows. Apertus retains both vLLM workers, MegatronPolicyWorker,
  AsyncTrajectoryCollector, ReplayBuffer and SyncRolloutActor.
- [x] Hash the manifest in `tools/generate_fingerprint.py`. Preserve local
  readiness markers, frozen wrappers, exact environment selection and retries.
- [x] Test manifest/runtime parity, profile coverage, skip-by-extras behavior,
  invalid input and fingerprint invalidation with stdlib/module-isolated tests.

## Task 2: Build cache and builder

- [x] Add `tools/image_build_manifest.py` and tests for deterministic identities,
  build-input changes and application-source stability. Write JSON atomically.
- [x] Replace hardcoded cache tags and MD5s in
  `infra/slurm/cscs/build_nemo_rl_image.slurm` with the generated manifest.
  Support `HERMETIC_CACHE_TAG=auto|rebuild|<digest>`, preserving the two-allocation
  flow and durable local registry. Verify the embedded manifest on cache reuse.
- [x] Pass `NRL_IMAGE_PROFILE=apertus` by default in the CSCS builder, together
  with the existing TE, architecture, base-image and optional-backend settings.
- [x] Record build/assembly/export elapsed time and storage availability. Keep
  compiler concurrency bounded and all existing source cleanliness gates.
- [x] Validate shell syntax and fixture-based cache selection tests.

## Task 3: Docker layering, overlays and qualification

- [x] Adapt dependency warmup to materialize selected worker dependencies using
  hardlinks in the owning layer, with the actor list generated once and checked.
- [x] Finish source installation using that same actor list; preserve frozen
  wrapper/readiness behavior and make final worker checks network-independent.
- [x] Reconcile `Dockerfile.overlay` with the manifest instead of removed
  negative-filter CLI. Preserve its refusal to change inherited submodule pins.
- [x] Add a worker qualification helper that verifies selected interpreter
  imports, versions, fingerprint and absence of missing shared libraries;
  generate a machine-readable qualification report.
- [x] Run focused existing and new tests, shell/static checks and an independent
  whole-branch review. Commit with DCO after checks pass.
- [x] Launch the dependency image build, then release assembly; record job IDs,
  actual timings and failures. Qualify the image before any broad training run.
  Reuse existing distributed refit and optimizer-resume qualification scripts
  with the new image where compatible, without modifying active experiments.

## Progress ledger

- Initial audit complete; baseline is origin/main 6196eabe33686a494268545c54040bbd842098cf.
- Ruling: this implements the approved Docker scope with existing dependency
  pins; the broader upstream/Bridge synchronization is a separate integration.

- Implemented the shared actor table, six-worker CSCS profile, hardlink dependency
  population, offline release assembly, generated cache identity and automatic
  cache selection, private graph reset, stage timings, and strict worker checks.
- Review found and corrected profile-independent TRT compilation and permissive
  worker qualification. CPU qualification explicitly records deferred GPU checks;
  `--require-gpu` verifies native imports and CUDA execution after image assembly.
- Prepared bounded 70B qualification from the existing GSM8K 2k recipe: 12 trainer
  nodes (TP2/PP4), four rollout nodes (TP4/PP1), two initial updates, then two
  resumed updates with optimizer state. Run-specific launch artifacts stay in
  the ignored `.tmp/image-build-refresh` directory; they do not change the recipe.
- Build 3346508 completed dependency compilation and all six worker warmups,
  including TE 2.18, but Podman exited 125 while committing that layer:
  `lgetxattr ... networkx/algorithms/bipartite/tests/__init__.py: no such file or directory`.
  Build execution took 1,509 seconds; the allocation took 1,884 seconds.
- The inspected host provides fuse-overlayfs 1.1.0. A seven-instruction,
  network-free hardlink/removal reproduction passed with both that helper and
  the official 1.18 ARM64 helper, so it did not reproduce or establish the exact
  root cause. The next full build pins 1.18 in the allocation-private directory
  and verifies its release SHA256 before execution. This is a mitigation under
  validation, not a confirmed fix. No host packages or runtime dependencies change.
- The actual helper setup was executed on the host: Podman reported the pinned
  mount program, and a corrupted-download test stopped at checksum validation.
  Shell syntax checks and an independent review passed. Upstream release notes
  document directory iteration, lookup, and deleted-file fixes after 1.1.0:
  https://github.com/containers/fuse-overlayfs/blob/v1.18/NEWS.
- Build 3346566 passed and published the hermetic dependency image in 3,511
  allocation seconds. Its generated cache key remains
  `c942b5ec4023a984b48ffe867687d2576ca78b5b7deb9b972eef0e4868be0649`.
- Release 3346567 failed after 2,130 allocation seconds at worker finalization.
  The general venv provisioner first selects base dependencies, then the actor
  extra, switching tilelang/protobuf versions twice. Its cross-layer hardlink
  replacements failed with EINVAL. This was reproduced with one actor using
  the release's cached parent image and no network access.
- The corrected image-only finalizer runs the actor's frozen offline sync
  directly, with copy mode for the release delta. All six actors passed against
  the cached parent: CUDA wheels were reused, and only Bridge/Core's small
  Python packages rebuilt (15.78 seconds). Before/after dependency versions
  matched; all wrappers, readiness markers and inventory checks passed.
- Qualification compares a package inventory recorded after successful frozen
  synchronization, alongside the existing source fingerprint, readiness and
  native-import gates. The prior `uv sync --check` incorrectly required installer
  convergence despite upstream wheel-tag, local-version and freshness quirks.
  Inventories reject missing metadata or duplicate distributions, record
  METADATA/WHEEL/RECORD/direct URL hashes, Python/prefix/extras, and are readable
  by non-root runtime users. No dependency pins or static metadata are changed.
- The stale causal-conv1d, Mamba and fast-hadamard static version hints were
  observed during diagnosis; reconciling them belongs in a separate dependency
  change. The custom lock validator explored during diagnosis was discarded.
- Finalization validation passed: 89 focused tests and 24 subtests, Ruff checks,
  all six real worker environments, and independent review. Builder fixtures now
  exercise the pinned helper checksum, including rejection of corrupt downloads.
  Full export and bounded 70B initial/resume qualification subsequently passed;
  see the final qualification record below for its scope and remaining warning.

## Final qualification, September 10

- Prepared image source: `2a044e94d9290997e33066994e64f22c5af60c03`;
  immutable image digest:
  `sha256:5825208d5b7ddd7939ad6018c14f0869646301eeb6807ef06152a51b888c2abd`.
- Independent assembly job 3348587 completed in 420 script seconds (430 seconds
  allocated), including a 221-second pull, 78-second export and 10-second full
  SquashFS data check. It reused the prepared image without package installation
  or compilation. These are assembly measurements, not a full-build speedup.
- Native worker qualification job 3348656 passed all six selected GPU worker
  environments, including vLLM 0.26's stable-libtorch native API and a CUDA
  RMSNorm parity check. Ray startup now propagates setup failures on every node.
- Apertus 70B GSM8K, thinking off, 2k total context: 12 trainer nodes at TP2/PP4
  plus four rollout nodes at TP4/PP1; 48 prompts times 16 responses, GBS 768.
  Job 3348662 completed two updates in 18m42s; job 3348663 loaded step 2 and
  completed updates 3 and 4 in 13m00s. Both exited 0:0. All 16 nodes passed GPU
  worker qualification in each run. Both saved 48 nonempty DCP shards with
  optimizer state enabled. Loss, gradient norm and generation KL remained finite.
- This establishes distributed refit, update execution, checkpoint artifacts and
  resume progression. It does not establish a throughput gain or a direct
  comparison of every Adam state tensor/counter across the restart.
- Resume logged one nonfatal TransferQueue schema error:
  `existing=torch.int64, incoming=torch.float32`. The restore path registers
  float32 warmup placeholders against restored fields; that code is unchanged by
  this build work. Training continued and saved step 4. A focused restored-schema
  regression remains necessary before claiming error-free replay restoration.
- Local evidence: `.tmp/image-assembly-split/full-validation/` in the
  `image-assembly-split` worktree contains receipts, per-node reports, initial and
  resume qualification JSON, run logs and `resume-log-review.md`.
- TE 2.18, vLLM 0.26 and all submodule pins remain the existing foundation pins.
  Full upstream synchronization is a separate change requiring a new image.
