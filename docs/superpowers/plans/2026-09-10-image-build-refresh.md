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
- [ ] Run focused existing and new tests, shell/static checks and an independent
  whole-branch review. Commit with DCO after checks pass.
- [ ] Launch the dependency image build, then release assembly; record job IDs,
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
