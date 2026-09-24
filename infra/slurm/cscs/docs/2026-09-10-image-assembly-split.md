# Image assembly split implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement the approved tasks inline.

**Goal:** Assembly consumes an immutable prepared image and cannot install or compile dependencies.

**Architecture:** The existing builder stops after publishing a complete image and a validated receipt. A separate assembler verifies that receipt, pulls its digest, and exports SquashFS using shared host storage helpers. Native API and training checks remain separate.

**Tech Stack:** Bash, Python standard library, Podman, Enroot, Slurm.

**Spec:** `infra/slurm/cscs/docs/2026-09-10-image-assembly-split-spec.md`

## Global constraints

- Keep TE 2.18, vLLM 0.26, and all existing dependency inputs unchanged.
- Preserve the clean image-build-refresh checkout and its running qualification.
- Do not invoke package installation or Dockerfile processing from assembly.
- An assembly retry consumes the same registry digest and refuses mismatched metadata.

## Tasks

- [x] Add `infra/slurm/cscs/image_release_receipt.py` with a strict versioned receipt,
  atomic publication, immutable-reference validation, and image/manifest checks.
  Test missing receipt, mutable reference, wrong source/platform/fingerprint,
  corrupted dependency manifest, and refusal to overwrite a different receipt.
- [x] Extract the existing private Podman/registry/timing functions into
  `infra/slurm/cscs/image_storage.sh`. Retain checksum validation, private-store
  guard, registry lock and failure timing in both entry points.
- [x] End `build_nemo_rl_image.slurm` after a digest-producing image push and
  receipt publication. Keep all Dockerfile instructions on the build side.
  Adapt its service fixture so a successful build never invokes Enroot.
- [x] Add `assemble_nemo_rl_image.slurm`, requiring `RELEASE_RECEIPT`, and
  a standalone assembly body that validates metadata before export, uses an
  allocation-specific partial path and publishes only a valid SquashFS.
  Test that assembly never invokes build/install, rejects metadata drift and
  missing images, refuses overwrites, and leaves no final artifact on failure.
- [x] Move the existing native Enroot probes into
  `qualify_nemo_rl_image.sh`; document the exact build -> receipt -> assemble ->
  qualify commands and independent retry behavior.
- [x] Run focused tests, Ruff and shell syntax checks; request independent review.
  Validate assembly on CSCS using the already prepared image and its real digest.
  Commit with DCO after checks pass; retain qualification evidence and job IDs.

## Validation evidence

- Clean baseline: 19 build-manifest tests and 14 subtests passed.
- Final focused validation: 39 tests and 14 subtests passed; Ruff, shell syntax
  and whitespace checks passed.
- New behavior failed first: receipt absent, builder still exported, assembly
  entry point absent, and cleanup adapter absent. Focused checks then passed.
- Native CSCS test: job 3346281 step 24 built/published a small ARM64 fixture;
  step 28 assembled the exact digest from a fresh store. Export, full-data
  SquashFS validation and allocation all returned zero (assembly allocation 5 s).
- Native export exposed the inherited Enroot deleted-cwd cleanup failure. The
  same cleanup command failed with exit 1 from a deleted cwd and passed after
  `cd /`; the narrow adapter resolved it without ignoring export errors.
- Independent review found and resolved Slurm spooling paths, Enroot digest URI
  parsing, and partial-export acceptance. Final review: LGTM.
- Full production-image validation subsequently passed: assembly 3348587,
  native GPU workers 3348656, initial 70B updates 3348662, and resumed updates
  3348663. Source image `2a044e94d` was reused unchanged. The initial run saved
  step 2; the resume completed steps 3 and 4 and saved step 4.
- Assembly took 420 script seconds without compilation or dependency install.
  Both training jobs exited 0:0 with finite metrics and 48 nonempty DCP shards
  per saved checkpoint. This is bounded correctness qualification, not a
  throughput benchmark or complete optimizer-tensor parity test.
- One nonfatal TransferQueue restored-schema dtype warning occurred on resume;
  the relevant runtime path is unchanged by this branch. Exact evidence and
  scope are recorded in `2026-09-10-image-build-refresh.md` alongside this plan.
