# Separate image build, assembly, and qualification

Approved in the conversation on 2026-09-10 after the release finalizer exposed
that the existing assembly job still installed packages.

Build owns every Dockerfile instruction, dependency resolution, compilation,
worker installation, and CPU image check. It publishes the complete OCI image
and then a JSON receipt containing its immutable registry digest, source commit,
build input identity, dependency manifest, runtime fingerprint, and platform.
The existing dependency-only cache remains reusable and keeps its current key.

Assembly has a separate Slurm entry point. It accepts only a prepared receipt,
starts the existing persistent local registry with an allocation-private Podman
store, pulls by digest, checks the image's metadata against the receipt, and
exports SquashFS to a temporary path. It validates the filesystem and publishes
the final path without overwriting an existing artifact. It never reads a
Dockerfile, invokes a package installer, or falls back to building an image.
Missing images, mutable references, fingerprint mismatches, partial exports,
and existing outputs fail explicitly. Host tool/bootstrap fetching is distinct
from modifying the prepared image.

Native API and training/resume qualification happen after assembly. Move the
existing Enroot API checks to a separate script; retain the existing 70B
qualification recipes and their source/fingerprint gates. No runtime dependency
pins, worker code, model recipes, or currently running worktrees change.

Reuse the existing private storage, registry lock, helper checksum, and timing
functions through one shared host shell library. Keep build-only source context
and cache decisions in the builder. Each phase has independent timing records.

Validate with real receipt/parser tests and mocked external-service orchestration
that rejects any build/install command during assembly; then assemble a prepared
real image on CSCS and exercise missing/tampered receipt gates. The original
candidate's training and resume qualification remains pinned to its old source.

Native validation exposed Enroot's deleted-cwd cleanup bug. The allocation-local
`enroot_podman_cleanup.sh` adapter accepts only its exact `rm -f -v enroot.*`
command, changes to `/`, and forwards to Podman with unchanged exit semantics.
The installed importer rejects `@digest`, so after digest verification assembly
uses the immutable local image ID. Full `unsquashfs -pf -` data traversal replaces
superblock-only validation. These host compatibility fixes preserve the strict
build/assembly boundary.
