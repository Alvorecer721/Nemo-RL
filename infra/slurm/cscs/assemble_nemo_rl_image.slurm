#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --account=infra01
#SBATCH --job-name=nrl-image-assemble
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE
set -euo pipefail
# sbatch spools this script elsewhere; resolve helpers from the submitted repository.
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$REPO_DIR/infra/slurm/cscs/image_storage.sh"

# REPO_DIR supplies host tools only. Application sources and Dockerfiles are
# not read; no dependency installation, compilation or rebuild fallback is used.
: "${RELEASE_RECEIPT:?Set RELEASE_RECEIPT to the prepared image JSON}"
CSCS_USER=${USER:-$(id -un)}
SCRATCH_ROOT=${SCRATCH_ROOT:-/iopsstor/scratch/cscs/$CSCS_USER}
CACHE_DIR=${CACHE_DIR:-$SCRATCH_ROOT/podman-cache/nemo-rl}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH_ROOT/ce-images/nemo-rl}
REGISTRY_IMAGE_ARCHIVE=${REGISTRY_IMAGE_ARCHIVE:-$SCRATCH_ROOT/podman-cache/bootstrap/registry-3.oci}
PODMAN_STORAGE_BASE=${PODMAN_STORAGE_BASE:-/tmp/nemo-rl-assemble-$CSCS_USER-$SLURM_JOB_ID}
IMAGE_TOOLS_DIR="$REPO_DIR/infra/slurm/cscs"
image_host_python
# Capture the producer receipt once; every subsequent check uses this snapshot.
RECEIPT_FIELDS=$("$HOST_PYTHON" "$IMAGE_TOOLS_DIR/image_release_receipt.py" fields "$RELEASE_RECEIPT")
mapfile -t fields <<< "$RECEIPT_FIELDS"
[[ ${#fields[@]} == 2 ]] || { echo "Invalid receipt fields" >&2; exit 1; }
IMAGE_REF=${fields[0]}
IMAGE_BASENAME=${fields[1]}
SQSH_PATH="$OUTPUT_DIR/$IMAGE_BASENAME.sqsh"
[[ ! -e "$SQSH_PATH" && ! -L "$SQSH_PATH" ]] || {
    echo "Refusing to overwrite existing image: $SQSH_PATH" >&2
    exit 1
}
create_image_store
cp -- "$RELEASE_RECEIPT" "$PODMAN_STORAGE_BASE/release.json"
RELEASE_RECEIPT="$PODMAN_STORAGE_BASE/release.json"
# Detect a producer receipt changed between initial validation and the copy.
[[ "$("$HOST_PYTHON" "$IMAGE_TOOLS_DIR/image_release_receipt.py" fields "$RELEASE_RECEIPT")" == "$RECEIPT_FIELDS" ]] || {
    echo "Release receipt changed during initialization" >&2; exit 1;
}
TIMING_REPORT="$OUTPUT_DIR/$IMAGE_BASENAME.assembly-$SLURM_JOB_ID.timings.log"
initialize_image_storage
report_storage assembly-initialization

run_timed prepared-image-pull podman pull --tls-verify=false "$IMAGE_REF"
podman image inspect "$IMAGE_REF" > "$PODMAN_STORAGE_BASE/inspection.json"
podman run --rm --network none --entrypoint /bin/cat "$IMAGE_REF" \
    /opt/nemo-rl-hermetic-manifest.json > "$PODMAN_STORAGE_BASE/hermetic.json"
podman run --rm --network none --entrypoint /bin/cat "$IMAGE_REF" \
    /opt/nemo_rl_container_fingerprint > "$PODMAN_STORAGE_BASE/fingerprint.json"
IMAGE_ID=$("$HOST_PYTHON" "$IMAGE_TOOLS_DIR/image_release_receipt.py" verify-image "$RELEASE_RECEIPT" \
    --inspection "$PODMAN_STORAGE_BASE/inspection.json" \
    --manifest "$PODMAN_STORAGE_BASE/hermetic.json" \
    --fingerprint "$PODMAN_STORAGE_BASE/fingerprint.json")
echo "Prepared image digest, source and fingerprints verified: $IMAGE_REF"

PARTIAL_SQSH="$SQSH_PATH.partial-$SLURM_JOB_ID"
[[ ! -e "$PARTIAL_SQSH" && ! -L "$PARTIAL_SQSH" ]] || {
    echo "Refusing to overwrite existing partial export: $PARTIAL_SQSH" >&2; exit 1;
}
trap 'status=$?; rm -f -- "$PARTIAL_SQSH"; finish_build "$status"' EXIT
# Enroot's local URI parser rejects @digests. The verified local image ID is
# immutable too and is accepted by the installed importer.
# Keep Enroot's cleanup outside the temporary directory it just removed.
# Its Docker compatibility call otherwise makes Podman fail getcwd after export.
mkdir "$PODMAN_STORAGE_BASE/enroot-bin"
cp "$REPO_DIR/infra/slurm/cscs/enroot_podman_cleanup.sh" "$PODMAN_STORAGE_BASE/enroot-bin/docker"
chmod 700 "$PODMAN_STORAGE_BASE/enroot-bin/docker"
PATH="$PODMAN_STORAGE_BASE/enroot-bin:$PATH" run_timed squashfs-export enroot import -x mount -o "$PARTIAL_SQSH" "podman://$IMAGE_ID"
# Traverse metadata and decompress all unique regular files to stdout, discarded
# without extracting another filesystem. A superblock-only check misses truncation.
[[ -s "$PARTIAL_SQSH" ]] || { echo "Export produced no SquashFS" >&2; exit 1; }
run_timed squashfs-validation unsquashfs -no-progress -pf - "$PARTIAL_SQSH" >/dev/null
chmod 644 "$PARTIAL_SQSH"
# Atomic no-replace publication, on the same filesystem as the temporary file.
ln -- "$PARTIAL_SQSH" "$SQSH_PATH"
rm -- "$PARTIAL_SQSH"
sha256sum "$SQSH_PATH" > "$SQSH_PATH.sha256"
du -h "$SQSH_PATH"
report_storage assembly-complete
echo "ASSEMBLY COMPLETE: $SQSH_PATH"
echo "Prepared image: $IMAGE_REF"
echo "Timing report: $TIMING_REPORT"
echo "Run native API and training/resume qualification separately."
