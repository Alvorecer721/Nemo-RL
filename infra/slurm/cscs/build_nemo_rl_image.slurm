#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#SBATCH --account=infra01
#SBATCH --job-name=nrl-vllm026-image
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE
set -euo pipefail

# Build the repository's release image with Podman on a Clariden compute node,
# retain intermediate layers in CSCS's persistent local-registry cache, and
# import the result into an Enroot-compatible SquashFS image before the
# allocation (and Podman's /dev/shm storage) disappears.

CSCS_USER=${USER:-$(id -un)}
SCRATCH_ROOT=${SCRATCH_ROOT:-/iopsstor/scratch/cscs/$CSCS_USER}
REPO_DIR=${REPO_DIR:-$SLURM_SUBMIT_DIR}
CACHE_DIR=${CACHE_DIR:-$SCRATCH_ROOT/podman-cache/nemo-rl}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH_ROOT/ce-images/nemo-rl}
REGISTRY_IMAGE_ARCHIVE=${REGISTRY_IMAGE_ARCHIVE:-$SCRATCH_ROOT/podman-cache/bootstrap/registry-3.oci}
MAX_JOBS=${MAX_JOBS:-4}
NVTE_BUILD_MAX_JOBS=${NVTE_BUILD_MAX_JOBS:-4}
NVTE_BUILD_THREADS_PER_JOB=${NVTE_BUILD_THREADS_PER_JOB:-2}
TRTLLM_BUILD_JOBS=${TRTLLM_BUILD_JOBS:-4}
NRL_IMAGE_PROFILE=${NRL_IMAGE_PROFILE:-apertus}
PLATFORM=${PLATFORM:-linux/arm64}
for variable in MAX_JOBS NVTE_BUILD_MAX_JOBS NVTE_BUILD_THREADS_PER_JOB TRTLLM_BUILD_JOBS; do
    [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || {
        echo "$variable must be a positive integer" >&2
        exit 2
    }
done
case "$NRL_IMAGE_PROFILE" in
    full|apertus) ;;
    *) echo "NRL_IMAGE_PROFILE must be full or apertus" >&2; exit 2 ;;
esac
[[ "$PLATFORM" == linux/arm64 ]] || {
    echo "This CSCS launcher and its pinned base support PLATFORM=linux/arm64" >&2
    exit 2
}
NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-90}
NVTE_WITH_NCCL_EP=${NVTE_WITH_NCCL_EP:-0}
[[ "$NVTE_WITH_NCCL_EP" == 0 || "$NVTE_WITH_NCCL_EP" == 1 ]] || {
    echo "NVTE_WITH_NCCL_EP must be 0 or 1, got: $NVTE_WITH_NCCL_EP" >&2
    exit 2
}
# The fingerprint helper needs modern Python, while the host interpreter varies
# across Clariden images. Prefer a system interpreter and fall back to the
# user's shared Miniconda installation.
if [[ -z "${HOST_PYTHON:-}" ]]; then
    for candidate in \
        /usr/bin/python3.13 \
        /usr/bin/python3.12 \
        /usr/bin/python3.11 \
        "/users/$CSCS_USER/miniconda3/bin/python"; do
        if [[ -x "$candidate" ]] \
            && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
            HOST_PYTHON=$candidate
            break
        fi
    done
fi
# This Apertus production image is validated with vLLM. Keep TensorRT-LLM an
# explicit opt-in until its GH200 source build has independent evidence.
BUILD_TRTLLM=${BUILD_TRTLLM:-0}
# Clariden launches the delivered SquashFS through Slurm/Pyxis/Enroot. It does
# not run a nested container engine, so avoid the default arm64 Apptainer
# source build and its anonymous GitHub dependency downloads. Set this to a
# script name only when deliberately producing an image that needs the
# upstream custom-setup layer.
CUSTOM_SETUP_FNAME=${CUSTOM_SETUP_FNAME-}
VLLM_VERSION=0.26.0
# Pin the arm64 manifest used on Clariden rather than resolving a mutable NGC
# tag differently on a later rebuild. Override only when deliberately updating
# the foundation image.
BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04@sha256:aae478019be74ae86ce33f32c1468dff70cbcc80118831bd64ac67d5a510ac2a}
case "$BUILD_TRTLLM" in
    0) SKIP_TRTLLM_BUILD=1 ;;
    1) SKIP_TRTLLM_BUILD= ;;
    *)
        echo "BUILD_TRTLLM must be 0 or 1, got: $BUILD_TRTLLM" >&2
        exit 1
        ;;
esac
# auto reuses only this build's generated digest, otherwise it builds and
# publishes the hermetic stage. A second allocation assembles the release.
HERMETIC_CACHE_TAG=${HERMETIC_CACHE_TAG:-auto}
[[ "$NRL_IMAGE_PROFILE" != apertus || "$BUILD_TRTLLM" == 0 ]] || {
    echo "BUILD_TRTLLM=1 requires NRL_IMAGE_PROFILE=full; apertus excludes TRT-LLM" >&2
    exit 2
}
# A new private store is mandatory because release assembly resets this store.
PODMAN_STORAGE_BASE=${PODMAN_STORAGE_BASE:-/tmp/nemo-rl-podman-$CSCS_USER-$SLURM_JOB_ID}

cd "$REPO_DIR"
REPO_DIR=$(pwd -P)
if [[ -z "${HOST_PYTHON:-}" || ! -x "$HOST_PYTHON" ]] \
    || ! "$HOST_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    echo "No host Python >=3.11 found; set HOST_PYTHON explicitly." >&2
    exit 1
fi
[[ -f docker/Dockerfile ]] || {
    echo "NeMo-RL Dockerfile not found under REPO_DIR=$REPO_DIR" >&2
    exit 1
}
[[ -f pyproject.toml && -f uv.lock ]] || {
    echo "pyproject.toml or uv.lock is missing under REPO_DIR=$REPO_DIR" >&2
    exit 1
}
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "Release builds require a clean committed repository:" >&2
    git status --short >&2
    exit 1
}
git submodule foreach --quiet --recursive '
    git diff --quiet --ignore-submodules=untracked HEAD -- \
        && git diff --cached --quiet --ignore-submodules=untracked -- \
        || {
        echo "Dirty submodule: $displaypath" >&2
        git status --short >&2
        exit 1
    }
'

SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_REV=${SOURCE_COMMIT:0:12}
SUBMODULE_STATUS=$(git submodule status --recursive)
if [[ -n "$SUBMODULE_STATUS" ]] && printf '%s\n' "$SUBMODULE_STATUS" | grep -q '^[+U-]'; then
    echo "Every submodule must be initialized at its recorded recursive pin" >&2
    exit 1
fi

# Never reuse a caller's existing directory: podman system reset deletes its
# graph. Canonicalize a newly created mode-700 directory before configuring it.
[[ "$PODMAN_STORAGE_BASE" == /* && ! -e "$PODMAN_STORAGE_BASE" && ! -L "$PODMAN_STORAGE_BASE" ]] || {
    echo "PODMAN_STORAGE_BASE must be a new absolute private directory: $PODMAN_STORAGE_BASE" >&2
    exit 1
}
mkdir -m 700 -- "$PODMAN_STORAGE_BASE"
PODMAN_STORAGE_BASE=$(realpath "$PODMAN_STORAGE_BASE")

HERMETIC_MANIFEST="$PODMAN_STORAGE_BASE/hermetic-manifest.json"
# Pass the same effective choices to identity generation and Podman. Compiler
# limits are conservative defaults; any intentional override is recorded.
DEPENDENCY_BUILD_ARGS=(
    "BASE_IMAGE=$BASE_IMAGE"
    "NRL_IMAGE_PROFILE=$NRL_IMAGE_PROFILE"
    "MAX_JOBS=$MAX_JOBS"
    "NVTE_BUILD_MAX_JOBS=$NVTE_BUILD_MAX_JOBS"
    "NVTE_BUILD_THREADS_PER_JOB=$NVTE_BUILD_THREADS_PER_JOB"
    "NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS"
    "NVTE_WITH_NCCL_EP=$NVTE_WITH_NCCL_EP"
    "TRTLLM_BUILD_JOBS=$TRTLLM_BUILD_JOBS"
    "SKIP_TRTLLM_BUILD=$SKIP_TRTLLM_BUILD"
    "CUSTOM_SETUP_FNAME=$CUSTOM_SETUP_FNAME"
)
MANIFEST_ARGS=()
PODMAN_DEPENDENCY_ARGS=()
for assignment in "${DEPENDENCY_BUILD_ARGS[@]}"; do
    MANIFEST_ARGS+=(--build-arg "$assignment")
    PODMAN_DEPENDENCY_ARGS+=(--build-arg "$assignment")
done
CURRENT_FINGERPRINT_SHA256=$("$HOST_PYTHON" tools/image_build_manifest.py generate \
    --repo-root "$REPO_DIR" --base-image "$BASE_IMAGE" --platform "$PLATFORM" \
    --profile "$NRL_IMAGE_PROFILE" "${MANIFEST_ARGS[@]}" --output "$HERMETIC_MANIFEST")
# Validate explicit pins before starting registry or container operations.
"$HOST_PYTHON" tools/image_build_manifest.py select "$HERMETIC_MANIFEST" \
    --cache-tag "$HERMETIC_CACHE_TAG" --available yes >/dev/null
BUILD_INPUTS_SHA256=$(
    {
        printf 'commit\0%s\0' "$SOURCE_COMMIT"
        printf 'hermetic\0%s\0' "$CURRENT_FINGERPRINT_SHA256"
    } | sha256sum | cut -d' ' -f1
)
SOURCE_ID="${SOURCE_REV}-${BUILD_INPUTS_SHA256:0:12}"
IMAGE="nemo-rl-${NRL_IMAGE_PROFILE}:vllm-${VLLM_VERSION}-${SOURCE_ID}"
SQSH_PATH="${OUTPUT_DIR}/nemo-rl-${NRL_IMAGE_PROFILE}-vllm-${VLLM_VERSION}-${SOURCE_ID}.sqsh"
# Runtime version checks continue to receive exactly one fingerprint JSON.
BUILD_FINGERPRINT_B64=$("$HOST_PYTHON" tools/generate_fingerprint.py | base64 -w0)
HERMETIC_MANIFEST_B64=$(base64 -w0 "$HERMETIC_MANIFEST")

echo "Repository:  $REPO_DIR"
echo "Source ID:   $SOURCE_ID"
echo "Base image:  $BASE_IMAGE"
echo "Inputs SHA:  $BUILD_INPUTS_SHA256"
echo "Hermetic mode: $HERMETIC_CACHE_TAG"
echo "Hermetic key:  $CURRENT_FINGERPRINT_SHA256"
echo "Image profile: $NRL_IMAGE_PROFILE"
echo "TensorRT-LLM build: $BUILD_TRTLLM"
echo "Custom setup: ${CUSTOM_SETUP_FNAME:-disabled}"
echo "Podman tag:  $IMAGE"
echo "Layer cache: $CACHE_DIR"
echo "SquashFS:    $SQSH_PATH"
echo "Submodules:"
printf '%s\n' "$SUBMODULE_STATUS"

[[ ! -e "$SQSH_PATH" ]] || {
    echo "Refusing to overwrite existing image: $SQSH_PATH" >&2
    exit 1
}

unset LD_PRELOAD
unset PYTHONPATH
# Image construction does not need a GPU. Prevent the NGC OCI hook from
# injecting GPU/CXI device mounts into transient Podman build containers;
# those FUSE-backed mounts can disconnect while a layer is being committed.
export NVIDIA_VISIBLE_DEVICES=void
for variable in "${!SLURM_SPANK_@}"; do
    unset "$variable"
done

# Rootless Podman briefly needs both the working container and a second copy of
# each layer while committing it. Keep that graph store allocation-local; the
# registry below remains the durable cache across allocations.
mkdir -p "$PODMAN_STORAGE_BASE/graphroot" "$PODMAN_STORAGE_BASE/runroot"
# Use a reproducible helper rather than the host's fuse-overlayfs 1.1.0.
# Newer releases fix directory lookup/iteration and deleted-file handling.
# The profiled hardlink build hit lgetxattr ENOENT while committing a layer on
# the host helper. A full image qualification is still required for this path.
FUSE_OVERLAYFS_VERSION=1.18
FUSE_OVERLAYFS_SHA256=82fed736197b2a881a822e5357b488796f654e8371ce8573a1592331510a0133
FUSE_OVERLAYFS_BIN="$PODMAN_STORAGE_BASE/fuse-overlayfs"
curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
    --output "$FUSE_OVERLAYFS_BIN" \
    "https://github.com/containers/fuse-overlayfs/releases/download/v${FUSE_OVERLAYFS_VERSION}/fuse-overlayfs-aarch64"
printf '%s  %s\n' "$FUSE_OVERLAYFS_SHA256" "$FUSE_OVERLAYFS_BIN" | sha256sum --check -
chmod 700 "$FUSE_OVERLAYFS_BIN"
"$FUSE_OVERLAYFS_BIN" --version
PODMAN_STORAGE_CONF="$PODMAN_STORAGE_BASE/storage.conf"
printf '[storage]\ndriver = "overlay"\nrunroot = "%s"\ngraphroot = "%s"\n' \
    "$PODMAN_STORAGE_BASE/runroot" "$PODMAN_STORAGE_BASE/graphroot" \
    > "$PODMAN_STORAGE_CONF"
printf '\n[storage.options.overlay]\nmount_program = "%s"\n' \
    "$FUSE_OVERLAYFS_BIN" >> "$PODMAN_STORAGE_CONF"
export CONTAINERS_STORAGE_CONF="$PODMAN_STORAGE_CONF"

# Rootless Podman keeps its pause process under XDG_RUNTIME_DIR. A batch step
# has no logind session, so /run/user/<uid> does not exist and every command
# dies on `open .../libpod/tmp/pause.pid`. Interactive allocations happen to
# have one, which is why this only bites plain sbatch submissions.
export XDG_RUNTIME_DIR="$PODMAN_STORAGE_BASE/xdg"
mkdir -m 700 "$XDG_RUNTIME_DIR"

verify_private_graph() {
    local graph_root
    graph_root=$(podman info --format '{{.Store.GraphRoot}}')
    [[ "$graph_root" == "$PODMAN_STORAGE_BASE/graphroot" ]] || {
        echo "Refusing operation on unexpected Podman graph root: $graph_root" >&2
        exit 1
    }
}
verify_private_graph
podman system migrate

report_storage() {
    echo "Storage at stage: $1"
    df -h "$PODMAN_STORAGE_BASE" "$CACHE_DIR" "$OUTPUT_DIR"
    df -i "$PODMAN_STORAGE_BASE" "$CACHE_DIR" "$OUTPUT_DIR"
    "$HOST_PYTHON" - "$PODMAN_STORAGE_BASE" "$CACHE_DIR" "$OUTPUT_DIR" <<'PY_STORAGE'
import os
import sys
for path in sys.argv[1:]:
    stat = os.statvfs(path)
    available = stat.f_bavail * stat.f_frsize
    print(f"Storage available: {path}: {available} bytes; {stat.f_favail} inodes")
    if available <= 0:
        raise SystemExit(f"No storage available: {path}")
    if stat.f_files > 0 and stat.f_favail == 0:
        raise SystemExit(f"No inodes available: {path}")
PY_STORAGE
}
run_timed() {
    local stage=$1 started=$SECONDS status=0
    shift
    "$@" || status=$?
    printf 'TIMING stage=%s elapsed_seconds=%s exit_code=%s\n' \
        "$stage" "$((SECONDS - started))" "$status" | tee -a "$TIMING_REPORT"
    return "$status"
}

mkdir -p "$CACHE_DIR"
# Only one writer may mutate the persistent registry cache. The kernel releases
# this lock automatically when Slurm kills or times out the build process.
exec 9>"$CACHE_DIR/.build.lock"
if ! flock --nonblock 9; then
    echo "Another NeMo-RL image build owns $CACHE_DIR/.build.lock" >&2
    exit 1
fi
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    # CSCS's progressive Lustre layout: one stripe for small files, then
    # spread large SquashFS files over 4 and finally 32 OSTs in 4 MiB chunks.
    lfs setstripe \
        --component-end 4M --stripe-count 1 \
        --component-end 64M --stripe-count 4 \
        --component-end -1 --stripe-count 32 --stripe-size 4M \
        "$OUTPUT_DIR"
fi
lfs getstripe -d "$OUTPUT_DIR"
TIMING_REPORT="$OUTPUT_DIR/${SOURCE_ID}.timings.log"
# Preserve a durable reviewable copy without ever exposing a partial JSON.
"$HOST_PYTHON" -B - "$HERMETIC_MANIFEST" "$CACHE_DIR/manifests/$CURRENT_FINGERPRINT_SHA256.json" <<'PY_MANIFEST'
import sys
from pathlib import Path
sys.path.insert(0, "tools")
from image_build_manifest import read_manifest, write_manifest
write_manifest(Path(sys.argv[2]), read_manifest(Path(sys.argv[1])))
PY_MANIFEST
report_storage initialization

# Construct a tracked-files-only source context from the clean checkout and
# each recursively pinned submodule. Preserve the checkout metadata: Podman
# includes COPY mtimes in its cache identity, so normalizing every file to the
# HEAD timestamp would invalidate unchanged dependency layers. The clean-tree
# guards above make these working-tree bytes identical to the recorded commits.
SOURCE_CONTEXT="$PODMAN_STORAGE_BASE/source-context"
rm -rf "$SOURCE_CONTEXT"
mkdir -p "$SOURCE_CONTEXT"
git ls-files -z \
    | tar --create --file=- --no-recursion --null --files-from=- \
    | tar --extract --file=- --directory="$SOURCE_CONTEXT"
export SOURCE_CONTEXT
git submodule foreach --quiet --recursive '
    destination="$SOURCE_CONTEXT/$displaypath"
    mkdir -p "$destination"
    git ls-files -z \
        | tar --create --file=- --no-recursion --null --files-from=- \
        | tar --extract --file=- --directory="$destination"
'

LOCAL_REGISTRY=127.0.0.1:5000
load_registry_image() {
    # Fresh Clariden nodes have intermittently stalled while pulling this tiny
    # bootstrap image from Docker Hub. Prefer the pinned local OCI copy.
    if podman image exists docker.io/library/registry:3; then
        echo "registry:3 is already present in Podman storage"
    elif [[ -r "$REGISTRY_IMAGE_ARCHIVE" ]]; then
        echo "a6943a0bbcc0395ed76d5bd46000b20ca1a858d7d2ffb5318e218c6f32480d62  $REGISTRY_IMAGE_ARCHIVE" | sha256sum -c -
        podman load --input "$REGISTRY_IMAGE_ARCHIVE"
    else
        podman pull docker.io/registry:3
    fi
}
stop_local_registry() {
    podman container rm --force local_registry >/dev/null 2>&1 || true
}
start_local_registry() {
    # Rootless port publishing depends on a user session bus, which plain
    # sbatch jobs do not have. Bind registry directly on the host network.
    stop_local_registry
    mkdir -p "$CACHE_DIR/data"
    podman run --detach \
        --network host \
        --name local_registry \
        --volume "$CACHE_DIR/data:/var/lib/registry" \
        --env REGISTRY_HTTP_ADDR=127.0.0.1:5000 \
        docker.io/library/registry:3 >/dev/null

    local registry_ready=0
    for _ in {1..30}; do
        if curl --fail --silent "http://${LOCAL_REGISTRY}/v2/" >/dev/null 2>&1; then
            registry_ready=1
            break
        fi
        sleep 1
    done
    [[ "$registry_ready" == 1 ]] || {
        echo "Local registry did not become ready on port 5000" >&2
        podman logs local_registry >&2 || true
        return 1
    }
}
load_registry_image
start_local_registry
finish_build() {
    local status=$?
    trap - EXIT
    stop_local_registry
    printf 'TIMING stage=allocation elapsed_seconds=%s exit_code=%s\n' "$SECONDS" "$status" \
        | tee -a "$TIMING_REPORT"
    exit "$status"
}
trap finish_build EXIT

# Resume source-only releases directly from the persisted final hermetic image.
# `--cache-from` alone cannot do this reliably with Podman: its complete named
# source context is one parent image, so changing fp8.py also changes the cache
# identity of COPY pyproject.toml/uv.lock even when their bytes are identical.
# Replacing only the release stage's FROM keeps the normal Dockerfile
# authoritative while bypassing that false dependency invalidation.
BUILD_DOCKERFILE="$SOURCE_CONTEXT/docker/Dockerfile"
HERMETIC_IMAGE="${LOCAL_REGISTRY}/cache:${CURRENT_FINGERPRINT_SHA256}"
CACHE_AVAILABLE=no
if [[ "$HERMETIC_CACHE_TAG" != rebuild ]]; then
    # Only 404 means a cache miss. Transport/registry failures must fail loudly.
    REGISTRY_STATUS=$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
        -H 'Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
        "http://${LOCAL_REGISTRY}/v2/cache/manifests/${CURRENT_FINGERPRINT_SHA256}")
    case "$REGISTRY_STATUS" in
        200) CACHE_AVAILABLE=yes ;;
        404) ;;
        *) echo "Registry cache lookup failed with HTTP $REGISTRY_STATUS" >&2; exit 1 ;;
    esac
fi
BUILD_TARGET=$("$HOST_PYTHON" tools/image_build_manifest.py select "$HERMETIC_MANIFEST" \
    --cache-tag "$HERMETIC_CACHE_TAG" --available "$CACHE_AVAILABLE")
if [[ "$BUILD_TARGET" == release-core ]]; then
    pull_rc=0
    run_timed hermetic-pull podman pull --tls-verify=false "$HERMETIC_IMAGE" || pull_rc=$?
    if ((pull_rc != 0)); then
        podman image exists "$HERMETIC_IMAGE" || exit "$pull_rc"
        echo "Podman pull exited $pull_rc after materializing $HERMETIC_IMAGE; checking embedded manifest" >&2
    fi
    EMBEDDED_MANIFEST="$PODMAN_STORAGE_BASE/embedded-hermetic-manifest.json"
    podman run --rm --network none --entrypoint /bin/cat "$HERMETIC_IMAGE" \
        /opt/nemo-rl-hermetic-manifest.json > "$EMBEDDED_MANIFEST"
    "$HOST_PYTHON" tools/image_build_manifest.py verify "$HERMETIC_MANIFEST" "$EMBEDDED_MANIFEST"
    echo "Verified embedded hermetic manifest: $CURRENT_FINGERPRINT_SHA256"

    BUILD_DOCKERFILE="$PODMAN_STORAGE_BASE/Dockerfile.release-core"
    sed "s|^FROM hermetic AS release-core$|FROM ${HERMETIC_IMAGE} AS release-core|" \
        "$SOURCE_CONTEXT/docker/Dockerfile" > "$BUILD_DOCKERFILE"
    grep -F "FROM ${HERMETIC_IMAGE} AS release-core" "$BUILD_DOCKERFILE"
    echo "Resuming from hermetic image: $HERMETIC_IMAGE"
else
    echo "Rebuilding only the hermetic dependency stage from source"
    echo "After this dependency build, submit a fresh allocation with HERMETIC_CACHE_TAG=auto"
fi

BUILD_SECRET_ARGS=()
ACCOUNT_HOME=$(getent passwd "$(id -u)" | cut -d: -f6)
GIT_CREDENTIALS_FILE=${GIT_CREDENTIALS_FILE:-$ACCOUNT_HOME/.git-credentials}
if [[ -r "$GIT_CREDENTIALS_FILE" ]]; then
    echo "Using Git credentials as an ephemeral build secret"
    BUILD_SECRET_ARGS+=(--secret "id=git_credentials,src=$GIT_CREDENTIALS_FILE")
fi

report_storage "before-$BUILD_TARGET"
run_timed "$BUILD_TARGET-build" podman build \
    "${BUILD_SECRET_ARGS[@]}" \
    --layers \
    --tls-verify=false \
    --cache-from="${LOCAL_REGISTRY}/cache" \
    --cache-to="${LOCAL_REGISTRY}/cache" \
    --network=host \
    --platform="$PLATFORM" \
    --target="$BUILD_TARGET" \
    --build-context "nemo-rl=$SOURCE_CONTEXT" \
    "${PODMAN_DEPENDENCY_ARGS[@]}" \
    --build-arg "NEMO_RL_HERMETIC_MANIFEST_B64=$HERMETIC_MANIFEST_B64" \
    --build-arg "NEMO_RL_COMMIT=$SOURCE_COMMIT" \
    --build-arg "NEMO_RL_BUILD_INPUTS_SHA256=$BUILD_INPUTS_SHA256" \
    --build-arg "NEMO_RL_BUILD_FINGERPRINT_B64=$BUILD_FINGERPRINT_B64" \
    --tag "$IMAGE" \
    --file "$BUILD_DOCKERFILE" \
    "$SOURCE_CONTEXT"

# Persist only the dependency stage in this allocation; release assembly gets
# a fresh graph in the next allocation, even for an automatic cache miss.
if [[ "$BUILD_TARGET" == "hermetic" ]]; then
    EMBEDDED_MANIFEST="$PODMAN_STORAGE_BASE/built-hermetic-manifest.json"
    podman run --rm --network none --entrypoint /bin/cat "$IMAGE" \
        /opt/nemo-rl-hermetic-manifest.json > "$EMBEDDED_MANIFEST"
    "$HOST_PYTHON" tools/image_build_manifest.py verify "$HERMETIC_MANIFEST" "$EMBEDDED_MANIFEST"
    run_timed hermetic-publish podman push --tls-verify=false "$IMAGE" "$HERMETIC_IMAGE"
    report_storage hermetic-complete
    echo "HERMETIC BUILD COMPLETE: $HERMETIC_IMAGE"
    echo "Manifest: $CACHE_DIR/manifests/$CURRENT_FINGERPRINT_SHA256.json"
    echo "Submit the fresh release allocation with HERMETIC_CACHE_TAG=auto"
    echo "Optional explicit pin: HERMETIC_CACHE_TAG=$CURRENT_FINGERPRINT_SHA256"
    exit 0
fi

# Persist the release core, then discard only this allocation's Podman graph.
# Clariden fixes /tmp at 334 GiB even when a job requests more memory. Keeping
# the broad build cache and the last control/Gym delta in that graph together
# exhausts it; rebuilding the final delta from this exact parent does not.
CORE_REGISTRY_IMAGE="${LOCAL_REGISTRY}/nemo-rl-release-core:${SOURCE_ID}"
run_timed release-core-publish podman push --tls-verify=false "$IMAGE" "$CORE_REGISTRY_IMAGE"
echo "Persistent release core: $CORE_REGISTRY_IMAGE"

stop_local_registry
verify_private_graph
run_timed graph-reset podman system reset --force
mkdir -p \
    "$PODMAN_STORAGE_BASE/graphroot" \
    "$PODMAN_STORAGE_BASE/runroot" \
    "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
podman system migrate
load_registry_image
start_local_registry

curl --fail --silent --show-error --output /dev/null \
    -H 'Accept: application/vnd.oci.image.manifest.v1+json' \
    "http://${LOCAL_REGISTRY}/v2/nemo-rl-release-core/manifests/${SOURCE_ID}"
run_timed release-core-pull podman pull --tls-verify=false "$CORE_REGISTRY_IMAGE"

FINAL_DOCKERFILE="$PODMAN_STORAGE_BASE/Dockerfile.release-final"
sed "s|^FROM release-core AS release$|FROM ${CORE_REGISTRY_IMAGE} AS release|" \
    "$SOURCE_CONTEXT/docker/Dockerfile" > "$FINAL_DOCKERFILE"
grep -F "FROM ${CORE_REGISTRY_IMAGE} AS release" "$FINAL_DOCKERFILE"
echo "Finalizing from a fresh graph: $CORE_REGISTRY_IMAGE"

report_storage before-release-final
run_timed release-final-build podman build \
    "${BUILD_SECRET_ARGS[@]}" \
    --layers \
    --tls-verify=false \
    --network=host \
    --platform="$PLATFORM" \
    --target=release \
    --build-context "nemo-rl=$SOURCE_CONTEXT" \
    "${PODMAN_DEPENDENCY_ARGS[@]}" \
    --build-arg "NEMO_RL_HERMETIC_MANIFEST_B64=$HERMETIC_MANIFEST_B64" \
    --build-arg "NEMO_RL_COMMIT=$SOURCE_COMMIT" \
    --build-arg "NEMO_RL_BUILD_INPUTS_SHA256=$BUILD_INPUTS_SHA256" \
    --build-arg "NEMO_RL_BUILD_FINGERPRINT_B64=$BUILD_FINGERPRINT_B64" \
    --tag "$IMAGE" \
    --file "$FINAL_DOCKERFILE" \
    "$SOURCE_CONTEXT"

# Persist the final manifest/tag as well as its cached layers. Registry blobs
# are content-addressed, so this reuses the cache payload instead of storing a
# second copy and lets Enroot export be retried in a later allocation.
REGISTRY_IMAGE="${LOCAL_REGISTRY}/nemo-rl-${NRL_IMAGE_PROFILE}:vllm-${VLLM_VERSION}-${SOURCE_ID}"
run_timed release-publish podman push --tls-verify=false "$IMAGE" "$REGISTRY_IMAGE"
echo "Persistent image: $REGISTRY_IMAGE"

# Podman storage is node-local tmpfs, so import in this same allocation.  Some
# Enroot/Podman combinations return nonzero only while cleaning their temporary
# mount after mksquashfs has already completed.  Accept that cleanup failure
# only when the resulting image has a valid SquashFS superblock.
report_storage before-squashfs-export
if ! run_timed squashfs-export enroot import -x mount -o "$SQSH_PATH" "podman://$IMAGE"; then
    if [[ -s "$SQSH_PATH" ]] && unsquashfs -s "$SQSH_PATH" >/dev/null; then
        echo "Enroot cleanup returned nonzero; the completed SquashFS is valid" >&2
    else
        exit 1
    fi
fi

# Verify the dependency/API boundary that motivated this image. A writable
# overlay is required because NeMo-RL applies narrowly scoped vLLM source
# compatibility patches at worker startup.
run_timed api-qualification enroot start --root --rw "$SQSH_PATH" \
    /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python - <<'PY'
from importlib.metadata import version

from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat

ensure_vllm_source_compat()

import openai
import vllm
import xgrammar
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tool_parsers import utils as tool_parser_utils

if vllm.__version__ != "0.26.0":
    raise RuntimeError(f"Expected vLLM 0.26.0, found {vllm.__version__}")
if any(
    symbol is None
    for symbol in (OnlineRenderer, ServingTokenization, tool_parser_utils.NamespaceTool)
):
    raise RuntimeError("Required vLLM 0.25 APIs are unavailable")
print("vLLM:", vllm.__version__)
print("OpenAI:", openai.__version__)
print("xgrammar:", version("xgrammar"))
print("vLLM 0.25 renderer, tokenization, and tool-parser imports: OK")
PY

# TE is compiled independently in the Megatron worker environment. Verify the
# exact MCore pairing and the TE APIs used by the dense and grouped-MoE paths.
run_timed api-qualification enroot start --root --rw "$SQSH_PATH" \
    /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
    - "$NVTE_WITH_NCCL_EP" <<'PY'
import ctypes
import sys
from importlib.metadata import version

import megatron.core.extensions.transformer_engine as mcore_te

if not mcore_te.HAVE_TE:
    raise RuntimeError("MCore did not discover Transformer Engine")

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

expected_te = "2.18.0+27486e03"
actual_te = version("transformer-engine")
if actual_te != expected_te:
    raise RuntimeError(f"Expected Transformer Engine {expected_te}, found {actual_te}")

expect_nccl_ep = bool(int(sys.argv[1]))
ep_symbols = [
    name
    for name in ("ep_initialize", "ep_finalize", "ep_get_zero_copy")
    if hasattr(tex, name)
]
if expect_nccl_ep:
    if len(ep_symbols) != 3:
        raise RuntimeError(f"NCCL-EP was enabled but only found symbols: {ep_symbols}")
    nccl = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_GLOBAL)
    nccl_version = ctypes.c_int()
    if nccl.ncclGetVersion(ctypes.byref(nccl_version)) != 0:
        raise RuntimeError("ncclGetVersion failed")
    if nccl_version.value < 23004:
        raise RuntimeError(f"NCCL-EP requires NCCL >=2.30.4, found {nccl_version.value}")
elif ep_symbols:
    raise RuntimeError(f"NCCL-EP was expected to be disabled, found symbols: {ep_symbols}")

if any(
    symbol is None
    for symbol in (mcore_te.TELinear, te.DotProductAttention, te.GroupedLinear)
):
    raise RuntimeError("Required MCore/Transformer Engine APIs are unavailable")
print("Transformer Engine:", actual_te)
print("MCore TE adapters: OK; NCCL-EP enabled:", expect_nccl_ep)
PY

echo "SquashFS checksum and size:"
sha256sum "$SQSH_PATH"
du -h "$SQSH_PATH"
report_storage release-complete
echo "Timing report: $TIMING_REPORT"
echo "BUILD COMPLETE: $SQSH_PATH"
