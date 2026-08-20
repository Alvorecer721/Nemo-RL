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
#SBATCH --job-name=nrl-vllm0251-image
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
NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-90}
# This Apertus production image is validated with vLLM. Keep TensorRT-LLM an
# explicit opt-in until its GH200 source build has independent evidence.
BUILD_TRTLLM=${BUILD_TRTLLM:-0}
VLLM_VERSION=0.25.1
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
# Completed hermetic stage for the dependency fingerprint below. Source-only
# releases resume directly from this content-addressed manifest, avoiding
# Podman's named-context identity bug that otherwise invalidates uv.lock when
# an unrelated application source file changes. Set to "rebuild" only when
# intentionally changing dependencies.
HERMETIC_CACHE_TAG=${HERMETIC_CACHE_TAG:-a25e913799d0e2bf6eb1cdc477b9cd97a3027f9e038bbc0fd30eae5dfb8595db}
HERMETIC_FINGERPRINT_SHA256=aea5963c8cb4a2371fb7b05b6ea1b2ad8f419ce208c688ab7d9e63949071b569
# Use an explicit allocation-local store so the launcher does not depend on a
# login-node containers/storage.conf. Large release layers are bounded in the
# Dockerfile so they fit this node-local tmpfs.
PODMAN_STORAGE_BASE=${PODMAN_STORAGE_BASE:-/tmp/nemo-rl-podman-$CSCS_USER-$SLURM_JOB_ID}

cd "$REPO_DIR"
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
BUILD_INPUTS_SHA256=$(
    {
        printf 'commit\0%s\0' "$SOURCE_COMMIT"
        printf 'submodules\0%s\0' "$SUBMODULE_STATUS"
        printf 'base_image\0%s\0' "$BASE_IMAGE"
        printf 'max_jobs\0%s\0' "$MAX_JOBS"
        printf 'nvte_cuda_archs\0%s\0' "$NVTE_CUDA_ARCHS"
        printf 'build_trtllm\0%s\0' "$BUILD_TRTLLM"
        printf 'platform\0linux/arm64\0'
        printf 'hermetic_cache_tag\0%s\0' "$HERMETIC_CACHE_TAG"
    } | sha256sum | cut -d' ' -f1
)
SOURCE_ID="${SOURCE_REV}-${BUILD_INPUTS_SHA256:0:12}"
IMAGE="nemo-rl-apertus:vllm-${VLLM_VERSION}-${SOURCE_ID}"
SQSH_PATH="${OUTPUT_DIR}/nemo-rl-apertus-vllm-${VLLM_VERSION}-${SOURCE_ID}.sqsh"
generate_build_fingerprint() {
    python tools/generate_fingerprint.py
    printf '\nbuild_trtllm=%s\n' "$BUILD_TRTLLM"
}
BUILD_FINGERPRINT_B64=$(generate_build_fingerprint | base64 -w0)
CURRENT_FINGERPRINT_SHA256=$(generate_build_fingerprint | sha256sum | cut -d' ' -f1)
BUILD_TARGET=release
if [[ "$HERMETIC_CACHE_TAG" == "rebuild" ]]; then
    BUILD_TARGET=hermetic
fi

echo "Repository:  $REPO_DIR"
echo "Source ID:   $SOURCE_ID"
echo "Base image:  $BASE_IMAGE"
echo "Inputs SHA:  $BUILD_INPUTS_SHA256"
echo "Hermetic:    $HERMETIC_CACHE_TAG"
echo "TensorRT-LLM build: $BUILD_TRTLLM"
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
PODMAN_STORAGE_CONF="$PODMAN_STORAGE_BASE/storage.conf"
printf '[storage]\ndriver = "overlay"\nrunroot = "%s"\ngraphroot = "%s"\n' \
    "$PODMAN_STORAGE_BASE/runroot" "$PODMAN_STORAGE_BASE/graphroot" \
    > "$PODMAN_STORAGE_CONF"
export CONTAINERS_STORAGE_CONF="$PODMAN_STORAGE_CONF"

# Rootless Podman keeps its pause process under XDG_RUNTIME_DIR. A batch step
# has no logind session, so /run/user/<uid> does not exist and every command
# dies on `open .../libpod/tmp/pause.pid`. Interactive allocations happen to
# have one, which is why this only bites plain sbatch submissions.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$PODMAN_STORAGE_BASE/xdg}"
if [[ ! -d "$XDG_RUNTIME_DIR" ]]; then
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
fi

# Reconcile rootless runtime state after an interrupted Slurm step while
# preserving images and layers already present in the allocation-local store.
podman system migrate
# Interrupted `podman build` processes leave Buildah working containers in the
# graph store.  They are not image cache entries, but each keeps an overlay
# mounted and can exhaust tmpfs during a later layer commit.  Remove only that
# transient build state; images remain available locally and in CACHE_DIR.
podman system prune --external --force
podman system prune --build --force

GRAPH_ROOT=$(podman info --format '{{.Store.GraphRoot}}')
[[ "$GRAPH_ROOT" == "$PODMAN_STORAGE_BASE/graphroot" ]] || {
    echo "Unexpected Podman graph root: $GRAPH_ROOT" >&2
    exit 1
}
echo "Podman graph root: $GRAPH_ROOT"
df -h "$PODMAN_STORAGE_BASE"

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

# The registry process is local to this allocation; its layer data persists in
# CACHE_DIR and is reused through Podman's --cache-from/--cache-to support.
# Fresh Clariden nodes have intermittently stalled while pulling this tiny
# bootstrap image from Docker Hub. Load the pinned local OCI copy when present;
# retain a visible network fallback instead of the helper's redirected pull.
if podman image exists docker.io/library/registry:3; then
    echo "registry:3 is already present in Podman storage"
elif [[ -r "$REGISTRY_IMAGE_ARCHIVE" ]]; then
    echo "a6943a0bbcc0395ed76d5bd46000b20ca1a858d7d2ffb5318e218c6f32480d62  $REGISTRY_IMAGE_ARCHIVE" | sha256sum -c -
    podman load --input "$REGISTRY_IMAGE_ARCHIVE"
else
    podman pull docker.io/registry:3
fi
# Rootless port publishing depends on a user session bus, which plain sbatch
# jobs do not have. Bind registry directly on the host network instead.
podman container rm --force local_registry >/dev/null 2>&1 || true
mkdir -p "$CACHE_DIR/data"
podman run --detach \
    --network host \
    --name local_registry \
    --volume "$CACHE_DIR/data:/var/lib/registry" \
    --env REGISTRY_HTTP_ADDR=127.0.0.1:5000 \
    docker.io/library/registry:3 >/dev/null
LOCAL_REGISTRY=127.0.0.1:5000
cleanup() {
    podman container rm --force local_registry >/dev/null 2>&1 || true
}
trap cleanup EXIT
registry_ready=0
for _ in {1..30}; do
    if curl --fail --silent "http://127.0.0.1:5000/v2/" >/dev/null 2>&1; then
        registry_ready=1
        break
    fi
    sleep 1
done
[[ "$registry_ready" == 1 ]] || {
    echo "Local registry did not become ready on port 5000" >&2
    podman logs local_registry >&2 || true
    exit 1
}

# Resume source-only releases directly from the persisted final hermetic image.
# `--cache-from` alone cannot do this reliably with Podman: its complete named
# source context is one parent image, so changing fp8.py also changes the cache
# identity of COPY pyproject.toml/uv.lock even when their bytes are identical.
# Replacing only the release stage's FROM keeps the normal Dockerfile
# authoritative while bypassing that false dependency invalidation.
BUILD_DOCKERFILE="$SOURCE_CONTEXT/docker/Dockerfile"
if [[ "$HERMETIC_CACHE_TAG" != "rebuild" ]]; then
    [[ "$CURRENT_FINGERPRINT_SHA256" == "$HERMETIC_FINGERPRINT_SHA256" ]] || {
        echo "The pinned hermetic image does not match current dependencies." >&2
        echo "Expected fingerprint: $HERMETIC_FINGERPRINT_SHA256" >&2
        echo "Current fingerprint:  $CURRENT_FINGERPRINT_SHA256" >&2
        echo "Rebuild dependencies and record their final cache tag." >&2
        exit 1
    }
    HERMETIC_IMAGE="${LOCAL_REGISTRY}/cache:${HERMETIC_CACHE_TAG}"
    curl --fail --silent --show-error --output /dev/null \
        -H 'Accept: application/vnd.oci.image.manifest.v1+json' \
        "http://${LOCAL_REGISTRY}/v2/cache/manifests/${HERMETIC_CACHE_TAG}"
    podman pull --tls-verify=false "$HERMETIC_IMAGE"
    HERMETIC_PYPROJECT_MD5=$(podman run --rm \
        --entrypoint /usr/bin/md5sum "$HERMETIC_IMAGE" \
        /opt/nemo-rl/pyproject.toml | cut -d' ' -f1)
    HERMETIC_LOCK_MD5=$(podman run --rm \
        --entrypoint /usr/bin/md5sum "$HERMETIC_IMAGE" \
        /opt/nemo-rl/uv.lock | cut -d' ' -f1)
    HERMETIC_VLLM_VERSION=$(podman run --rm \
        --entrypoint /opt/nemo_rl_venv/bin/python "$HERMETIC_IMAGE" -c \
        'import tomllib; lock = tomllib.load(open("/opt/nemo-rl/uv.lock", "rb")); print(next(package["version"] for package in lock["package"] if package["name"] == "vllm"))')
    [[ "$HERMETIC_PYPROJECT_MD5" == "97ef0774da410409aadeb5cb9abc73a2" ]] # pragma: allowlist secret
    [[ "$HERMETIC_LOCK_MD5" == "f15bff1a3d93527478961a86c99efc14" ]] # pragma: allowlist secret
    [[ "$HERMETIC_VLLM_VERSION" == "$VLLM_VERSION" ]]

    BUILD_DOCKERFILE="$PODMAN_STORAGE_BASE/Dockerfile.release"
    sed "s|^FROM hermetic AS release$|FROM ${HERMETIC_IMAGE} AS release|" \
        "$SOURCE_CONTEXT/docker/Dockerfile" > "$BUILD_DOCKERFILE"
    grep -F "FROM ${HERMETIC_IMAGE} AS release" "$BUILD_DOCKERFILE"
    echo "Resuming from hermetic image: $HERMETIC_IMAGE"
else
    echo "Rebuilding only the hermetic dependency stage from source"
    echo "Release assembly will require a second build with the printed cache pin"
fi

podman build \
    --layers \
    --tls-verify=false \
    --cache-from="${LOCAL_REGISTRY}/cache" \
    --cache-to="${LOCAL_REGISTRY}/cache" \
    --network=host \
    --platform=linux/arm64 \
    --target="$BUILD_TARGET" \
    --build-context "nemo-rl=$SOURCE_CONTEXT" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "MAX_JOBS=$MAX_JOBS" \
    --build-arg "NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS" \
    --build-arg "SKIP_TRTLLM_BUILD=$SKIP_TRTLLM_BUILD" \
    --build-arg "NEMO_RL_COMMIT=$SOURCE_COMMIT" \
    --build-arg "NEMO_RL_BUILD_INPUTS_SHA256=$BUILD_INPUTS_SHA256" \
    --build-arg "NEMO_RL_BUILD_FINGERPRINT_B64=$BUILD_FINGERPRINT_B64" \
    --tag "$IMAGE" \
    --file "$BUILD_DOCKERFILE" \
    "$SOURCE_CONTEXT"

# A dependency-changing build and release assembly cannot safely share the
# 334 GiB allocation-local Podman graph: committing a release worker layer may
# briefly duplicate enough data to exhaust /tmp. Persist the completed
# hermetic image under its dependency fingerprint and stop. After recording
# the printed pin values, a second job starts from a clean graph and builds the
# release target only.
if [[ "$BUILD_TARGET" == "hermetic" ]]; then
    PUBLISHED_HERMETIC_TAG="$CURRENT_FINGERPRINT_SHA256"
    PUBLISHED_HERMETIC_IMAGE="${LOCAL_REGISTRY}/cache:${PUBLISHED_HERMETIC_TAG}"
    podman push --tls-verify=false "$IMAGE" "$PUBLISHED_HERMETIC_IMAGE"
    echo "HERMETIC BUILD COMPLETE: $PUBLISHED_HERMETIC_IMAGE"
    echo "Pin these values before submitting the fresh release build:"
    echo "HERMETIC_CACHE_TAG=$PUBLISHED_HERMETIC_TAG"
    echo "HERMETIC_FINGERPRINT_SHA256=$CURRENT_FINGERPRINT_SHA256"
    echo "HERMETIC_PYPROJECT_MD5=$(md5sum pyproject.toml | cut -d' ' -f1)"
    echo "HERMETIC_LOCK_MD5=$(md5sum uv.lock | cut -d' ' -f1)"
    exit 0
fi

# Persist the final manifest/tag as well as its cached layers. Registry blobs
# are content-addressed, so this reuses the cache payload instead of storing a
# second copy and lets Enroot export be retried in a later allocation.
REGISTRY_IMAGE="${LOCAL_REGISTRY}/nemo-rl-apertus:vllm-${VLLM_VERSION}-${SOURCE_ID}"
podman push --tls-verify=false "$IMAGE" "$REGISTRY_IMAGE"
echo "Persistent image: $REGISTRY_IMAGE"

# Podman storage is node-local tmpfs, so import in this same allocation.  Some
# Enroot/Podman combinations return nonzero only while cleaning their temporary
# mount after mksquashfs has already completed.  Accept that cleanup failure
# only when the resulting image has a valid SquashFS superblock.
if ! enroot import -x mount -o "$SQSH_PATH" "podman://$IMAGE"; then
    if [[ -s "$SQSH_PATH" ]] && unsquashfs -s "$SQSH_PATH" >/dev/null; then
        echo "Enroot cleanup returned nonzero; the completed SquashFS is valid" >&2
    else
        exit 1
    fi
fi

# Verify the dependency/API boundary that motivated this image. A writable
# overlay is required because NeMo-RL applies narrowly scoped vLLM source
# compatibility patches at worker startup.
enroot start --root --rw "$SQSH_PATH" \
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

assert vllm.__version__ == "0.25.1", vllm.__version__
assert OnlineRenderer is not None
assert ServingTokenization is not None
assert tool_parser_utils.NamespaceTool is not None
print("vLLM:", vllm.__version__)
print("OpenAI:", openai.__version__)
print("xgrammar:", version("xgrammar"))
print("vLLM 0.25 renderer, tokenization, and tool-parser imports: OK")
PY

echo "SquashFS checksum and size:"
sha256sum "$SQSH_PATH"
du -h "$SQSH_PATH"
echo "BUILD COMPLETE: $SQSH_PATH"
