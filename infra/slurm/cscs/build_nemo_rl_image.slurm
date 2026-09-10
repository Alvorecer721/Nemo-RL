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
# sbatch spools this script elsewhere; resolve helpers from the submitted repository.
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$REPO_DIR/infra/slurm/cscs/image_storage.sh"

# Build the repository's release image with Podman on a Clariden compute node,
# retain intermediate layers in CSCS's persistent local-registry cache, and
# publish a complete OCI image and immutable receipt for a separate assembly job.

CSCS_USER=${USER:-$(id -un)}
SCRATCH_ROOT=${SCRATCH_ROOT:-/iopsstor/scratch/cscs/$CSCS_USER}
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
    full) NRL_ACTORS="" ;;
    apertus)
        NRL_ACTORS=$(cat "$REPO_DIR/infra/slurm/cscs/profiles/apertus.actors")
        [[ -n "${NRL_ACTORS//[[:space:]]/}" ]] || { echo "Empty Apertus actor profile" >&2; exit 2; }
        ;;
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
image_host_python
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
# publishes the hermetic stage. A second build allocation prepares the release.
HERMETIC_CACHE_TAG=${HERMETIC_CACHE_TAG:-auto}
[[ "$NRL_IMAGE_PROFILE" != apertus || "$BUILD_TRTLLM" == 0 ]] || {
    echo "BUILD_TRTLLM=1 requires NRL_IMAGE_PROFILE=full; apertus excludes TRT-LLM" >&2
    exit 2
}
# A new private store is mandatory because release preparation resets this store.
PODMAN_STORAGE_BASE=${PODMAN_STORAGE_BASE:-/tmp/nemo-rl-podman-$CSCS_USER-$SLURM_JOB_ID}

cd "$REPO_DIR"
REPO_DIR=$(pwd -P)
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

create_image_store

HERMETIC_MANIFEST="$PODMAN_STORAGE_BASE/hermetic-manifest.json"
# Pass the same effective choices to identity generation and Podman. Compiler
# limits are conservative defaults; any intentional override is recorded.
DEPENDENCY_BUILD_ARGS=(
    "BASE_IMAGE=$BASE_IMAGE"
    "NRL_ACTORS=$NRL_ACTORS"
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

TIMING_REPORT="$OUTPUT_DIR/${SOURCE_ID}.build.timings.log"
initialize_image_storage
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

# Persist only the dependency stage in this allocation; release preparation gets
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
run_timed release-publish podman push --tls-verify=false \
    --digestfile "$PODMAN_STORAGE_BASE/release.digest" "$IMAGE" "$REGISTRY_IMAGE"
echo "Persistent image: $REGISTRY_IMAGE"

# Only a successfully published complete image receives an assembly handoff.
RELEASE_DIGEST=$(cat "$PODMAN_STORAGE_BASE/release.digest")
RELEASE_RECEIPT="${SQSH_PATH%.sqsh}.release.json"
"$HOST_PYTHON" "$REPO_DIR/infra/slurm/cscs/image_release_receipt.py" create \
    --manifest "$HERMETIC_MANIFEST" --source-commit "$SOURCE_COMMIT" \
    --image-ref "${LOCAL_REGISTRY}/nemo-rl-${NRL_IMAGE_PROFILE}@${RELEASE_DIGEST}" \
    --vllm-version "$VLLM_VERSION" --output "$RELEASE_RECEIPT"
report_storage prepared-image-complete
echo "PREPARED IMAGE BUILD COMPLETE: $RELEASE_RECEIPT"
echo "Assemble separately with RELEASE_RECEIPT=$RELEASE_RECEIPT"
echo "Timing report: $TIMING_REPORT"
