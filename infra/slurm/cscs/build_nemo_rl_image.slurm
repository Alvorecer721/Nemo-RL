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
LOCAL_REGISTRY_TOOL=${LOCAL_REGISTRY_TOOL:-$SCRATCH_ROOT/tools/local-registry}
CACHE_DIR=${CACHE_DIR:-$SCRATCH_ROOT/podman-cache/nemo-rl}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH_ROOT/ce-images/nemo-rl}
REGISTRY_IMAGE_ARCHIVE=${REGISTRY_IMAGE_ARCHIVE:-$SCRATCH_ROOT/podman-cache/bootstrap/registry-3.oci}
MAX_JOBS=${MAX_JOBS:-4}
NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-90}
VLLM_VERSION=0.25.1
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
[[ -r "$LOCAL_REGISTRY_TOOL/env-registry" ]] || {
    echo "CSCS local-registry helper not found at $LOCAL_REGISTRY_TOOL" >&2
    exit 1
}

SOURCE_REV=$(git rev-parse --short=12 HEAD)
# Hash every non-ignored input that can enter the build context. `git diff`
# covers tracked staged and unstaged changes; the second stream covers new
# source files. Without the latter, two builds with different untracked files
# could receive the same image tag and SquashFS name.
SOURCE_DIFF=$(
    {
        git diff HEAD --binary
        git ls-files --others --exclude-standard -z \
            | sort -z \
            | while IFS= read -r -d '' path; do
                printf 'untracked\0%s\0' "$path"
                git hash-object "$path" | tr '\n' '\0'
            done
    } | sha256sum | cut -c1-12
)
SOURCE_ID="${SOURCE_REV}-${SOURCE_DIFF}"
IMAGE="nemo-rl-apertus:vllm-${VLLM_VERSION}-${SOURCE_ID}"
SQSH_PATH="${OUTPUT_DIR}/nemo-rl-apertus-vllm-${VLLM_VERSION}-${SOURCE_ID}.sqsh"

echo "Repository:  $REPO_DIR"
echo "Source ID:   $SOURCE_ID"
echo "Podman tag:  $IMAGE"
echo "Layer cache: $CACHE_DIR"
echo "SquashFS:    $SQSH_PATH"
echo "Diff status:"
git status --short

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

# The registry process is local to this allocation; its layer data persists in
# CACHE_DIR and is reused by podman-cached through --cache-from/--cache-to.
# shellcheck source=/dev/null
source "$LOCAL_REGISTRY_TOOL/env-registry"
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
# A Slurm-step interruption can leave the helper's stopped container name in
# Podman even though registry data lives separately under CACHE_DIR.
podman container rm --force local_registry >/dev/null 2>&1 || true
registry up "$CACHE_DIR"
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
cleanup() {
    registry down >/dev/null 2>&1 || true
}
trap cleanup EXIT

podman-cached \
    --network=host \
    --platform=linux/arm64 \
    --target=release \
    --build-context "nemo-rl=$REPO_DIR" \
    --build-arg "MAX_JOBS=$MAX_JOBS" \
    --build-arg "NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS" \
    --build-arg "NEMO_RL_COMMIT=${SOURCE_REV}-dirty-${SOURCE_DIFF}" \
    --tag "$IMAGE" \
    --file "$REPO_DIR/docker/Dockerfile" \
    "$REPO_DIR"

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
