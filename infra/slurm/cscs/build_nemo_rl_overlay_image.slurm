#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=nrl-overlay-image
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --export=NONE

# Build an overlay image on top of a published release instead of rerunning
# the hermetic dependency stage. Two phases, submitted separately:
#
#   PHASE=lock   regenerate uv.lock inside the base image, where the uv
#                workspace's submodule path dependencies exist, and copy it
#                back into the checkout for review and commit.
#   PHASE=build  build docker/Dockerfile.overlay from a committed checkout,
#                export the SquashFS, and gate it.
#
# Required: BASE_SQSH, the release SquashFS to extend. Its registry tag is
# derived from the file name, so the base must have been built by
# build_nemo_rl_image.slurm on this scratch cache.

set -euo pipefail

CSCS_USER=${USER:-$(id -un)}
SCRATCH_ROOT=${SCRATCH_ROOT:-/iopsstor/scratch/cscs/$CSCS_USER}
REPO_DIR=${REPO_DIR:-$SLURM_SUBMIT_DIR}
CACHE_DIR=${CACHE_DIR:-$SCRATCH_ROOT/podman-cache/nemo-rl}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH_ROOT/ce-images/nemo-rl}
REGISTRY_IMAGE_ARCHIVE=${REGISTRY_IMAGE_ARCHIVE:-$SCRATCH_ROOT/podman-cache/bootstrap/registry-3.oci}
PHASE=${PHASE:-build}
BASE_SQSH=${BASE_SQSH:?set BASE_SQSH to the release SquashFS to extend}
VLLM_VERSION=0.26.0
OVERLAY_NEGATIVE_FILTERS=${OVERLAY_NEGATIVE_FILTERS:-"trtllm dynamo nemo_gym"}
EXPECTED_TE=${EXPECTED_TE:-2.18.0+27486e03}
GATE_VENVS="\
nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker \
nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker \
nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"

[[ "$PHASE" == lock || "$PHASE" == build ]] || {
    echo "PHASE must be lock or build, got: $PHASE" >&2
    exit 2
}
[[ -r "$BASE_SQSH" ]] || {
    echo "BASE_SQSH is not readable: $BASE_SQSH" >&2
    exit 2
}

BASE_NAME=$(basename "$BASE_SQSH" .sqsh)
BASE_TAG=${BASE_NAME#nemo-rl-apertus-}
[[ "$BASE_TAG" == vllm-${VLLM_VERSION}-* ]] || {
    echo "BASE_SQSH name does not match nemo-rl-apertus-vllm-${VLLM_VERSION}-<source-id>.sqsh" >&2
    exit 2
}

cd "$REPO_DIR"
echo "Repository:  $REPO_DIR"
echo "Phase:       $PHASE"
echo "Base image:  $BASE_SQSH"
echo "Base tag:    $BASE_TAG"

if [[ "$PHASE" == lock ]]; then
    mkdir -p "$OUTPUT_DIR"
    enroot start --root --rw --mount "$REPO_DIR:/mnt/src" "$BASE_SQSH" bash -exu -o pipefail <<'EOF'
cd /opt/nemo-rl
cp /mnt/src/pyproject.toml pyproject.toml
uv lock
cp uv.lock /mnt/src/uv.lock
EOF
    echo "Lock regenerated with the base image's uv:"
    md5sum pyproject.toml uv.lock
    git diff --stat -- pyproject.toml uv.lock
    echo "LOCK COMPLETE: commit pyproject.toml and uv.lock, then submit PHASE=build"
    exit 0
fi

git diff --quiet && git diff --cached --quiet || {
    echo "The checkout has uncommitted changes; commit them so the fingerprint names a commit" >&2
    git status --short >&2
    exit 1
}

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
[[ -n "${HOST_PYTHON:-}" ]] || {
    echo "No host Python >= 3.11 found; set HOST_PYTHON" >&2
    exit 1
}

SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_REV=${SOURCE_COMMIT:0:12}
SUBMODULE_STATUS=$(git submodule status --recursive)
BUILD_INPUTS_SHA256=$(
    {
        printf 'commit\0%s\0' "$SOURCE_COMMIT"
        printf 'submodules\0%s\0' "$SUBMODULE_STATUS"
        printf 'base_release\0%s\0' "$BASE_TAG"
        printf 'overlay_negative_filters\0%s\0' "$OVERLAY_NEGATIVE_FILTERS"
        printf 'platform\0linux/arm64\0'
    } | sha256sum | cut -d' ' -f1
)
SOURCE_ID="${SOURCE_REV}-${BUILD_INPUTS_SHA256:0:12}"
IMAGE="nemo-rl-apertus:vllm-${VLLM_VERSION}-${SOURCE_ID}"
SQSH_PATH="${OUTPUT_DIR}/nemo-rl-apertus-vllm-${VLLM_VERSION}-${SOURCE_ID}.sqsh"
BUILD_FINGERPRINT_B64=$("$HOST_PYTHON" tools/generate_fingerprint.py | base64 -w0)
echo "Commit:      $SOURCE_COMMIT"
echo "Image:       $IMAGE"
echo "SquashFS:    $SQSH_PATH"
[[ ! -e "$SQSH_PATH" ]] || {
    echo "Output already exists: $SQSH_PATH" >&2
    exit 1
}

for variable in $(env | grep -oE '^SLURM_SPANK[^=]+'); do
    unset "$variable"
done
unset LD_PRELOAD

PODMAN_STORAGE_BASE=${PODMAN_STORAGE_BASE:-/tmp/nemo-rl-podman-$CSCS_USER-$SLURM_JOB_ID}
mkdir -p "$PODMAN_STORAGE_BASE/graphroot" "$PODMAN_STORAGE_BASE/runroot"
PODMAN_STORAGE_CONF="$PODMAN_STORAGE_BASE/storage.conf"
printf '[storage]\ndriver = "overlay"\nrunroot = "%s"\ngraphroot = "%s"\n' \
    "$PODMAN_STORAGE_BASE/runroot" "$PODMAN_STORAGE_BASE/graphroot" \
    > "$PODMAN_STORAGE_CONF"
export CONTAINERS_STORAGE_CONF="$PODMAN_STORAGE_CONF"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$PODMAN_STORAGE_BASE/xdg}"
if [[ ! -d "$XDG_RUNTIME_DIR" ]]; then
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
fi
podman system migrate
GRAPH_ROOT=$(podman info --format '{{.Store.GraphRoot}}')
[[ "$GRAPH_ROOT" == "$PODMAN_STORAGE_BASE/graphroot" ]] || {
    echo "Unexpected Podman graph root: $GRAPH_ROOT" >&2
    exit 1
}
df -h "$PODMAN_STORAGE_BASE"

mkdir -p "$CACHE_DIR"
exec 9>"$CACHE_DIR/.build.lock"
if ! flock --nonblock 9; then
    echo "Another NeMo-RL image build owns $CACHE_DIR/.build.lock" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

SOURCE_CONTEXT="$PODMAN_STORAGE_BASE/source-context"
rm -rf "$SOURCE_CONTEXT"
mkdir -p "$SOURCE_CONTEXT"
git ls-files -z \
    | grep --null-data --invert-match '^3rdparty/' \
    | tar --create --file=- --no-recursion --null --files-from=- \
    | tar --extract --file=- --directory="$SOURCE_CONTEXT"

LOCAL_REGISTRY=127.0.0.1:5000
load_registry_image() {
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
        echo "Local registry did not become ready" >&2
        exit 1
    }
}
load_registry_image
start_local_registry
trap stop_local_registry EXIT

BASE_RELEASE_IMAGE="${LOCAL_REGISTRY}/nemo-rl-apertus:${BASE_TAG}"
curl --fail --silent --output /dev/null \
    --header 'Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
    "http://${LOCAL_REGISTRY}/v2/nemo-rl-apertus/manifests/${BASE_TAG}" || {
    echo "Base release ${BASE_TAG} is not in the persistent registry; rebuild it with build_nemo_rl_image.slurm" >&2
    exit 1
}
echo "Base release: $BASE_RELEASE_IMAGE"
podman pull --tls-verify=false "$BASE_RELEASE_IMAGE"

podman build \
    --layers \
    --tls-verify=false \
    --network=host \
    --platform=linux/arm64 \
    --file docker/Dockerfile.overlay \
    --build-arg "BASE_RELEASE_IMAGE=$BASE_RELEASE_IMAGE" \
    --build-arg "NEMO_RL_COMMIT=$SOURCE_COMMIT" \
    --build-arg "NEMO_RL_BUILD_INPUTS_SHA256=$BUILD_INPUTS_SHA256" \
    --build-arg "NEMO_RL_BUILD_FINGERPRINT_B64=$BUILD_FINGERPRINT_B64" \
    --build-arg "OVERLAY_NEGATIVE_FILTERS=$OVERLAY_NEGATIVE_FILTERS" \
    --tag "$IMAGE" \
    "$SOURCE_CONTEXT"

REGISTRY_IMAGE="${LOCAL_REGISTRY}/nemo-rl-apertus:vllm-${VLLM_VERSION}-${SOURCE_ID}"
podman push --tls-verify=false "$IMAGE" "$REGISTRY_IMAGE"
echo "Persistent image: $REGISTRY_IMAGE"

if ! enroot import -x mount -o "$SQSH_PATH" "podman://$IMAGE"; then
    if [[ -s "$SQSH_PATH" ]] && unsquashfs -s "$SQSH_PATH" >/dev/null; then
        echo "Enroot cleanup returned nonzero; the completed SquashFS is valid" >&2
    else
        exit 1
    fi
fi

for venv in $GATE_VENVS; do
    enroot start --root --rw "$SQSH_PATH" \
        "/opt/ray_venvs/${venv}/bin/python" /opt/nemo-rl/tools/check_nccl_m2n.py
done
enroot start --root --rw "$SQSH_PATH" \
    /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
    - "$EXPECTED_TE" <<'PY'
import sys
from importlib.metadata import version

actual_te = version("transformer-engine")
if actual_te != sys.argv[1]:
    raise RuntimeError(f"Expected Transformer Engine {sys.argv[1]}, found {actual_te}")
print("Transformer Engine unchanged:", actual_te)
PY
enroot start --root --rw "$SQSH_PATH" bash -c \
    'cmp /opt/nemo_rl_container_fingerprint /opt/nemo-rl/.nemo_rl_source_fingerprint.json && cat /opt/nemo_rl_container_fingerprint'

echo "SquashFS checksum and size:"
sha256sum "$SQSH_PATH"
du -h "$SQSH_PATH"
echo "BUILD COMPLETE: $SQSH_PATH"
