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

# Shared host storage only: this file never builds or installs image contents.

image_host_python() {
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
if [[ -z "${HOST_PYTHON:-}" || ! -x "$HOST_PYTHON" ]] \
    || ! "$HOST_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    echo "No host Python >=3.11 found; set HOST_PYTHON explicitly." >&2
    exit 1
fi

}

create_image_store() {
# Never reuse a caller's existing directory: podman system reset deletes its
# graph. Canonicalize a newly created mode-700 directory before configuring it.
[[ "$PODMAN_STORAGE_BASE" == /* && ! -e "$PODMAN_STORAGE_BASE" && ! -L "$PODMAN_STORAGE_BASE" ]] || {
    echo "PODMAN_STORAGE_BASE must be a new absolute private directory: $PODMAN_STORAGE_BASE" >&2
    exit 1
}
mkdir -m 700 -- "$PODMAN_STORAGE_BASE"
PODMAN_STORAGE_BASE=$(realpath "$PODMAN_STORAGE_BASE")


}

initialize_image_storage() {
unset LD_PRELOAD
unset PYTHONPATH
# Image construction does not need a GPU. Prevent the NGC OCI hook from
# injecting GPU/CXI device mounts into transient Podman build containers;
# those FUSE-backed mounts can disconnect while a layer is being committed.
export NVIDIA_VISIBLE_DEVICES=void
for variable in "${!SLURM_SPANK_@}" "${!_SLURM_SPANK_@}"; do
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
    local status=${1:-$?}
    trap - EXIT
    stop_local_registry
    printf 'TIMING stage=allocation elapsed_seconds=%s exit_code=%s\n' "$SECONDS" "$status" \
        | tee -a "$TIMING_REPORT"
    exit "$status"
}
trap finish_build EXIT
}
