#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CSCS Enroot's exit trap calls Docker after removing its current directory.
# Limit this allocation-local compatibility command to that exact cleanup.
set -euo pipefail
if [[ $# != 4 || "$1" != rm || "$2" != -f || "$3" != -v || ! "$4" =~ ^enroot[.][[:alnum:]]+$ ]]; then
    echo "Unexpected command in the Enroot Podman cleanup adapter" >&2
    exit 2
fi
cd /
exec podman rm -f -v -- "$4"
