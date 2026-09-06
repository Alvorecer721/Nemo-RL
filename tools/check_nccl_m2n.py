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
"""Verify native M2N imports, API signature and NCCL library compatibility.

Runs without a GPU. Communicator and network capability require a separate
topology probe; a passing import check does not qualify cross-node M2N.
"""

import ctypes
import inspect
import re
import sys

MINIMUM_NCCL_VERSION = 23005
REQUIRED_RESHARD_KEYWORDS = {
    "src_mesh",
    "src_placements",
    "dst_mesh",
    "dst_placements",
    "stream",
}


def mapped_nccl_libraries() -> list[str]:
    with open("/proc/self/maps") as maps:
        return sorted(set(re.findall(r"\S*libnccl\.so\S*", maps.read())))


def main() -> int:
    import torch

    from nccl.m2n import reshard

    parameters = inspect.signature(reshard).parameters
    positional = list(parameters)[:3]
    if positional != ["src", "dst", "comm"]:
        raise SystemExit(f"nccl.m2n.reshard positional parameters are {positional}")
    missing = REQUIRED_RESHARD_KEYWORDS - set(parameters)
    if missing:
        raise SystemExit(f"nccl.m2n.reshard lacks keywords {sorted(missing)}")

    import nemo_rl.weight_sync.xferdtensor as xferdtensor

    if xferdtensor._reshard is None:
        raise SystemExit("xferdtensor did not bind the native nccl.m2n.reshard op")

    nccl = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_GLOBAL)
    version = ctypes.c_int()
    if nccl.ncclGetVersion(ctypes.byref(version)) != 0:
        raise SystemExit("ncclGetVersion failed")
    if version.value < MINIMUM_NCCL_VERSION:
        raise SystemExit(
            f"NCCL M2N requires NCCL >= {MINIMUM_NCCL_VERSION}, found {version.value}"
        )

    libraries = mapped_nccl_libraries()
    if len(libraries) != 1:
        raise SystemExit(f"expected exactly one libnccl mapping, found {libraries}")

    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"nccl runtime: {version.value} at {libraries[0]}")
    print("xferdtensor: native nccl.m2n.reshard bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
