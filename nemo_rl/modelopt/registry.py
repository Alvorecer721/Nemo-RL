# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os

from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, git_root

USE_SYSTEM_EXECUTABLE = os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1"

MODELOPT_VLLM_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM
    if USE_SYSTEM_EXECUTABLE
    else f"uv run --locked --extra modelopt --extra vllm --directory {git_root}"
)
MODELOPT_AUTOMODEL_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM
    if USE_SYSTEM_EXECUTABLE
    else f"uv run --locked --extra modelopt --extra automodel --directory {git_root}"
)
MODELOPT_MCORE_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM
    if USE_SYSTEM_EXECUTABLE
    else f"uv run --locked --extra modelopt --extra mcore --directory {git_root}"
)

_MODELOPT_EXECUTABLES = {
    ("modelopt", "vllm"): MODELOPT_VLLM_EXECUTABLE,
    ("modelopt", "automodel"): MODELOPT_AUTOMODEL_EXECUTABLE,
    ("modelopt", "mcore"): MODELOPT_MCORE_EXECUTABLE,
}

MODELOPT_ACTOR_REGISTRY: dict[str, str] = {
    actor: _MODELOPT_EXECUTABLES[tuple(extras)]
    for actor, extras in ACTOR_ENVIRONMENTS.items()
    if extras is not None and "modelopt" in extras
}
