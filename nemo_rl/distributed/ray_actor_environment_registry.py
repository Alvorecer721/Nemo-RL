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
import tomllib
from pathlib import Path

from nemo_rl.distributed.actor_environments import (
    ACTOR_ENVIRONMENTS,
    VLLM_CONTROLLER_ACTORS,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, git_root
from nemo_rl.modelopt.registry import MODELOPT_ACTOR_REGISTRY

USE_SYSTEM_EXECUTABLE = os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1"
VLLM_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.VLLM
)
SGLANG_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.SGLANG
)
MCORE_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.MCORE
)
TRTLLM_EXECUTABLE = (
    PY_EXECUTABLES.SYSTEM if USE_SYSTEM_EXECUTABLE else PY_EXECUTABLES.TRTLLM
)
_EXECUTABLES_BY_EXTRAS = {
    ("vllm",): VLLM_EXECUTABLE,
    ("sglang",): SGLANG_EXECUTABLE,
    ("fsdp",): PY_EXECUTABLES.FSDP,
    ("automodel",): PY_EXECUTABLES.AUTOMODEL,
    ("mcore",): MCORE_EXECUTABLE,
    ("trtllm",): TRTLLM_EXECUTABLE,
    ("nemo_gym",): PY_EXECUTABLES.NEMO_GYM,
}


def _reject_undeclared_extras() -> None:
    """Fail on an invalid manifest before any actor venv is created."""
    with (Path(git_root) / "pyproject.toml").open("rb") as project_file:
        declared = tomllib.load(project_file)["project"]["optional-dependencies"]
    required = {
        extra for extras in ACTOR_ENVIRONMENTS.values() for extra in extras or ()
    }
    undeclared = required - declared.keys()
    if undeclared:
        raise ValueError(
            f"Actor environments use undeclared extras: {sorted(undeclared)}"
        )


def _actor_python_env(actor: str, extras: list[str] | None) -> str:
    """Resolve shared extras through the existing runtime executable constants."""
    if extras is None:
        return PY_EXECUTABLES.SYSTEM
    if "modelopt" in extras:
        return MODELOPT_ACTOR_REGISTRY[actor]
    if actor in VLLM_CONTROLLER_ACTORS and extras == ["vllm"]:
        return PY_EXECUTABLES.VLLM
    try:
        return _EXECUTABLES_BY_EXTRAS[tuple(extras)]
    except KeyError as error:
        raise ValueError(
            f"No runtime executable for {actor} with extras {extras}"
        ) from error


_reject_undeclared_extras()

# Worker selections filter build rows only. All registered actors remain available
# at runtime, including system actors and workers omitted from a smaller image.
ACTOR_ENVIRONMENT_REGISTRY: dict[str, str] = {
    actor: _actor_python_env(actor, extras)
    for actor, extras in ACTOR_ENVIRONMENTS.items()
}


def get_actor_python_env(actor_class_fqn: str) -> str:
    if actor_class_fqn in ACTOR_ENVIRONMENT_REGISTRY:
        return ACTOR_ENVIRONMENT_REGISTRY[actor_class_fqn]
    else:
        raise ValueError(
            f"No actor environment registered for {actor_class_fqn}. "
            f"You're attempting to create an actor ({actor_class_fqn}) "
            "without specifying a python environment for it. Please either"
            "specify a python environment in the registry "
            "(nemo_rl.distributed.ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY) "
            "or pass a py_executable to the RayWorkerBuilder. If you're unsure about which "
            "environment to use, a good default is PY_EXECUTABLES.SYSTEM for ray actors that "
            "don't have special dependencies. If you do have special dependencies (say, you're "
            "adding a new generation framework or training backend), you'll need to specify the "
            "appropriate environment. See uv.md for more details."
        )
