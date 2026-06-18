# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Side-load Apertus-specific env + dataset into NeMo-RL's registries.

Call `install()` before `examples/run_grpo.py:main` so the format recipe can refer
to `env_name: apertus_format` and `dataset_name: ApertusFormatDataset` by string.
"""

from nemo_rl.data.datasets.response_datasets import DATASET_REGISTRY
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.utils import ENV_REGISTRY, register_env
from nemo_rl_apertus.data.format_dataset import ApertusFormatDataset

ENV_NAME = "apertus_format"
ENV_FQN = "nemo_rl_apertus.environments.format_env.ApertusFormatEnvironment"
DATASET_NAME = "ApertusFormatDataset"


def install() -> None:
    if ENV_NAME not in ENV_REGISTRY:
        register_env(ENV_NAME, ENV_FQN)
    DATASET_REGISTRY.setdefault(DATASET_NAME, ApertusFormatDataset)
    # Ray actor needs a python executable; transformers + torch are enough.
    ACTOR_ENVIRONMENT_REGISTRY.setdefault(ENV_FQN, PY_EXECUTABLES.SYSTEM)
