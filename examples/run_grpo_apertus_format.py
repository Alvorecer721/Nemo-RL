# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apertus format-reward GRPO entrypoint.

Side-loads `apertus_format` env + `ApertusFormatDataset` into NeMo-RL's registries
before delegating to the standard `examples/run_grpo.py:main`.
"""

from nemo_rl_apertus import register
from examples.run_grpo import main


if __name__ == "__main__":
    register.install()
    main()
