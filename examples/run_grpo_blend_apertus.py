# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-env GRPO entrypoint for Apertus 1.5.

Delegates to ``examples/nemo_gym/run_grpo_nemo_gym.py:main`` after enforcing
the Apertus xIELU/Bridge runtime guard.
"""

from nemo_rl_apertus.runtime_guard import assert_apertus_runtime
from examples.nemo_gym.run_grpo_nemo_gym import main


if __name__ == "__main__":
    assert_apertus_runtime()
    main()
