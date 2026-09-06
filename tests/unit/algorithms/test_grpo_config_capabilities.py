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

from types import SimpleNamespace
from typing import cast

import pytest

from nemo_rl.algorithms.grpo import (
    AsyncGRPOConfig,
    MasterConfig,
    _validate_async_dynamic_sampling_capability,
)


def _config(*, async_enabled: bool, dynamic_sampling: bool) -> MasterConfig:
    return cast(
        MasterConfig,
        SimpleNamespace(
            grpo=SimpleNamespace(
                async_grpo=AsyncGRPOConfig(enabled=async_enabled),
                use_dynamic_sampling=dynamic_sampling,
            )
        ),
    )


def test_async_grpo_rejects_silently_ignored_dynamic_sampling() -> None:
    with pytest.raises(NotImplementedError, match="silently ignored"):
        _validate_async_dynamic_sampling_capability(
            _config(async_enabled=True, dynamic_sampling=True)
        )


@pytest.mark.parametrize(
    ("async_enabled", "dynamic_sampling"),
    [(False, True), (True, False)],
)
def test_dynamic_sampling_capability_accepts_supported_modes(
    async_enabled: bool, dynamic_sampling: bool
) -> None:
    _validate_async_dynamic_sampling_capability(
        _config(
            async_enabled=async_enabled,
            dynamic_sampling=dynamic_sampling,
        )
    )
