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
