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
"""Minimal behavioral invariants for the data-plane wiring.

* ``examples/run_grpo._select_trainer`` dispatches the legacy trainer
  when ``data_plane`` is absent and the sync trainer when enabled.
* The ``DataPlaneClient`` ABC carries every method adapters depend on.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_run_grpo_dispatches_both_trainers():
    """``examples/run_grpo._select_trainer`` returns the TQ-mediated
    ``grpo_train_sync`` iff ``data_plane.enabled`` is true, and the
    legacy ``grpo_train`` otherwise."""
    import sys

    sys.path.insert(0, str(REPO / "examples"))
    try:
        from run_grpo import _select_trainer
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert _select_trainer(cfg_legacy) is grpo_train

    cfg_sync = MasterConfig.model_construct(data_plane={"enabled": True})
    assert _select_trainer(cfg_sync) is grpo_train_sync


def test_run_grpo_entrypoint_rejects_mixed_async_transport() -> None:
    """Legacy async cannot construct TQ and then consume an in-memory buffer."""
    import sys

    sys.path.insert(0, str(REPO / "examples"))
    try:
        from run_grpo import _validate_entrypoint_contract
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import GRPOConfig, MasterConfig

    cfg = MasterConfig.model_construct(
        grpo=GRPOConfig(async_grpo={"enabled": True}),
        data_plane={"enabled": True},
        policy={"generation": {"backend": "vllm", "vllm_cfg": {"async_engine": True}}},
        data={"use_multiple_dataloader": False},
    )
    with pytest.raises(ValueError, match="in-memory ReplayBuffer"):
        _validate_entrypoint_contract(cfg)


def test_run_grpo_entrypoint_accepts_each_supported_transport() -> None:
    import sys

    sys.path.insert(0, str(REPO / "examples"))
    try:
        from run_grpo import _validate_entrypoint_contract
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import GRPOConfig, MasterConfig

    sync_tq = MasterConfig.model_construct(
        grpo=GRPOConfig(async_grpo={"enabled": False}),
        data_plane={"enabled": True},
    )
    _validate_entrypoint_contract(sync_tq)

    legacy_async = MasterConfig.model_construct(
        grpo=GRPOConfig(async_grpo={"enabled": True}),
        data_plane={"enabled": False},
        policy={"generation": {"backend": "vllm", "vllm_cfg": {"async_engine": True}}},
        data={"use_multiple_dataloader": False},
    )
    _validate_entrypoint_contract(legacy_async)


def test_run_grpo_entrypoint_rejects_single_controller_schema() -> None:
    import sys

    sys.path.insert(0, str(REPO / "examples"))
    try:
        from run_grpo import _validate_entrypoint_contract
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import GRPOConfig, MasterConfig

    null_legacy_block = MasterConfig.model_construct(
        grpo=GRPOConfig(async_grpo=None), data_plane={"enabled": True}
    )
    with pytest.raises(ValueError, match="SingleController"):
        _validate_entrypoint_contract(null_legacy_block)

    ignored_async_rl = MasterConfig.model_construct(
        grpo=GRPOConfig(async_grpo={"enabled": False}),
        data_plane={"enabled": False},
        async_rl={"sampler": {"name": "in_order"}},
    )
    with pytest.raises(ValueError, match=r"async_rl\.\*"):
        _validate_entrypoint_contract(ignored_async_rl)


def test_sync_trainer_rejects_message_level_advantage_penalties():
    from nemo_rl.algorithms.grpo import GRPOConfig, MasterConfig
    from nemo_rl.algorithms.grpo_sync import (
        _raise_if_message_level_advantage_penalties_enabled,
    )

    cfg_disabled = MasterConfig.model_construct(grpo=GRPOConfig())
    _raise_if_message_level_advantage_penalties_enabled(cfg_disabled)

    cfg_enabled = MasterConfig.model_construct(
        grpo=GRPOConfig(
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=None,
        )
    )
    with pytest.raises(
        NotImplementedError,
        match="grpo.invalid_tool_call_advantage",
    ):
        _raise_if_message_level_advantage_penalties_enabled(cfg_enabled)


@pytest.mark.parametrize(
    "method",
    [
        "register_partition",
        "claim_meta",
        "get_data",
        "put_samples",
        "get_samples",
        "list_sample_ids",
        "clear_samples",
        "check_consumption_status",
        "save_checkpoint",
        "load_checkpoint",
        "close",
    ],
)
def test_data_plane_client_abc_method_present(method: str) -> None:
    """The ``DataPlaneClient`` ABC is the swap surface; a silent rename
    is a breaking change for every adapter."""
    from nemo_rl.data_plane.interfaces import DataPlaneClient

    assert hasattr(DataPlaneClient, method), (
        f"DataPlaneClient ABC is missing required method {method!r}. "
        "This is a breaking change for every adapter."
    )
