# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nemo_gym.token_id_capture.records import TokenEntry
from nemo_gym.token_id_capture.store import TokenCaptureStore

from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.experience.rollouts import _merge_nemo_gym_shard_streams

pytestmark = pytest.mark.nemo_gym


async def _captured_rollout(tmp_path):
    store = TokenCaptureStore(tmp_path)
    await store.put(
        TokenEntry(
            rollout_id="r0",
            model_call_id="c0",
            prompt_token_ids=[1, 5],
            generation_token_ids=[7, 72],
            generation_log_probs=[-0.1, -0.2],
        )
    )
    env = object.__new__(NemoGym.__ray_metadata__.modified_class)
    env._native_token_source = store
    result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 5],
                    "generation_token_ids": [7, 72],
                    "generation_log_probs": [-0.1, -0.2],
                }
            ]
        },
        "reward": 1.0,
    }
    receipt = await env._finalize_native_token_capture({"_ng_rollout_id": "r0"}, result)
    return env, store, result, receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [False, True])
async def test_native_capture_waits_for_consumer_acceptance(tmp_path, accept):
    env, store, result, receipt = await _captured_rollout(tmp_path)
    assert result["response"]["output"][0]["generation_token_ids"] == [7, 72]
    result["native_capture_receipt"] = receipt

    async def stream():
        item = asyncio.get_running_loop().create_future()
        item.set_result((0, {"name": "agent"}, result, None))
        yield item

    actor = SimpleNamespace(
        run_rollouts=SimpleNamespace(
            options=lambda **_: SimpleNamespace(remote=lambda *_: stream())
        ),
        acknowledge_native_token_capture=SimpleNamespace(
            remote=AsyncMock(side_effect=env.acknowledge_native_token_capture)
        ),
    )
    merged = _merge_nemo_gym_shard_streams([("s0", actor, [{}])], "test")
    row = await anext(merged)
    assert "native_capture_receipt" not in row[2]
    assert store.path_for("r0").exists()
    actor.acknowledge_native_token_capture.remote.assert_not_called()
    if accept:
        with pytest.raises(StopAsyncIteration):
            await anext(merged)
        actor.acknowledge_native_token_capture.remote.assert_awaited_once()
        assert not store.path_for("r0").exists()
    else:
        await merged.aclose()
        actor.acknowledge_native_token_capture.remote.assert_not_called()
        assert store.path_for("r0").exists()


@pytest.mark.asyncio
async def test_native_capture_keeps_a_snapshot_changed_after_delivery(tmp_path):
    env, store, _, receipt = await _captured_rollout(tmp_path)
    await store.mark_incomplete("r0", "late-call")
    assert not await env.acknowledge_native_token_capture(receipt)
    assert store.path_for("r0").exists()
    assert store.incomplete_path_for("r0").exists()


@pytest.mark.asyncio
async def test_native_capture_is_inactive_without_a_source():
    env = object.__new__(NemoGym.__ray_metadata__.modified_class)
    env._native_token_source = None
    result = {"response": {"output": []}}
    assert await env._finalize_native_token_capture({}, result) is None
    assert result == {"response": {"output": []}}


def test_native_capture_reads_settings_from_config_paths(tmp_path, monkeypatch):
    import nemo_gym.global_config as gym_config
    from nemo_gym.cli.env import RunHelper

    import nemo_rl.environments.nemo_gym as integration

    capture_dir = tmp_path / "capture"
    config_path = tmp_path / "capture.yaml"
    config_path.write_text(
        f"token_id_capture:\n  enabled: true\n  all_agents: true\n  dir: {capture_dir}\n"
    )
    monkeypatch.setattr(gym_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.delenv(gym_config.NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv"))
    monkeypatch.setattr(integration, "pin_uv_to_path", lambda: None)
    monkeypatch.setattr(integration, "_get_node_ip_local", lambda: "127.0.0.1")
    monkeypatch.setattr(integration, "_get_free_port_local", lambda *_: 18080)
    monkeypatch.setattr(integration.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        integration.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="127.0.0.1:6379"),
    )

    def start_without_servers(self, global_config_dict_parser_config):
        # Use Gym's real parser and cache; only process launch is stubbed.
        gym_config.get_global_config_dict(global_config_dict_parser_config)

    monkeypatch.setattr(RunHelper, "start", start_without_servers)
    env = NemoGym.__ray_metadata__.modified_class(
        {
            "model_name": "test-model",
            "base_urls": ["http://127.0.0.1:8000/v1"],
            "initial_global_config_dict": {"config_paths": [str(config_path)]},
        }
    )
    env._spinup()
    assert env._native_token_source is not None
    assert env._native_token_source.path_for("r0").parent == capture_dir
