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
"""Unit tests for the train-side layer -> PP stage map used by the bulk refit.

``_build_layer_to_pp_stage`` mirrors Megatron-LM's layer distribution and
assigns the non-layer groups the bulk path transfers: the embedding lives on
the first stage, the output head and final norm on the last.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
)

pytestmark = pytest.mark.mcore


def _worker(num_layers, **overrides):
    w = object.__new__(MegatronPolicyWorkerImpl)
    config = SimpleNamespace(
        num_layers=num_layers,
        pipeline_model_parallel_layout=None,
        virtual_pipeline_model_parallel_size=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
    )
    for k, v in overrides.items():
        setattr(config, k, v)
    w.model = SimpleNamespace(config=config)
    return w


def test_layer_to_pp_stage_places_embedding_first_and_head_last():
    stages = _worker(4)._build_layer_to_pp_stage(pp_size=2, layer_prefix="model")

    assert stages["model.layers.0"] == 0 and stages["model.layers.3"] == 1
    assert stages["model.embed_tokens"] == 0
    assert stages["lm_head"] == 1
    assert stages["model.norm"] == 1


def test_layer_to_pp_stage_honors_uneven_first_stage():
    stages = _worker(5, num_layers_in_first_pipeline_stage=1)._build_layer_to_pp_stage(
        pp_size=3, layer_prefix="model"
    )

    assert [stages[f"model.layers.{i}"] for i in range(5)] == [0, 1, 1, 2, 2]
    assert stages["model.embed_tokens"] == 0 and stages["lm_head"] == 2
