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
