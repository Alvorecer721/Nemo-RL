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
"""Unit tests for the train-side bulk-refit shard iterator.

``_iter_local_hf_param_shards`` yields the HF-named local views that the
nccl_reshard bulk path transfers without collectives. The views come from
Megatron-Bridge's per-parameter ``local_hf_param_specs`` contract, filtered by
the bulk-path whitelist; the worker no longer special-cases mapping classes.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

from megatron.bridge.models.conversion.param_mapping import (  # noqa: E402
    LocalHFParamSpec,
)

from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
)

pytestmark = pytest.mark.mcore


class _Mapping:
    def __init__(self, hf_param, specs):
        self.hf_param = hf_param
        self._specs = tuple(specs)

    def local_hf_param_specs(self, global_param_name=None):
        return self._specs


def _task(global_param_name, mapping, param_weight):
    return SimpleNamespace(
        global_param_name=global_param_name,
        mapping=mapping,
        param_weight=param_weight,
        local_hf_param_specs=lambda: mapping.local_hf_param_specs(global_param_name),
    )


def _worker(tasks, bulk_vocab_parallel=False):
    w = object.__new__(MegatronPolicyWorkerImpl)
    w.refit_conversion_tasks = tasks
    w._bulk_vocab_parallel = bulk_vocab_parallel
    return w


def test_iter_local_hf_param_shards_carries_vocab_parallel_views_only_when_enabled():
    name = "model.embed_tokens.weight"

    def tasks():
        mapping = _Mapping(name, [LocalHFParamSpec(name)])
        return [_task("embedding.word_embeddings.weight", mapping, torch.randn(6, 4))]

    assert list(_worker(tasks())._iter_local_hf_param_shards()) == []
    assert [
        n
        for n, _ in _worker(
            tasks(), bulk_vocab_parallel=True
        )._iter_local_hf_param_shards()
    ] == [name]


def test_iter_local_hf_param_shards_yields_the_views_the_mapping_declares():
    fused = torch.randn(8, 4)
    gate = "model.layers.0.mlp.gate_proj.weight"
    up = "model.layers.0.mlp.up_proj.weight"
    mapping = _Mapping(
        {"gate": gate, "up": up},
        [
            LocalHFParamSpec(gate, split_dim=0, split_index=0, split_count=2),
            LocalHFParamSpec(up, split_dim=0, split_index=1, split_count=2),
        ],
    )
    w = _worker([_task("decoder.layers.0.mlp.linear_fc1.weight", mapping, fused)])

    out = dict(w._iter_local_hf_param_shards())

    assert set(out) == {gate, up}
    assert torch.equal(out[gate], fused[:4]) and torch.equal(out[up], fused[4:])


def test_iter_local_hf_param_shards_honors_an_empty_view_declaration():
    name = "model.layers.0.mlp.down_proj.weight"
    w = _worker(
        [
            _task(
                "decoder.layers.0.mlp.linear_fc2.weight",
                _Mapping(name, []),
                torch.randn(4, 8),
            )
        ]
    )

    assert list(w._iter_local_hf_param_shards()) == []


def test_iter_local_hf_param_shards_keeps_misc_path_views_out_of_the_bulk_path():
    name = "model.layers.0.self_attn.q_proj.weight"
    w = _worker(
        [
            _task(
                "decoder.layers.0.self_attention.linear_q.weight",
                _Mapping(name, [LocalHFParamSpec(name)]),
                torch.randn(4, 4),
            )
        ]
    )

    assert list(w._iter_local_hf_param_shards()) == []


def test_iter_local_hf_param_shards_skips_non_local_pp_params_and_scale_tasks():
    name = "model.layers.0.mlp.down_proj.weight"
    mapping = _Mapping(name, [LocalHFParamSpec(name)])
    w = _worker(
        [
            _task("decoder.layers.0.mlp.linear_fc2.weight", mapping, None),
            _task(
                "decoder.layers.0.mlp.linear_fc2.weight_scale_inv",
                mapping,
                torch.ones(1),
            ),
        ]
    )

    assert list(w._iter_local_hf_param_shards()) == []
