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

import pytest
import torch

from nemo_rl.models.policy.lm_policy import Policy, RefitManifestMismatchError
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker


def test_policy_forwards_nccl_peer_to_workers():
    calls = []

    class WorkerGroup:
        def run_all_workers_single_data(self, method_name, **kwargs):
            calls.append((method_name, kwargs))
            return ["future"]

        def shutdown(self, **_kwargs):
            pass

    policy = Policy.__new__(Policy)
    policy.worker_group = WorkerGroup()

    futures = policy.init_collective(
        "127.0.0.1",
        1234,
        4,
        train_world_size=2,
        nccl_peer="vllm",
    )

    assert futures == ["future"]
    assert calls == [
        (
            "init_collective",
            {
                "ip": "127.0.0.1",
                "port": 1234,
                "world_size": 4,
                "train_world_size": 2,
                "nccl_peer": "vllm",
            },
        )
    ]


def test_policy_worker_initializes_requested_nccl_peer(monkeypatch):
    calls = []

    class ProcessGroup:
        def __init__(self, **kwargs):
            calls.append(("create", kwargs))

        def init_nccl_communicator(self, **kwargs):
            calls.append(("init", kwargs))

    monkeypatch.setattr(
        "nemo_rl.distributed.stateless_process_group.StatelessProcessGroup",
        ProcessGroup,
    )
    monkeypatch.setattr(
        "nemo_rl.models.policy.workers.base_policy_worker.torch.cuda.current_device",
        lambda: 3,
    )

    worker = AbstractPolicyWorker.__new__(AbstractPolicyWorker)
    worker.rank = 1
    worker.init_collective(
        "127.0.0.1",
        1234,
        4,
        train_world_size=2,
        nccl_peer="vllm",
    )

    assert calls == [
        (
            "create",
            {
                "master_address": "127.0.0.1",
                "port": 1234,
                "rank": 1,
                "world_size": 4,
            },
        ),
        ("init", {"device": 3, "peer": "vllm"}),
    ]


def test_policy_forwards_packed_collective_options_to_workers():
    calls = []

    class WorkerGroup:
        def run_all_workers_single_data(self, method_name, **kwargs):
            calls.append((method_name, kwargs))
            return ["future"]

        def shutdown(self, **_kwargs):
            pass

    policy = Policy.__new__(Policy)
    policy.worker_group = WorkerGroup()

    futures = policy.broadcast_weights_for_collective(
        kv_scales={"k_scale": 1.25},
        buffer_size_bytes=1024**3,
        num_buffers=2,
    )

    assert futures == ["future"]
    assert calls == [
        (
            "broadcast_weights_for_collective",
            {
                "kv_scales": {"k_scale": 1.25},
                # Forwarded unconditionally alongside the packing options, and asserted
                # here because this test pins the exact kwarg set: every worker has to
                # accept all of it, and one that does not fails at the Ray boundary
                # rather than anywhere near its own signature.
                "refit_timeout_s": None,
                "buffer_size_bytes": 1024**3,
                "num_buffers": 2,
            },
        )
    ]


def test_prepare_refit_info_accepts_identical_worker_manifests(monkeypatch):
    manifest = {
        "model.layers.0.weight": (torch.Size([4, 8]), torch.bfloat16),
        "model.layers.0.bias": (torch.Size([4]), torch.float32),
    }

    class WorkerGroup:
        def run_all_workers_single_data(self, method_name, **kwargs):
            assert method_name == "prepare_refit_info"
            assert kwargs == {}
            return [manifest, dict(manifest)]

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr("nemo_rl.models.policy.lm_policy.ray.get", lambda value: value)
    policy = Policy.__new__(Policy)
    policy.worker_group = WorkerGroup()

    assert policy.prepare_refit_info() is manifest


def _policy_with_worker_results(monkeypatch, results):
    """Return a detached policy whose worker group yields ``results`` verbatim."""

    class WorkerGroup:
        def run_all_workers_single_data(self, _method_name, **_kwargs):
            return results

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr("nemo_rl.models.policy.lm_policy.ray.get", lambda value: value)
    policy = Policy.__new__(Policy)
    policy.worker_group = WorkerGroup()
    return policy


def test_prepare_refit_info_rejects_pipeline_rank_key_mismatch(monkeypatch):
    rank_zero = {
        "model.layers.0.mlp.act_fn.alpha_p": (
            torch.Size([1]),
            torch.bfloat16,
        ),
    }
    rank_one = {
        **rank_zero,
        "model.layers.1.mlp.act_fn.beta": (torch.Size([]), torch.bfloat16),
    }

    policy = _policy_with_worker_results(monkeypatch, [rank_zero, rank_one])

    with pytest.raises(
        RefitManifestMismatchError,
        match=r"HF-schema refit.*worker 0.*worker 1.*unexpected: .*act_fn\.beta",
    ):
        policy.prepare_refit_info()


def _nccl_refit_info(*, misc_meta):
    return {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.up_proj.weight",
                    "global_shape": (8, 4),
                    "dtype": "torch.bfloat16",
                }
            ]
        },
        "misc_meta": misc_meta,
    }


def test_prepare_nccl_reshard_refit_info_checks_misc_and_bulk_manifests(
    monkeypatch,
):
    rank_zero = _nccl_refit_info(
        misc_meta={
            "model.layers.0.mlp.act_fn.alpha_p": {
                "shape": [1],
                "dtype": "torch.bfloat16",
            }
        }
    )
    rank_one = _nccl_refit_info(
        misc_meta={
            **rank_zero["misc_meta"],
            "model.layers.1.mlp.act_fn.eps": {
                "shape": [],
                "dtype": "torch.bfloat16",
            },
        }
    )

    policy = _policy_with_worker_results(monkeypatch, [rank_zero, rank_one])

    with pytest.raises(
        RefitManifestMismatchError,
        match=r"NCCL-reshard refit.*unexpected: .*act_fn\.eps",
    ):
        policy.prepare_nccl_reshard_refit_info({}, {}, 2, 1)
