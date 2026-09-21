# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The stage oracle must survive a transport bug overwriting its source buffer."""

import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


@pytest.mark.parametrize("use_v2", [False, True])
def test_oracle_records_routes_before_worker_transport_can_corrupt_them(
    monkeypatch, use_v2
):
    # Extract the self-contained RPC builder without importing optional Ray
    # actor environments. Execute the actual helper on CPU, not a copied body.
    source = (
        Path(__file__).parents[4]
        / "tests/functional/pipeline_routed_experts_workers.py"
    )
    builder = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "make_route_observer"
    )
    namespace = {"Any": Any, "Callable": Callable}
    exec(
        compile(ast.Module(body=[builder], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    class MoERunner:
        layer_id = 1

    class CaptureSource:
        pass

    modules = {
        "vllm.distributed": {
            "get_pp_group": lambda: SimpleNamespace(rank_in_group=1),
            "get_tp_group": lambda: SimpleNamespace(rank_in_group=0),
        },
        "vllm.model_executor.layers.fused_moe.layer": {"MoERunner": MoERunner},
        "vllm.model_executor.layers.fused_moe.routed_experts_capturer": {
            "RoutedExpertsCaptureSource": CaptureSource
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    buffer = torch.zeros((3, 2, 2), dtype=torch.int32)
    actual = torch.tensor([[1, 3], [2, 4]])

    def forward(_scheduler, _intermediate=None):
        buffer[:2, 1] = actual
        return None if use_v2 else SimpleNamespace(tensors={"hidden_states": 1})

    tokens = torch.tensor([100, 101, 0])
    positions = torch.tensor([0, 1, 0])
    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: [MoERunner()]),
        routed_experts_capturer=SimpleNamespace(get_device_buffer=lambda: buffer),
        input_ids=SimpleNamespace(gpu=tokens),
        positions=positions,
        input_buffers=SimpleNamespace(input_ids=tokens, positions=positions),
        execute_model=forward,
    )

    def corrupt_after_forward(scheduler):
        output = runner.execute_model(scheduler, None)
        buffer[:2, 1] = 0
        return output

    worker = SimpleNamespace(
        model_runner=runner,
        use_v2_model_runner=use_v2,
        execute_model=corrupt_after_forward,
    )
    namespace["make_route_observer"]()(worker)
    worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=2))
    observed = worker._test_route_observation["records"][0]
    assert observed["routes"] == [[[1, 3]], [[2, 4]]]
    assert observed["token_ids"] == [100, 101]
    assert observed["positions"] == [0, 1]
    assert not buffer[:2, 1].any()
