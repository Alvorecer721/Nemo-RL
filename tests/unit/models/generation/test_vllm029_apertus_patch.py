# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the Apertus loader patch across supported upstream call layouts."""

import logging
from collections.abc import Iterable

import pytest
import torch

from nemo_rl.models.generation.vllm import patches


@pytest.mark.parametrize(
    "call",
    ["AutoWeightsLoader(self)", "AutoWeightsLoader(\n            self,\n        )"],
)
def test_static_constants_are_validated_and_not_refitted(tmp_path, monkeypatch, call):
    source = tmp_path / "apertus.py"
    source.write_text(
        "class Model(torch.nn.Module):\n"
        "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
        f"        loader = {call}\n"
        "        return loader.load_weights(weights)\n"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(source))
    patches._patch_vllm_apertus_static_xielu_loader(logging.getLogger(__name__))
    once = source.read_text()
    patches._patch_vllm_apertus_static_xielu_loader(logging.getLogger(__name__))
    assert source.read_text() == once

    received = {}

    class Loader:
        def __init__(self, model):
            self.model = model

        def load_weights(self, weights):
            received.update(weights)
            return set(received)

    namespace = dict(torch=torch, Iterable=Iterable, AutoWeightsLoader=Loader)
    exec(compile(once, str(source), "exec"), namespace)
    model = namespace["Model"]()
    model.activation = torch.nn.Module()
    model.activation.register_buffer("beta", torch.tensor(0.5), persistent=False)
    weight = torch.tensor([3.0])
    assert model.load_weights(
        [("activation.beta", torch.tensor(0.5)), ("weight", weight)]
    ) == {"weight"}
    assert received["weight"] is weight
    with pytest.raises(ValueError, match="architecture constant"):
        model.load_weights([("activation.beta", torch.tensor(0.75))])
    assert model.activation.beta.item() == 0.5
