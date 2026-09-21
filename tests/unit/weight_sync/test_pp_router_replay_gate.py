# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the functional replay coverage gate."""

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from tests.functional.nccl_reshard_pp import check_complete_routes


@pytest.fixture
def replay_sample():
    routes = torch.full((1, 4, 3, 2), -1, dtype=torch.int32)
    routes[0, :3, 1:] = torch.tensor([1, 6])
    return (
        BatchedDataDict(
            input_ids=torch.tensor([[10, 11, 12, 13]]),
            output_ids=torch.tensor([[10, 11, 12, 13]]),
            unpadded_sequence_lengths=torch.tensor([4]),
            routed_experts=routes,
        ),
        {
            "num_hidden_layers": 3,
            "first_k_dense_replace": 1,
            "num_experts_per_tok": 2,
            "n_routed_experts": 8,
        },
    )


def test_only_dense_layers_and_final_sample_may_lack_routes(replay_sample):
    assert check_complete_routes(*replay_sample) == {
        "complete_expert_ids": 12,
        "missing_expert_ids": 0,
    }


@pytest.mark.parametrize("dtype, num_experts", [(torch.int8, 128), (torch.int16, 256)])
def test_narrow_route_dtype_accepts_largest_valid_expert(
    replay_sample, dtype, num_experts
):
    result, config = replay_sample
    config["n_routed_experts"] = num_experts
    result["routed_experts"] = result["routed_experts"].to(dtype)
    result["routed_experts"][0, 0, 1] = torch.tensor([1, num_experts - 1])
    assert check_complete_routes(result, config)["complete_expert_ids"] == 12


@pytest.mark.parametrize("ids", [[-1, -1], [1, 8], [1, 1]])
@pytest.mark.parametrize("layer", [1, 2])
def test_missing_invalid_or_duplicate_routes_fail_on_every_stage(
    replay_sample, ids, layer
):
    result, config = replay_sample
    result["routed_experts"][0, 0, layer] = torch.tensor(ids)
    with pytest.raises(AssertionError):
        check_complete_routes(result, config)
