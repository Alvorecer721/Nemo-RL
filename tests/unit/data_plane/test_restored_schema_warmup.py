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
"""Schema warmup against a partition that already holds integer rows.

A TQ checkpoint restore leaves the controller with field metadata whose
dtypes come from the stored rows, while every adapter created afterwards
starts with an empty warmup cache. Re-warming such a partition with float32
placeholders is rejected by the controller with ``dtype mismatch:
existing=torch.int64, incoming=torch.float32``. A client with a cleared
warmup cache stands in for the fresh adapter here.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter


def _unwrap(client):
    return getattr(client, "_inner", client)


def test_reregistration_over_stored_integer_rows_issues_no_placeholder(
    tq_client, monkeypatch
) -> None:
    client = tq_client
    partition_id = "restored-int-schema"
    sample_ids = ["r0", "r1"]
    input_ids = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int64)

    client.register_partition(
        partition_id=partition_id,
        fields=["input_ids"],
        num_samples=len(sample_ids),
        consumer_tasks=["train"],
    )
    try:
        client.put_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            fields=TensorDict({"input_ids": input_ids}, batch_size=[2]),
        )

        _unwrap(client)._warmed_fields.pop(partition_id, None)
        placeholder_puts = []
        real_put = tq_adapter.tq.kv_batch_put

        def spying_put(**kwargs):
            placeholder_puts.append(kwargs)
            return real_put(**kwargs)

        monkeypatch.setattr(tq_adapter.tq, "kv_batch_put", spying_put)
        client.register_partition(
            partition_id=partition_id,
            fields=["input_ids"],
            num_samples=len(sample_ids),
            consumer_tasks=["train"],
        )

        assert placeholder_puts == []
        out = client.get_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            select_fields=["input_ids"],
        )
        assert out["input_ids"].dtype == torch.int64
        assert torch.equal(out["input_ids"], input_ids)
    finally:
        client.clear_samples(sample_ids=sample_ids, partition_id=partition_id)
