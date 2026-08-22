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

"""Unit tests for the rl_prompt GRPO data processor.

These exercise the load-bearing contract — that an already-tokenized omni doc is
passed through as ``token_ids`` with no re-tokenization, and that
``answer``/``answer_variants`` reach ``extra_env_info`` — without needing megatron
or a real MMIDIDX store: the store-open is monkeypatched to a fake.
"""

import numpy as np
import torch

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class _FakeStore:
    def __init__(self, docs):
        self._docs = docs

    def __getitem__(self, i):
        return self._docs[i]


def _row(**over):
    row = {
        "store_prefix": "/fake/train",
        "doc_index": 0,
        "ground_truth": "42",
        "answer_variants": ["42", "forty-two"],
        "task_name": "rl_prompt",
    }
    row.update(over)
    return row


def test_processor_passes_token_ids_through(monkeypatch):
    doc = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    monkeypatch.setattr(
        processors, "_open_rl_prompt_store", lambda prefix: _FakeStore({0: doc})
    )
    out = processors.mmididx_grpo_data_processor(
        _row(), TaskDataSpec(task_name="rl_prompt"), None, 100, 7
    )
    tok = out["message_log"][0]["token_ids"]
    assert isinstance(tok, torch.Tensor) and tok.dtype == torch.long
    assert tok.tolist() == [1, 2, 3, 4, 5]
    assert out["message_log"][0]["role"] == "user"
    assert out["length"] == 5
    assert out["loss_multiplier"] == 1.0
    assert out["idx"] == 7
    assert out["task_name"] == "rl_prompt"
    assert out["extra_env_info"] == {
        "ground_truth": "42",
        "answer_variants": ["42", "forty-two"],
    }


def test_processor_unwraps_multimodal_tuple(monkeypatch):
    doc = np.array([9, 8, 7], dtype=np.int32)
    monkeypatch.setattr(
        processors, "_open_rl_prompt_store", lambda prefix: _FakeStore({0: (doc, None)})
    )
    out = processors.mmididx_grpo_data_processor(
        _row(), TaskDataSpec(task_name="rl_prompt"), None, 100, 0
    )
    assert out["message_log"][0]["token_ids"].tolist() == [9, 8, 7]


def test_processor_masks_oversized_prompt(monkeypatch):
    doc = np.arange(50, dtype=np.int32)
    monkeypatch.setattr(
        processors, "_open_rl_prompt_store", lambda prefix: _FakeStore({0: doc})
    )
    out = processors.mmididx_grpo_data_processor(
        _row(), TaskDataSpec(task_name="rl_prompt"), None, 10, 0
    )
    assert out["loss_multiplier"] == 0.0
    assert out["length"] == 50  # reported length is the true length
    assert len(out["message_log"][0]["token_ids"]) <= 4


def test_processor_registered():
    assert "mmididx_grpo_data_processor" in processors.PROCESSOR_REGISTRY
    assert (
        processors.PROCESSOR_REGISTRY["mmididx_grpo_data_processor"]
        is processors.mmididx_grpo_data_processor
    )
