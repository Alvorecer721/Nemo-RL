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
"""Tests for the shared on-policy prefix splice used by OpenAI-compatible servers."""

import pytest

from nemo_rl.models.generation.openai_server_utils import (
    load_generation_eos_token_ids,
    replace_prefix_tokens,
)


def test_load_generation_eos_token_ids_supports_multiple_ids(monkeypatch):
    class _GenerationConfig:
        eos_token_id = [2, 17, 19]

    monkeypatch.setattr(
        "transformers.GenerationConfig.from_pretrained",
        lambda model_path: _GenerationConfig(),
    )

    assert load_generation_eos_token_ids("model-path") == [2, 17, 19]


def test_replace_prefix_tokens_empty_model_prefix_returns_template():
    """Turn 1 has no prior model output; the full template is returned unchanged."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = []
    template_prefix_token_ids = [9, 2]
    template_token_ids = [9, 2, 33, 44]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == template_token_ids


def test_replace_prefix_tokens_missing_eos_in_template_prefix_raises():
    """A template prefix with no EOS has no valid splice boundary and must raise."""

    class _T:
        eos_token_id = 2

        def decode(self, *args, **kwargs):
            pass

    tokenizer = _T()
    model_prefix_token_ids = [7, 2]
    template_prefix_token_ids = [9, 9, 9]
    template_token_ids = [9, 9, 9, 2, 10]
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=model_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=template_token_ids,
        )


def test_replace_prefix_tokens_tokenizer_without_eos_raises():
    """A tokenizer that has no EOS token cannot locate the splice boundary."""

    class _T:
        eos_token_id = None

    tokenizer = _T()
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=[1],
            template_prefix_token_ids=[1, 2],
            template_token_ids=[1, 2],
        )


def test_replace_prefix_tokens_accepts_generation_config_eos_without_tokenizer_eos():
    """Model-defined turn terminators are sufficient when tokenizer EOS is unset."""

    class _T:
        eos_token_id = None

    result = replace_prefix_tokens(
        tokenizer=_T(),
        model_prefix_token_ids=[100, 17],
        template_prefix_token_ids=[9, 17],
        template_token_ids=[9, 17, 77, 88],
        eos_token_ids=[17, 19],
    )

    assert result == [100, 17, 77, 88]


def test_replace_prefix_tokens_counts_all_generation_config_eos_ids():
    """Mixed valid terminators retain the correct multi-turn splice boundary."""

    class _T:
        eos_token_id = 2

    result = replace_prefix_tokens(
        tokenizer=_T(),
        model_prefix_token_ids=[100, 17],
        template_prefix_token_ids=[9, 2, 10, 17],
        template_token_ids=[9, 2, 10, 17, 77, 88],
        eos_token_ids=[2, 17, 19],
    )

    assert result == [100, 17, 77, 88]


def test_replace_prefix_tokens_uses_last_eos_in_template_prefix():
    """When the prefix contains multiple EOS tokens, the splice cuts at the last one."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = [100, 2]
    template_prefix_token_ids = [9, 2, 9, 2]
    template_token_ids = [9, 2, 9, 2, 77, 88]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == [100, 2, 77, 88]


def test_replace_prefix_tokens_qwen3_think_shift_picks_assistant_eos_not_user_eos():
    """Non-strict-prefix: Qwen3 strips <think> from history when the last message is a
    user turn, so the template's prefix region is shorter and a later user-turn EOS
    lands within the first len(template_prefix) positions. The count-based algorithm
    must cut at the assistant EOS and preserve the intervening user turn.
    """

    class _T:
        eos_token_id = 2

        def decode(self, ids, **kwargs):
            return " ".join(str(i) for i in ids)

    tokenizer = _T()
    model_prefix_token_ids = [11, 12, 99, 99, 99, 55, 2]
    template_prefix_token_ids = [11, 12, 88, 88, 88, 56, 2]
    template_token_ids = [11, 12, 56, 2, 70, 71, 2, 40, 41]

    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )

    assert result == [11, 12, 99, 99, 99, 55, 2, 70, 71, 2, 40, 41]
    assert 70 in result and 71 in result
