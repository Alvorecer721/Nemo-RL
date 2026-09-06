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
"""The Apertus chat template emits BOS itself, so re-tokenizing it must not add another.

Rendered prompts start with ``<s>`` (id 1). Tokenizing that text with the
``add_special_tokens=True`` default yields ``[1, 1, ...]`` -- a second BOS the model
never saw in training. It corrupts silently: no error, no shape change, just a
prompt that is off-distribution by one token.
"""

import pytest
from transformers import AutoTokenizer

from nemo_rl_apertus.tests.conftest import TOKENIZER_PATH


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def test_chat_template_emits_exactly_one_bos(tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert text.startswith(tokenizer.bos_token)
    assert text.count(tokenizer.bos_token) == 1


def test_retokenizing_the_template_must_not_double_bos(tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert ids.count(tokenizer.bos_token_id) == 1
    assert ids[0] == tokenizer.bos_token_id

    # Guard the failure mode itself, so this test still means something if the
    # template ever stops emitting BOS.
    doubled = tokenizer(text, add_special_tokens=True)["input_ids"]
    assert doubled.count(tokenizer.bos_token_id) == 2, (
        "add_special_tokens=True no longer doubles BOS -- the template or tokenizer "
        "changed; re-check every tokenizer() call on Apertus paths."
    )
