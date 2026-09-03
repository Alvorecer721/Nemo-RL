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
"""Bracket-answer reward shared with the verl side of the Apertus RL framework benchmark.

Port of ``apertus-benchmarks/reward.py`` (eth-cscs/alps-extended-images): the final
answer is the last ``[[[...]]]`` span, numbers are compared after comma stripping
and float normalisation, a 0.1 format bonus rewards any bracket pair, and a
length penalty of up to -0.2 starts at 350 words. The only deliberate deviation
is the unfinished-deliberation check: Apertus reasons between
``<|inner_prefix|>`` and ``<|inner_suffix|>``, never ``<think>``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

DELIBERATION_PREFIX = "<|inner_prefix|>"
DELIBERATION_SUFFIX = "<|inner_suffix|>"
FORMAT_REWARD = 0.1
OUTCOME_REWARD = 1.0
LENGTH_PENALTY_MAX = 0.2
LENGTH_PENALTY_START_WORDS = 350
LENGTH_PENALTY_SPAN_WORDS = 350

_BRACKET_ANSWER = re.compile(r"\[\[\[(.*?)\]\]\]", re.DOTALL)


@dataclass(frozen=True)
class BracketMathScore:
    reward: float
    outcome: float
    format: float
    length_penalty: float
    extracted_answer: str | None
    unfinished_deliberation: bool


def normalize_number(raw: str) -> str:
    raw = raw.strip().replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(value):
        return str(value)
    return str(int(value)) if value == int(value) else str(value)


def extract_bracket_answer(response: str) -> str | None:
    matches = _BRACKET_ANSWER.findall(response)
    return normalize_number(matches[-1]) if matches else None


def has_unfinished_deliberation(response: str) -> bool:
    return DELIBERATION_PREFIX in response and DELIBERATION_SUFFIX not in response


def score_bracket_math(response: str, ground_truth: str) -> BracketMathScore:
    if has_unfinished_deliberation(response):
        return BracketMathScore(0.0, 0.0, 0.0, 0.0, None, True)
    extracted = extract_bracket_answer(response)
    has_answer = "[[[" in response and "]]]" in response
    format_reward = FORMAT_REWARD if has_answer else 0.0
    outcome = (
        OUTCOME_REWARD
        if extracted is not None and extracted == normalize_number(str(ground_truth))
        else 0.0
    )
    words = len(response.split())
    overflow = (words - LENGTH_PENALTY_START_WORDS) / LENGTH_PENALTY_SPAN_WORDS
    length_penalty = -LENGTH_PENALTY_MAX * min(1.0, max(0.0, overflow))
    return BracketMathScore(
        reward=outcome + format_reward + length_penalty,
        outcome=outcome,
        format=format_reward,
        length_penalty=length_penalty,
        extracted_answer=extracted,
        unfinished_deliberation=False,
    )
