# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
r"""Marked-answer reward shared with the verl side of the Apertus RL framework benchmark.

Port of ``apertus-benchmarks/reward.py`` (eth-cscs/alps-extended-images): the final
answer is the last marker span, numbers are compared after comma stripping and
float normalisation, a 0.1 format bonus rewards any marker, and a length penalty
of up to -0.2 ramps from 2000 to 4000 words. The marker is either ``[[[answer]]]``, the
benchmark's original, or ``\\boxed{answer}``, the form Apertus produces on its
own. Native Apertus deliberation spans are excluded from final-answer extraction;
unfinished or malformed spans cannot receive outcome or format rewards.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

AnswerMarker = Literal["bracket", "boxed"]

FORMAT_REWARD = 0.1
OUTCOME_REWARD = 1.0
LENGTH_PENALTY_MAX = 0.2
LENGTH_PENALTY_START_WORDS = 2000
LENGTH_PENALTY_SPAN_WORDS = 2000

_BRACKET_ANSWER = re.compile(r"\[\[\[(.*?)\]\]\]", re.DOTALL)
_BOXED = "\\boxed"
_LATEX_WRAPPERS = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\{([^{}]*)\}")
_LATEX_NOISE = re.compile(r"\\left|\\right|\\[$%,;!]|[$~]")


@dataclass(frozen=True)
class BracketMathScore:
    reward: float
    outcome: float
    format: float
    length_penalty: float
    extracted_answer: str | None


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


def _last_boxed_content(response: str) -> str | None:
    start = response.rfind(_BOXED + "{")
    if start < 0:
        return None
    open_brace = response.find("{", start)
    if open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[open_brace + 1 : index]
    return None


def _unwrap_numeric_latex(match: re.Match[str]) -> str:
    """Keep wrapped numbers while dropping textual unit suffixes."""
    content = match.group(1)
    try:
        float(content.replace(",", ""))
    except ValueError:
        return ""
    return content


def extract_boxed_answer(response: str) -> str | None:
    content = _last_boxed_content(response)
    if content is None:
        return None
    content = _LATEX_NOISE.sub("", content)
    content = _LATEX_WRAPPERS.sub(_unwrap_numeric_latex, content)
    return normalize_number(content)


def has_marker(response: str, marker: AnswerMarker) -> bool:
    if marker == "bracket":
        return "[[[" in response and "]]]" in response
    return _BOXED + "{" in response


def extract_answer(response: str, marker: AnswerMarker) -> str | None:
    if marker == "bracket":
        return extract_bracket_answer(response)
    return extract_boxed_answer(response)


def completed_final_text(response: str) -> str | None:
    """Return text outside completed native deliberation, rejecting bad spans."""
    visible = []
    in_thinking = False
    for part in re.split(r"(<\|inner_prefix\|>|<\|inner_suffix\|>)", response):
        if part == "<|inner_prefix|>":
            if in_thinking:
                return None
            in_thinking = True
        elif part == "<|inner_suffix|>":
            if not in_thinking:
                return None
            in_thinking = False
        elif not in_thinking:
            visible.append(part)
    return None if in_thinking else "".join(visible)


def score_marked_answer(
    response: str, ground_truth: str, marker: AnswerMarker = "bracket"
) -> BracketMathScore:
    final_text = completed_final_text(response)
    extracted = extract_answer(final_text, marker) if final_text is not None else None
    format_reward = (
        FORMAT_REWARD
        if final_text is not None and has_marker(final_text, marker)
        else 0.0
    )
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
    )
