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

from nemo_rl.environments.bracket_math_reward import (
    extract_bracket_answer,
    normalize_number,
    score_bracket_math,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("18", "18"),
        (" 18 ", "18"),
        ("18.0", "18"),
        ("1,000", "1000"),
        ("2.5", "2.5"),
        ("-3", "-3"),
        ("$18", "$18"),
        ("18 dollars", "18 dollars"),
        ("inf", "inf"),
    ],
)
def test_normalize_number_matches_the_verl_reference(raw: str, expected: str) -> None:
    assert normalize_number(raw) == expected


def test_extract_bracket_answer_takes_the_last_span() -> None:
    assert extract_bracket_answer("[[[42]]] then [[[ 7 ]]]") == "7"
    assert extract_bracket_answer("no brackets") is None
    assert extract_bracket_answer("[[[1,000.0]]]") == "1000"


def test_correct_answer_earns_outcome_plus_format() -> None:
    score = score_bracket_math("The total is 18. [[[18]]]", "18")
    assert score.outcome == 1.0
    assert score.format == 0.1
    assert score.length_penalty == 0.0
    assert score.reward == pytest.approx(1.1)
    assert score.extracted_answer == "18"


def test_wrong_answer_keeps_only_the_format_bonus() -> None:
    score = score_bracket_math("[[[17]]]", "18")
    assert score.outcome == 0.0
    assert score.reward == pytest.approx(0.1)


def test_missing_brackets_score_zero() -> None:
    score = score_bracket_math("The answer is 18.", "18")
    assert score.reward == 0.0
    assert score.extracted_answer is None


def test_length_penalty_starts_at_350_words_and_saturates() -> None:
    words = " ".join(["w"] * 525) + " [[[18]]]"
    score = score_bracket_math(words, "18")
    assert score.length_penalty == pytest.approx(-0.2 * (526 - 350) / 350)
    saturated = score_bracket_math(" ".join(["w"] * 2000) + " [[[18]]]", "18")
    assert saturated.length_penalty == pytest.approx(-0.2)
    assert saturated.reward == pytest.approx(0.9)


def test_unfinished_deliberation_scores_zero() -> None:
    score = score_bracket_math("<|inner_prefix|>still thinking [[[18]]]", "18")
    assert score.unfinished_deliberation
    assert score.reward == 0.0
    finished = score_bracket_math("<|inner_prefix|>think<|inner_suffix|>[[[18]]]", "18")
    assert not finished.unfinished_deliberation
    assert finished.reward == pytest.approx(1.1)


def test_literal_think_tags_are_not_special() -> None:
    score = score_bracket_math("<think>never closed [[[18]]]", "18")
    assert score.reward == pytest.approx(1.1)
