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

from nemo_rl.environments.single_turn_verifier_environment import (
    extract_final_answer,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"first \boxed{1}; final \boxed{\frac{2}{3}}", r"\frac{2}{3}"),
        ("Answer: draft\nReasoning\nANSWER: final", "final"),
        ("Candidate 1\nFinal Answer: accepted", "accepted"),
        ("<answer>draft</answer>\n<ANSWER>final\nvalue</ANSWER>", "final\nvalue"),
        ("reasoning\n\nfinal free-form answer\n", "final free-form answer"),
        ("\n\t\n", None),
    ],
)
def test_extract_final_answer_formats(response, expected):
    assert extract_final_answer(response) == expected


def test_extract_final_answer_honors_format_priority():
    response = (
        r"The answer is \boxed{boxed}."
        "\nAnswer: labelled\n<answer>tagged</answer>\nlast"
    )

    assert extract_final_answer(response) == "boxed"
