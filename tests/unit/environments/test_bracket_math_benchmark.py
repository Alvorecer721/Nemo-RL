# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression cases for boxed reward and independent binary success."""

import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo_rl.environments.bracket_math_environment import (
    BracketMathEnvConfig,
    BracketMathEnvironment,
)
from nemo_rl.environments.bracket_math_reward import score_marked_answer


@pytest.mark.parametrize(
    "text,target,outcome",
    [
        (r"\boxed{42}", "42", 1),
        (r"\[\boxed{42}\]", "42", 1),
        (r"\boxed{1,234.0}", "1234", 1),
        (r"\boxed{100\text{ apples}}", "100", 1),
        (r"\boxed{42} then \boxed{41}", "42", 0),
        (r"\boxed{42} then \boxed{", "42", 0),
        (r"\boxedP{42}", "42", 0),
        ("[[[42]]]", "42", 0),
        (r"\boxed{[[42]]}", "42", 0),
        (r"<|inner_prefix|>\boxed{42}", "42", 0),
        (r"<|inner_prefix|>\boxed{42}<|inner_suffix|>", "42", 0),
        (r"<|inner_prefix|>\boxed{41}<|inner_suffix|>\boxed{42}", "42", 1),
        (r"<|inner_suffix|>\boxed{42}", "42", 0),
        (r"<|inner_prefix|><|inner_prefix|><|inner_suffix|>\boxed{42}", "42", 0),
        (r"\boxed{-2.5}", "-2.5", 1),
    ],
)
def test_boxed_outcome(text, target, outcome):
    assert score_marked_answer(text, target, "boxed").outcome == outcome


@pytest.mark.parametrize("wrapper", ["text", "textbf", "mathrm", "mathbf"])
@pytest.mark.parametrize(
    "content,target",
    [("42", "42"), ("-2.5", "-2.5"), ("1,234.0", "1234"), (r"\$1,234.0", "1234")],
)
def test_boxed_numeric_wrappers_preserve_answer(
    *, wrapper: str, content: str, target: str
) -> None:
    response = rf"\boxed{{\{wrapper}{{{content}}}}}"
    score = score_marked_answer(response, target, "boxed")
    assert score.extracted_answer == target
    assert score.outcome == 1.0
    assert score.reward == pytest.approx(1.1)
    assert score_marked_answer(response, "999", "boxed").outcome == 0.0


def test_environment_preserves_success_when_composite_reward_is_below_one():
    good = score_marked_answer("work " * 4100 + r"\boxed{42}", "42", "boxed")
    bad = score_marked_answer(r"\boxed{41}", "42", "boxed")
    assert good.reward == pytest.approx(0.9)
    cls = BracketMathEnvironment.__ray_metadata__.modified_class
    env = cls.__new__(cls)
    env.cfg = BracketMathEnvConfig(
        num_workers=1, reward="composite", answer_marker="boxed"
    )
    env._worker_counter = itertools.count()
    env.workers = [
        SimpleNamespace(verify=SimpleNamespace(remote=lambda *args: [good, bad]))
    ]
    messages = [[{"role": "assistant", "content": "response"}]] * 2
    metadata = [{"ground_truth": "42"}] * 2
    with patch("nemo_rl.environments.bracket_math_environment.ray.get", lambda x: x):
        result = env.step(messages, metadata, return_extracted_answer=True)
    torch.testing.assert_close(result.episode_successes, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(sum(result.rewards.values()), torch.tensor([0.9, 0.1]))
    assert result.answers == ["42", "41"]
    assert result.observations[0]["content"] == "Environment: correct"
