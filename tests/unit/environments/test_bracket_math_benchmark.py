# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression cases for boxed reward and independent binary success."""

import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.environments.bracket_math_environment import (
    BracketMathEnvConfig,
    BracketMathEnvironment,
)
from nemo_rl.environments.bracket_math_reward import (
    score_marked_answer,
    thinking_mode_compliant,
)
from nemo_rl.utils.config import load_config


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


@pytest.mark.parametrize(
    "response,on_compliant,off_compliant",
    [
        (r"\boxed{42}", False, True),
        ("", False, True),
        (r"<|inner_prefix|>work<|inner_suffix|>\boxed{42}", True, False),
        (" \n<|inner_prefix|>work\n<|inner_suffix|> final ", True, False),
        (r"<|inner_prefix|> <|inner_suffix|>\boxed{42}", False, False),
        (r"<|inner_prefix|>work\boxed{42}", False, False),
        (r"<|inner_suffix|>\boxed{42}", False, False),
        (r"<|inner_suffix|>work<|inner_prefix|>\boxed{42}", False, False),
        ("<|inner_prefix|>work<|inner_suffix|> ", False, False),
        ("final<|inner_prefix|>work<|inner_suffix|>more", False, False),
        (
            "<|inner_prefix|><|inner_prefix|>work<|inner_suffix|><|inner_suffix|>final",
            False,
            False,
        ),
        (
            "<|inner_prefix|>a<|inner_suffix|><|inner_prefix|>b<|inner_suffix|>final",
            False,
            False,
        ),
    ],
)
def test_native_thinking_mode_contract(
    response: str, on_compliant: bool, off_compliant: bool
) -> None:
    assert thinking_mode_compliant(response, enable_thinking=True) is on_compliant
    assert thinking_mode_compliant(response, enable_thinking=False) is off_compliant


@pytest.mark.parametrize("enable_thinking", [False, True])
@pytest.mark.parametrize("coefficient", [0.0, 0.1, 0.3])
def test_mode_penalty_preserves_binary_success_and_changes_grpo_preference(
    enable_thinking: bool, coefficient: float
) -> None:
    responses = [
        r"\boxed{42}",
        r"<|inner_prefix|>work<|inner_suffix|>\boxed{42}",
        r"\boxed{41}",
        r"<|inner_prefix|>work<|inner_suffix|>\boxed{41}",
    ]
    cls = BracketMathEnvironment.__ray_metadata__.modified_class
    env = cls.__new__(cls)
    env.cfg = BracketMathEnvConfig(
        num_workers=1,
        reward="composite",
        answer_marker="boxed",
        enable_thinking=enable_thinking,
        thinking_mode_penalty=coefficient,
    )
    env._worker_counter = itertools.count()
    env.workers = [
        SimpleNamespace(
            verify=SimpleNamespace(
                remote=lambda texts, targets, marker: [
                    score_marked_answer(text, target, marker)
                    for text, target in zip(texts, targets, strict=True)
                ]
            )
        )
    ]
    with patch("nemo_rl.environments.bracket_math_environment.ray.get", lambda x: x):
        result = env.step(
            [[{"role": "assistant", "content": text}] for text in responses],
            [{"ground_truth": "42"}] * len(responses),
        )
    torch.testing.assert_close(
        result.episode_successes, torch.tensor([1.0, 1.0, 0.0, 0.0])
    )
    violations = torch.tensor(
        [1.0, 0.0, 1.0, 0.0] if enable_thinking else [0.0, 1.0, 0.0, 1.0]
    )
    expected_penalty = -coefficient * violations
    total_reward = sum(result.rewards.values())
    torch.testing.assert_close(
        total_reward, torch.tensor([1.1, 1.1, 0.1, 0.1]) + expected_penalty
    )
    if coefficient > 0:
        torch.testing.assert_close(
            result.rewards["reward/thinking_mode_penalty"], expected_penalty
        )
        estimator = GRPOAdvantageEstimator(AdvEstimatorConfig(), ClippedPGLossConfig())
        advantages = estimator.compute_advantage(
            torch.zeros(4, 1, dtype=torch.long), total_reward, torch.ones(4, 1)
        )[:, 0]
        compliant = violations == 0
        assert torch.all(advantages[compliant] > advantages[~compliant])
    else:
        assert "reward/thinking_mode_penalty" not in result.rewards


def test_uniform_mode_penalty_cancels_in_grpo() -> None:
    estimator = GRPOAdvantageEstimator(AdvEstimatorConfig(), ClippedPGLossConfig())
    rewards = torch.tensor([1.1] * 8 + [0.1] * 8)
    prompt_ids, mask = torch.zeros(16, 1, dtype=torch.long), torch.ones(16, 1)
    torch.testing.assert_close(
        estimator.compute_advantage(prompt_ids, rewards, mask),
        estimator.compute_advantage(prompt_ids, rewards - 0.1, mask),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"thinking_mode_penalty": 0.1},
        {"thinking_mode_penalty": -0.1, "enable_thinking": False},
        {"thinking_mode_penalty": float("nan"), "enable_thinking": False},
        {"thinking_mode_penalty": float("inf"), "enable_thinking": False},
        {"thinking_mode_penalty": 0.1, "enable_thinking": "false"},
        {"thinking_mode_penalty": 0.1, "enable_thinking": 1},
        {"thinking_mode_penalty": 0.1, "enable_thinking": False, "reward": "outcome"},
    ],
)
def test_invalid_thinking_mode_penalty_config(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        BracketMathEnvConfig.model_validate(
            dict(num_workers=1, reward="composite", answer_marker="boxed") | overrides
        )


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_mode_penalty_recipe_tracks_chat_template(
    enable_thinking: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("AP_CKPT", "AP_TOKENIZER", "AP_RUN_DIR", "AP_EXPERIMENT_DIR"):
        monkeypatch.setenv(name, "/unused-test-path")
    monkeypatch.setenv("AP_ANSWER_MARKER", "boxed")
    root = Path(__file__).resolve().parents[3]
    config = load_config(
        root / "examples/configs/recipes/llm/"
        "grpo-apertus1p5-70b-16n4g-tp2pp4-gsm8k-2k-thinking-mode.yaml"
    )
    config.policy.tokenizer.chat_template_kwargs.enable_thinking = enable_thinking
    env_cfg = BracketMathEnvConfig.model_validate(
        OmegaConf.to_container(config.env.bracket_math, resolve=True)
    )
    assert env_cfg.enable_thinking is enable_thinking
    assert env_cfg.thinking_mode_penalty == 0.1
