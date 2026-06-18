# Copyright (c) 2026, the Apertus project.
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
"""Unit tests for the pluggable online-DPO judge core (no network, no Ray).

The judge HTTP path is exercised by monkeypatching ``openai.AsyncOpenAI`` with a
fake whose ``chat.completions.create`` returns a canned top-logprobs payload, so
the scoring math, aspect averaging, and per-sample mapping are tested end-to-end
without a server.
"""

import math
import types

import pytest

from nemo_rl_apertus.online_judge import (
    DEFAULT_ASPECT_PROMPTS,
    TARGET_TOKENS,
    UltraFeedbackJudge,
    build_judge,
    expected_score,
    format_conversation,
    judge_inputs_from_conversation,
    last_assistant_index,
    probs_from_token_logprobs,
    split_at_last_assistant,
)


# ---------------------------------------------------------------------------
# pure scoring helpers
# ---------------------------------------------------------------------------
def test_probs_one_hot():
    # logprob 0.0 -> exp=1; all others absent (-inf -> 0). Full mass on "4".
    probs = probs_from_token_logprobs({"4": 0.0})
    assert probs["4"] == pytest.approx(1.0)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(probs[t] == 0.0 for t in TARGET_TOKENS if t != "4")


def test_probs_softmax_two_tokens():
    # equal logprobs on "1" and "5" -> 0.5 each
    probs = probs_from_token_logprobs({"1": -1.0, "5": -1.0})
    assert probs["1"] == pytest.approx(0.5)
    assert probs["5"] == pytest.approx(0.5)


def test_probs_ignores_non_target_tokens():
    probs = probs_from_token_logprobs({"5": 0.0, "hello": 5.0, "the": 5.0})
    assert probs["5"] == pytest.approx(1.0)


def test_probs_all_absent_returns_zeros():
    probs = probs_from_token_logprobs({"foo": 0.0})
    assert all(v == 0.0 for v in probs.values())


def test_expected_score_values():
    assert expected_score({t: 0.0 for t in TARGET_TOKENS}) == pytest.approx(0.0)
    assert expected_score(probs_from_token_logprobs({"3": 0.0})) == pytest.approx(3.0)
    # 50/50 on 1 and 5 -> 3.0
    assert expected_score(probs_from_token_logprobs({"1": 0.0, "5": 0.0})) == pytest.approx(3.0)


def test_expected_score_matches_softmax_manual():
    token_logprobs = {"4": math.log(3.0), "5": math.log(1.0)}
    probs = probs_from_token_logprobs(token_logprobs)
    # p4 = 3/4, p5 = 1/4 -> 4*0.75 + 5*0.25 = 4.25
    assert expected_score(probs) == pytest.approx(4.25)


# ---------------------------------------------------------------------------
# prompt formatting
# ---------------------------------------------------------------------------
def test_format_conversation_str():
    assert format_conversation("  hi  ") == "hi"


def test_format_conversation_single_turn():
    assert format_conversation([{"role": "user", "content": " solve x "}]) == "solve x"


def test_format_conversation_multi_turn():
    out = format_conversation(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "final"},
        ]
    )
    assert "CONVERSATION HISTORY" in out
    assert "[USER]: first" in out
    assert "[ASSISTANT]: reply" in out
    assert out.strip().endswith("final")


# ---------------------------------------------------------------------------
# split_at_last_assistant (judge prompt/completion split; multi-turn safe)
# ---------------------------------------------------------------------------
def test_split_at_last_assistant_single_turn():
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    prompt_msgs, completion = split_at_last_assistant(conv)
    assert [m["role"] for m in prompt_msgs] == ["user"]
    assert completion == "a"


def test_split_at_last_assistant_multi_turn_keeps_earlier_assistant():
    conv = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    prompt_msgs, completion = split_at_last_assistant(conv)
    # earlier assistant turn stays in the prompt context; only the last is the completion
    assert [m["role"] for m in prompt_msgs] == ["user", "assistant", "user"]
    assert completion == "a2"


def test_split_at_last_assistant_no_assistant():
    conv = [{"role": "user", "content": "q"}]
    prompt_msgs, completion = split_at_last_assistant(conv)
    assert prompt_msgs == conv
    assert completion == ""


def test_last_assistant_index():
    assert last_assistant_index([{"role": "user", "content": "q"}]) == -1
    conv = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert last_assistant_index(conv) == 3  # the LAST assistant, not the first


# ---------------------------------------------------------------------------
# judge_inputs_from_conversation (env's per-sample extraction; metadata branch)
# ---------------------------------------------------------------------------
def test_judge_inputs_default_split():
    conv = [{"role": "user", "content": "the question"}, {"role": "assistant", "content": "the answer"}]
    prompt_text, completion, images = judge_inputs_from_conversation(conv, None)
    assert prompt_text == "the question"
    assert completion == "the answer"
    assert images is None


def test_judge_inputs_prefers_clean_judge_prompt():
    conv = [
        {"role": "user", "content": "<|im_start|>user rendered<|im_end|>"},  # template-rendered
        {"role": "assistant", "content": "ans"},
    ]
    meta = {"judge_prompt": "clean original prompt"}
    prompt_text, completion, _ = judge_inputs_from_conversation(conv, meta)
    assert prompt_text == "clean original prompt"  # override wins over the rendered turn
    assert completion == "ans"


def test_judge_inputs_judge_prompt_message_list_and_images():
    conv = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    meta = {
        "judge_prompt": [{"role": "user", "content": "clean q"}],
        "judge_images": ["data:image/png;base64,AAAA"],
    }
    prompt_text, completion, images = judge_inputs_from_conversation(conv, meta)
    assert prompt_text == "clean q"  # message-list judge_prompt rendered via format_conversation
    assert completion == "a"
    assert images == ["data:image/png;base64,AAAA"]


# ---------------------------------------------------------------------------
# completion re-decode (reasoning aspects need the stripped thinking delimiters back)
# ---------------------------------------------------------------------------
class _ReDecodeTok:
    """Fake tokenizer that decodes any token_ids to a fixed string WITH thinking markers.

    The judge MUST request skip_special_tokens=False (else the delimiters the rollout already
    stripped stay gone). The leading/trailing whitespace checks the result is stripped.
    """

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "  <|inner_prefix|>reason<|inner_suffix|>answer  "


def test_judge_inputs_no_tokenizer_uses_stripped_content():
    conv = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer", "token_ids": [1, 2, 3]},
    ]
    _, completion, _ = judge_inputs_from_conversation(conv, None)
    assert completion == "answer"  # default path: the (stripped) content, no re-decode


def test_judge_inputs_tokenizer_redecodes_with_special_tokens():
    conv = [
        {"role": "user", "content": "answer"},
        {"role": "assistant", "content": "answer", "token_ids": [1, 2, 3]},
    ]
    _, completion, _ = judge_inputs_from_conversation(conv, None, tokenizer=_ReDecodeTok())
    # re-decoded from token_ids keeping the thinking delimiters (and edge whitespace stripped)
    assert completion == "<|inner_prefix|>reason<|inner_suffix|>answer"


def test_judge_inputs_tokenizer_without_token_ids_falls_back_to_content():
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    _, completion, _ = judge_inputs_from_conversation(conv, None, tokenizer=_ReDecodeTok())
    assert completion == "a"  # no token_ids -> cannot re-decode, use content unchanged


# ---------------------------------------------------------------------------
# construction / config
# ---------------------------------------------------------------------------
def _judge(**overrides):
    cfg = dict(base_url="http://x/v1", api_key="k", model="m")
    cfg.update(overrides)
    return UltraFeedbackJudge(**cfg)


def test_default_aspect_is_helpfulness():
    judge = _judge()
    assert judge.aspects == ("helpfulness",)
    assert judge.aspect_prompts["helpfulness"] == DEFAULT_ASPECT_PROMPTS["helpfulness"]


def test_unknown_aspect_raises():
    with pytest.raises(ValueError, match="Unknown judge aspect"):
        _judge(aspects=("nonsense",))


def test_empty_aspects_raises():
    with pytest.raises(ValueError, match="at least one aspect"):
        _judge(aspects=())


def test_aspect_prompt_override():
    judge = _judge(aspects=("custom",), aspect_prompts={"custom": "RATE {prompt} {completion}"})
    assert judge.aspect_prompts["custom"] == "RATE {prompt} {completion}"


def test_custom_system_prompt():
    judge = _judge(system_prompt="be terse")
    assert judge.system_prompt == "be terse"


def test_from_config_env_fallback(monkeypatch):
    monkeypatch.setenv("JUDGE_BASE_URL", "http://env/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "envkey")
    monkeypatch.setenv("JUDGE_MODEL", "env-model")
    judge = build_judge({"type": "ultrafeedback"})
    assert judge.base_url == "http://env/v1"
    assert judge.api_key == "envkey"
    assert judge.model == "env-model"


def test_from_config_cfg_overrides_env(monkeypatch):
    monkeypatch.setenv("JUDGE_BASE_URL", "http://env/v1")
    judge = build_judge({"type": "ultrafeedback", "base_url": "http://cfg/v1", "model": "m"})
    assert judge.base_url == "http://cfg/v1"


def test_build_judge_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown judge.type"):
        build_judge({"type": "does-not-exist"})


def test_build_judge_defaults_to_ultrafeedback():
    judge = build_judge({"base_url": "http://x/v1", "model": "m"})
    assert isinstance(judge, UltraFeedbackJudge)
    assert judge.aspects == ("helpfulness",)  # __init__ default applied (not duplicated in from_config)


# ---------------------------------------------------------------------------
# multimodal content builder
# ---------------------------------------------------------------------------
def test_build_user_content_text_only():
    judge = _judge()
    assert judge._build_user_content("hi", None) == "hi"
    assert judge._build_user_content("hi", []) == "hi"


def test_build_user_content_with_images():
    judge = _judge()
    content = judge._build_user_content("hi", ["data:image/png;base64,AAAA"])
    assert content[0] == {"type": "text", "text": "hi"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }


# ---------------------------------------------------------------------------
# score() guards
# ---------------------------------------------------------------------------
def test_score_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        _judge().score(["p"], ["c1", "c2"])


def test_score_missing_endpoint_raises():
    judge = UltraFeedbackJudge(base_url=None, api_key="k", model="m")
    with pytest.raises(RuntimeError, match="endpoint not configured"):
        judge.score(["p"], ["c"])


def test_score_empty_returns_empty():
    assert _judge().score([], []) == []


# ---------------------------------------------------------------------------
# end-to-end scoring with a fake OpenAI client
# ---------------------------------------------------------------------------
def _fake_response(token: str):
    top = [types.SimpleNamespace(token=token, logprob=0.0)]
    content = [types.SimpleNamespace(top_logprobs=top)]
    logprobs = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(logprobs=logprobs)
    return types.SimpleNamespace(choices=[choice])


def _install_fake_openai(monkeypatch, token_for):
    """Patch openai.AsyncOpenAI so create() returns a one-hot payload from token_for(messages)."""
    import openai

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        async def _create(self, **kwargs):
            return _fake_response(token_for(kwargs["messages"]))

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)


def test_score_maps_per_sample(monkeypatch):
    # token chosen from the completion text embedded in the user message
    def token_for(messages):
        user = messages[1]["content"]
        return "5" if "GOOD" in user else "1"

    _install_fake_openai(monkeypatch, token_for)
    scores = _judge().score(["p1", "p2"], ["a GOOD answer", "a poor answer"])
    assert scores == pytest.approx([5.0, 1.0])


def test_score_averages_over_aspects(monkeypatch):
    def token_for(messages):
        user = messages[1]["content"]
        if "Helpfulness Assessment" in user:
            return "5"
        if "Instruction Following Assessment" in user:
            return "3"
        return "1"

    _install_fake_openai(monkeypatch, token_for)
    judge = _judge(aspects=("helpfulness", "instruction_following"))
    scores = judge.score(["p"], ["c"])
    assert scores == pytest.approx([4.0])  # mean(5, 3)


def test_score_multi_sample_multi_aspect(monkeypatch):
    # 2 samples x 2 aspects: verify aspect scores are attributed to the right sample
    # (the owners/sums/counts accounting in _score_async).
    def token_for(messages):
        user = messages[1]["content"]
        sample_alpha = "ALPHA" in user
        if "Helpfulness Assessment" in user:
            return "5" if sample_alpha else "2"
        # instruction following
        return "3" if sample_alpha else "4"

    _install_fake_openai(monkeypatch, token_for)
    judge = _judge(aspects=("helpfulness", "instruction_following"))
    scores = judge.score(["p1", "p2"], ["ALPHA answer", "BETA answer"])
    # sample0 = mean(5, 3) = 4.0 ; sample1 = mean(2, 4) = 3.0
    assert scores == pytest.approx([4.0, 3.0])


def test_score_api_error_yields_zero(monkeypatch):
    import openai

    class BoomAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        async def _create(self, **kwargs):
            raise RuntimeError("endpoint down")

    monkeypatch.setattr(openai, "AsyncOpenAI", BoomAsyncOpenAI)
    scores = _judge().score(["p"], ["c"])
    assert scores == pytest.approx([0.0])


def test_score_with_images_passes_content_parts(monkeypatch):
    seen = {}

    def token_for(messages):
        seen["content"] = messages[1]["content"]
        return "4"

    _install_fake_openai(monkeypatch, token_for)
    scores = _judge().score(["p"], ["c"], images=[["data:image/png;base64,AAAA"]])
    assert scores == pytest.approx([4.0])
    # the user content is the multimodal parts list, not a bare string
    assert isinstance(seen["content"], list)
    assert seen["content"][1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# reasoning aspects: thinking_appropriateness + thinking_formatting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("aspect", ["thinking_appropriateness", "thinking_formatting"])
def test_reasoning_aspect_registered_and_resolvable(aspect):
    assert aspect in DEFAULT_ASPECT_PROMPTS
    judge = _judge(aspects=(aspect,))
    assert judge.aspect_prompts[aspect] == DEFAULT_ASPECT_PROMPTS[aspect]
    # carries the standard placeholders so .format(prompt=, completion=) works like any aspect
    assert "{prompt}" in judge.aspect_prompts[aspect]
    assert "{completion}" in judge.aspect_prompts[aspect]


@pytest.mark.parametrize("aspect", ["thinking_appropriateness", "thinking_formatting"])
def test_reasoning_aspects_reference_special_thinking_tokens(aspect):
    # both reasoning rubrics must point the judge at the inner_prefix/suffix (or <think>) span
    rubric = DEFAULT_ASPECT_PROMPTS[aspect]
    assert "<|inner_prefix|>" in rubric and "<|inner_suffix|>" in rubric
    assert "<think>" in rubric and "</think>" in rubric


def test_thinking_appropriateness_scores_via_fake_judge(monkeypatch):
    def token_for(messages):
        return "4" if "Reasoning Appropriateness Assessment" in messages[1]["content"] else "1"

    _install_fake_openai(monkeypatch, token_for)
    assert _judge(aspects=("thinking_appropriateness",)).score(["p"], ["c"]) == pytest.approx([4.0])


def test_thinking_formatting_scores_via_fake_judge(monkeypatch):
    def token_for(messages):
        return "5" if "Reasoning Formatting Assessment" in messages[1]["content"] else "1"

    _install_fake_openai(monkeypatch, token_for)
    assert _judge(aspects=("thinking_formatting",)).score(["p"], ["c"]) == pytest.approx([5.0])


# ---------------------------------------------------------------------------
# aspect_weights (configurable per-axis weighting of the cross-aspect mean)
# ---------------------------------------------------------------------------
def _two_aspect_token_for(messages):
    user = messages[1]["content"]
    if "Helpfulness Assessment" in user:
        return "5"
    if "Reasoning Appropriateness Assessment" in user:
        return "1"
    return "3"


def test_aspect_weights_default_is_equal_mean(monkeypatch):
    _install_fake_openai(monkeypatch, _two_aspect_token_for)
    judge = _judge(aspects=("helpfulness", "thinking_appropriateness"))  # no weights
    assert judge.score(["p"], ["c"]) == pytest.approx([3.0])  # mean(5, 1)


def test_aspect_weights_weighted_mean(monkeypatch):
    _install_fake_openai(monkeypatch, _two_aspect_token_for)
    judge = _judge(
        aspects=("helpfulness", "thinking_appropriateness"),
        aspect_weights={"helpfulness": 1.0, "thinking_appropriateness": 3.0},
    )
    # (1*5 + 3*1) / (1 + 3) = 2.0  (equal-weight would be 3.0)
    assert judge.score(["p"], ["c"]) == pytest.approx([2.0])


def test_aspect_weights_missing_key_defaults_to_one(monkeypatch):
    _install_fake_openai(monkeypatch, _two_aspect_token_for)
    # only thinking_appropriateness is weighted; helpfulness keeps the implicit 1.0
    judge = _judge(
        aspects=("helpfulness", "thinking_appropriateness"),
        aspect_weights={"thinking_appropriateness": 3.0},
    )
    assert judge.score(["p"], ["c"]) == pytest.approx([2.0])  # (1*5 + 3*1)/4


def test_from_config_forwards_aspect_weights():
    judge = build_judge(
        {
            "base_url": "http://x/v1",
            "model": "m",
            "aspects": ["helpfulness", "thinking_appropriateness"],
            "aspect_weights": {"helpfulness": 1.0, "thinking_appropriateness": 0.3},
        }
    )
    assert judge.aspect_weights == {"helpfulness": 1.0, "thinking_appropriateness": 0.3}


def test_aspect_weights_absent_in_config_is_none():
    judge = build_judge({"base_url": "http://x/v1", "model": "m"})
    assert judge.aspect_weights is None  # default lives in __init__, not duplicated in from_config


def test_aspect_weights_unknown_key_raises():
    # a weight for an aspect that isn't enabled (usually a typo) fails loudly rather than
    # silently no-op'ing the intended re-weighting
    with pytest.raises(ValueError, match="not enabled"):
        _judge(aspects=("helpfulness",), aspect_weights={"helpfullness": 1.0})
