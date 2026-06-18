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
"""Pluggable LLM-as-judge for Apertus online DPO.

Online DPO generates ``R`` rollouts per prompt and needs a *reward* per rollout so
the driver can pick best = chosen / worst = rejected. That reward comes from a
**judge** served over an OpenAI-compatible HTTP endpoint (two-job architecture:
a separate inference server serves the judge; the training job calls it).

The judge is **pluggable**: the driver depends only on the small :class:`Judge`
protocol (``score(prompts, completions, images=None) -> list[float]``, higher =
better). :func:`build_judge` selects a concrete backend from config
(``judge.type``); :class:`UltraFeedbackJudge` is the first one. New backends
(pairwise LLM, Bradley-Terry reward model, custom HTTP) register in
``JUDGE_REGISTRY`` and drop in via config alone — no driver changes.

:class:`JudgeEnvironment` wraps *any* :class:`Judge` as a NeMo-RL
``EnvironmentInterface`` Ray actor, so the judge slots into
``run_multi_turn_rollout`` via ``task_to_env`` and scores the whole batch
concurrently — the ``rewards`` it returns are what the driver ranks.

``UltraFeedbackJudge`` is ported from the SwissAI online-PO (SPIN) reward
(``active_ultrafeedback_reward.py``); the annotation prompts below are adapted
from the ActiveUltraFeedback project by the LAS Group (ETH Zurich,
https://github.com/lasgroup/ActiveUltraFeedback), full credit to the original
authors. Scoring is **logprob-based**: the judge emits a single ``1``-``5`` token;
we read the first token's top-logprobs, softmax over ``{"1".."5"}`` and take the
expected value ``Σ tokenᵢ·pᵢ`` ∈ [1, 5] — a continuous score — averaged (optionally
weighted, see ``aspect_weights``) over the enabled aspects. ``aspects`` is a free
choice from ``DEFAULT_ASPECT_PROMPTS`` (or config ``aspect_prompts``); beyond the
ActiveUltraFeedback four it adds two reasoning-model axes — ``thinking_appropriateness``
(reasoning length vs difficulty, directness/quality, and the answer following from the
reasoning) and ``thinking_formatting`` (reasoning wrapped in matching open/close thinking
tokens) — which slot in like any other.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# The judge emits one of these as its first (and only) generated token.
TARGET_TOKENS = ["1", "2", "3", "4", "5"]

# --- Annotation prompts (adapted from ActiveUltraFeedback, LAS Group / ETH Zurich) ---
PREFERENCE_ANNOTATION_SYSTEM_PROMPT = """You are an impartial judge. Your role is to critically evaluate the quality of an AI assistant response based on a given criteria. You'll receive an input with two sections, enclosed in tags: <USER_INPUT>...</USER_INPUT> for the task instructions (and any accompanying context, if applicable), and <ASSISTANT_RESPONSE_TO_EVALUATE>...</ASSISTANT_RESPONSE_TO_EVALUATE> for the AI assistant's response.

Carefully read the provided input to understand the task, then assess how well the response fulfills the criteria requirements. If conversation history is present, ensure the response aligns with it; otherwise, evaluate based solely on the instruction. You will be given a scoring rubric below, based on which you should provide a rating from 1 to 5. Your output should only be an integer from 1 to 5. Do not output any additional text or explanations."""

INSTRUCTION_FOLLOWING_ANNOTATION_PROMPT = """You will be doing an Instruction Following Assessment of an AI assistant response.

Carefully read the <USER_INPUT> to assess how well the candidate response fulfills the task requirements. If the input includes a conversation history, the response must align with that context as well as the final instruction.

**Scoring**: Rate the text on a scale of 1 to 5:
1. **Irrelevant**: No alignment.
2. **Partial Focus**: Addresses one aspect poorly.
3. **Partial Compliance**:
    - (1) Meets goal or restrictions, neglecting other.
    - (2) Acknowledges both but slight deviations.
4. **Almost There**: Near alignment, minor deviations.
5. **Comprehensive Compliance**: Fully aligns, meets all requirements.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{completion}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

HONESTY_ANNOTATION_PROMPT = """You will be doing a Honesty and Uncertainty Expression Assessment of an AI assistant response.

Assess how well the candidate response conveys honesty and uncertainty based on the provided <USER_INPUT>. Evaluate if the text's confidence aligns with response correctness, factoring in any conversation context if it is present in the input.

- **Indicators of Uncertainty**:
    1. Weakeners: e.g., 'I guess,' 'probably.'
    2. Refusal to answer.
    3. Verbalized confidence scores: [0, 20] low; (20, 40] uncertain; (40, 60] moderate; (60, 80] leaning confident; (80, 100] high.
- **No uncertainty expression indicate confidence.**
- **Response Correctness**: Align with ground truth, or provide accurate content without fabrication.

**Scoring**: Rate outputs on a scale of 1 to 5:
1. **Confidently Incorrect**: Confident but entirely wrong.
2. **Confident with Significant Mistakes / Unconfident Incorrect**:
- Confident but contains major errors.
- Unconfident and entirely wrong.
3. **Uncertain / 'I Don't Know' / Subtle Mistakes**:
- 'I don't know' or declines.
- confident but contains minor errors.
- Unconfident and contains significant mistakes.
4. **Correct but Uncertain / Expressed Subtle Mistakes**:
- Correct but unconfident.
- Makes subtle mistakes but expresses uncertainty without specifying the exact area of doubt.
5. **Correct and Confident / Precisely Express Uncertainty**:
- Correct and confident.
- Makes mistakes, but precisely acknowledges minor errors and indicates uncertainty on potential mistakes.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{completion}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

TRUTHFULNESS_ANNOTATION_PROMPT = """You will be doing a Truthfulness and Hallucination Assessment of an AI assistant response.

Evaluate the candidate response's accuracy in providing information without introducing misleading or fabricated details.

When evaluating truthfulness, consider the following types of hallucination:
1. **Contradictory with the World (Factual Error)**: Entities, locations, concepts, or events that conflict with established knowledge.
2. **Contradictory with Instruction/Context**: Responses diverge, introducing new facts not aligned with the user's instructions (or conversation history, if provided).
3. **Self-Contradictory / Logical Error**: Responses contain internal contradictions or logical errors within each independent text.

Reflect on whether any of these hallucination types are present in the response, and take them into account when assigning your rating.

**Scoring**: Rate outputs on a scale of 1 to 5 based on extent of hallucination:
1. **Completely Hallucinated**: Entirely unreliable due to hallucinations.
2. **Severe Hallucination**: Nearly half contains hallucinations, severe deviation from main points.
3. **Partial Hallucination / Misunderstanding**: Overall truthful, partial misunderstanding due to hallucinations.
4. **Insignificant Hallucination**: Mostly truthful, slight hallucination not affecting main points.
5. **No Hallucination**: Free of hallucinations.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{completion}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

HELPFULNESS_ANNOTATION_PROMPT = """You will be doing an Informativeness / Helpfulness Assessment of an AI assistant response.

Evaluate if the candidate response fulfills the task objectives, provides high-quality, correct, and informative content, and respects any preceding conversation context if provided in the input.

Helpfulness assessment emphasizes **Overall Quality** regarding correctness and informativenss.

**Correctness**: Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication.

When assessing informativeness, consider the following aspects:
1. **Clarity and Relevance**: Does the response relate to the task and seek clarifications if needed?
2. **Useful and Comprehensive Information**: Does it provide relevant background, reasoning steps, or detailed description?
3. **Not Lengthy, No Repetition**: Is the response concise, avoiding verbosity or repetition?

Score on a scale of 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:
1. **Severely Incorrect**: Contains significant inaccuracies or fabricated content, even if comprehensive information is provided.
2. **Partially Incorrect**: Contains errors that may cause confusion, even though comprehensive information is present.
3. **Correct**: Accurate and provides useful information that meets the task's requirements.
4. **Highly Informative**: Accurate and extensive, providing valuable insights and detailed information.
5. **Outstandingly Helpful**: Both accurate and in-depth, offering profound insights and comprehensive information.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{completion}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

# In the Apertus chat template the assistant's reasoning is wrapped in the special thinking
# tokens <|inner_prefix|> … <|inner_suffix|> (the model's own bundled template instead uses
# <think> … </think>); both reasoning aspects below tell the judge to look for that span.
THINKING_APPROPRIATENESS_ANNOTATION_PROMPT = """You will be doing a Reasoning Appropriateness Assessment of an AI assistant response.

The assistant's reasoning (its visible thinking / deliberation / chain-of-thought) appears between the special thinking tokens `<|inner_prefix|>` … `<|inner_suffix|>` (or, with some templates, `<think>` … `</think>`), followed by its final answer. Assess that reasoning on three dimensions together:

1. **Length vs difficulty**: First gauge the difficulty of the task in the <USER_INPUT> (trivial recall or a single obvious step → low; multi-step, subtle, ambiguous, or open-ended → high). The amount of reasoning should be **proportionate** — little or none for easy problems, just enough (never padded) for hard ones.
2. **Quality**: The reasoning should be **straight and goal-oriented** — every step advancing toward the answer, with no rambling, restating of the question, hedging, circular or repeated steps, dead-ends, or self-contradiction.
3. **Consistency with the answer**: The final answer should **follow from and agree with** the reasoning — it uses the conclusion the reasoning reached, with no contradiction between the trace and the answer, and without ignoring or discarding the work. (This is the *faithfulness* of the answer to the reasoning, NOT whether the answer is objectively correct.)

- **Under-reasoning**: a hard problem with little or no working, skipped steps, or an unjustified leap straight to the answer.
- **Over-reasoning / low quality**: padded, repetitive, circular, meandering, or backtracking deliberation that does not advance toward the answer.
- **Inconsistent**: the final answer contradicts, ignores, or does not follow from the reasoning's conclusion (e.g. the working derives one result but the answer states another).
- **Incomplete / never answered**: the reasoning runs on without converging, or breaks off mid-thought (e.g. generation was cut off / truncated), so **no final answer is ever produced** — the worst form of over-reasoning, since the whole point of the reasoning is to reach an answer.
- **Well-calibrated**: depth matches difficulty, the reasoning is direct and goal-oriented, AND the final answer follows cleanly from it — every step earns its place, with no padding, no unsupported jumps, and no trace↔answer mismatch.

Judge the reasoning's length, quality, and how faithfully the final answer follows from it — but NOT whether the answer is objectively correct (other criteria cover correctness), and not how the thinking is delimited (the thinking_formatting criterion covers that).

**Scoring**: Rate the response on a scale of 1 to 5:
1. **Severely Inappropriate**: Grossly over- or under-reasoned, rambling/circular throughout, an answer that flatly contradicts or ignores the reasoning, or reasoning that **never reaches a final answer** (e.g. it runs until cut off mid-thought).
2. **Poor**: Clearly too much or too little reasoning for the difficulty, substantial padding / repetition / missing steps, or an answer that only loosely matches the reasoning.
3. **Acceptable**: Roughly proportionate and mostly on-track, with some verbosity or a few missing steps, and an answer broadly consistent with the reasoning.
4. **Good**: Length fits the difficulty, the reasoning is largely direct, and the answer follows from it; only minor excess, omission, or detours.
5. **Excellent**: Depth precisely matches the difficulty, every step is tight and goal-oriented, and the final answer follows cleanly and consistently from the reasoning — no padding, gaps, or trace↔answer mismatch.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{completion}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

THINKING_FORMATTING_ANNOTATION_PROMPT = """You will be doing a Reasoning Formatting Assessment of an AI assistant response.

When the assistant reasons, its thinking must be wrapped in the special thinking tokens: it **opens** with `<|inner_prefix|>` and **closes** with `<|inner_suffix|>` (or, with some templates, opens with `<think>` and closes with `</think>`). Check ONLY that this delimiting is well-formed — not the content, length, or quality of the reasoning (other criteria cover those):

- A **matching opening AND closing** marker are both present (no missing or stray half of the pair). A block that opens but is **never closed** — e.g. the response was cut off / truncated mid-reasoning — fails this.
- The reasoning sits **inside** the markers and the final answer sits **outside** (after the closing marker) — reasoning does not leak past the close, and the answer is not trapped inside. A response that ends while still inside the thinking block (no closing marker, no answer after it) is broken.
- The correct tokens are used, once, and not mismatched (e.g. an `<|inner_prefix|>` opener closed by `</think>`) or wrongly nested.

If the response contains **no reasoning at all** (a direct answer with no thinking section), there is nothing to delimit — treat that as correctly formatted (rate 5).

**Scoring**: Rate the response on a scale of 1 to 5:
1. **Broken**: Reasoning is present but has no markers, or only an opening or only a closing marker (unmatched pair) — including reasoning that opens but is never closed (cut off mid-thought before the `<|inner_suffix|>` / `</think>`), so the closing token and the final answer never appear.
2. **Badly Malformed**: Markers present but seriously misused — mismatched opener/closer tokens, reasoning leaking outside, or the final answer trapped inside the thinking section.
3. **Partially Correct**: Both markers present but with a minor structural slip (e.g. stray text at a boundary, or a small leak across a marker).
4. **Well Formed**: Correct matching open/close markers with the reasoning inside and the answer outside; only cosmetic imperfections.
5. **Perfectly Formed**: Reasoning cleanly opened and closed with the correct matching markers and the answer fully outside — or no reasoning section at all (nothing to delimit).

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{completion}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

DEFAULT_ASPECT_PROMPTS: dict[str, str] = {
    "instruction_following": INSTRUCTION_FOLLOWING_ANNOTATION_PROMPT,
    "honesty": HONESTY_ANNOTATION_PROMPT,
    "truthfulness": TRUTHFULNESS_ANNOTATION_PROMPT,
    "helpfulness": HELPFULNESS_ANNOTATION_PROMPT,
    # Reasoning reward axes (opt-in; for thinking models). Enable by adding the key to
    # env.online_dpo_judge.aspects. Both read the reasoning trace straight from the completion:
    #   thinking_appropriateness — length vs difficulty + directness/quality + answer follows from it
    #   thinking_formatting      — reasoning is wrapped in matching open/close thinking tokens
    "thinking_appropriateness": THINKING_APPROPRIATENESS_ANNOTATION_PROMPT,
    "thinking_formatting": THINKING_FORMATTING_ANNOTATION_PROMPT,
}

# Aspects whose rubric inspects the policy's reasoning trace (the span between the special
# thinking tokens). The rollout decodes the assistant content with skip_special_tokens=True,
# which strips those delimiters; when any of these is enabled the judge env re-decodes the
# completion keeping special tokens (see JudgeEnvironment / judge_inputs_from_conversation),
# and the entry point hands the env the policy tokenizer to do so.
REASONING_ASPECTS = ("thinking_appropriateness", "thinking_formatting")


# =======================================================
# Pure scoring helpers (no network — unit-testable)
# =======================================================
def format_conversation(prompt_data: Any) -> str:
    """Render a prompt (str or message list) into the judge's ``<USER_INPUT>`` text.

    A single-turn prompt is its content; a multi-turn prompt is rendered with an
    explicit conversation-history section so the judge can weigh context. Mirrors
    the SPIN reward's ``_format_prompt_input``.
    """
    if isinstance(prompt_data, str):
        return prompt_data.strip()
    if isinstance(prompt_data, list):
        if len(prompt_data) == 1:
            return str(prompt_data[0].get("content", "")).strip()
        formatted = "### CONVERSATION HISTORY ###\n"
        for turn in prompt_data[:-1]:
            role = str(turn.get("role", "")).upper()
            content = str(turn.get("content", "")).strip()
            formatted += f"[{role}]: {content}\n\n"
        formatted += "### FINAL INSTRUCTION ###\n"
        formatted += str(prompt_data[-1].get("content", "")).strip()
        return formatted
    return str(prompt_data)


def last_assistant_index(conversation: list[dict[str, Any]]) -> int:
    """Index of the last assistant turn in a conversation, or -1 if there is none.

    The single source of truth for "the last assistant turn" — shared by the judge
    (the span it scores) and the DPO loss (the span it trains, via
    ``_trim_to_last_assistant``) so the two can never drift apart.
    """
    last = -1
    for i, message in enumerate(conversation):
        if message.get("role") == "assistant":
            last = i
    return last


def split_at_last_assistant(
    conversation: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Split a conversation at its last assistant turn for judging.

    Returns ``(prompt_messages, completion_text)``: ``prompt_messages`` is the full
    context up to (excluding) the last assistant turn — keeping any *earlier*
    assistant turns as context — and ``completion_text`` is that last assistant
    turn's content (the exact span the DPO loss trains on, via
    ``only_unmask_final``). Multi-turn safe; for single-turn it is just
    (prompt, response). Returns ``(conversation, "")`` if there is no assistant turn.
    """
    last = last_assistant_index(conversation)
    if last < 0:
        return list(conversation), ""
    return list(conversation[:last]), str(conversation[last].get("content", ""))


def _completion_with_special_tokens(tokenizer: Any, token_ids: Any) -> str:
    """Decode an assistant turn keeping special tokens so thinking delimiters survive.

    The rollout sets the assistant ``content`` via a ``skip_special_tokens=True`` decode,
    which strips Apertus' ``<|inner_prefix|>``/``<|inner_suffix|>`` thinking markers. The
    reasoning judge aspects need them, so when the env supplies a tokenizer we re-decode the
    turn's ``token_ids`` with ``skip_special_tokens=False`` instead of using the content.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=False).strip()


def judge_inputs_from_conversation(
    conversation: list[dict[str, Any]],
    meta: Optional[dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
) -> tuple[str, str, Optional[list[str]]]:
    """Derive ``(prompt_text, completion_text, images)`` for one rollout's judge call.

    Splits at the last assistant turn (multi-turn safe): the completion is that
    turn, the prompt is the context before it. A clean prompt the data processor
    stashed in ``meta["judge_prompt"]`` (a str or ``[{role,content}]`` list, like
    the SPIN reference's ``extra_info["prompt"]``) is preferred over the rollout's
    own prompt turns (whose content is the chat-template-rendered string).
    ``meta["judge_images"]`` (optional) supplies multimodal image references.

    When ``tokenizer`` is supplied (the reasoning-aspect path), the completion is
    re-decoded from the last assistant turn's ``token_ids`` keeping special tokens, so
    the thinking delimiters the rollout strips reach the judge; otherwise the (stripped)
    turn content is used unchanged.
    """
    meta = meta or {}
    prompt_messages, completion = split_at_last_assistant(conversation)
    if tokenizer is not None:
        last = last_assistant_index(conversation)
        if last >= 0 and conversation[last].get("token_ids") is not None:
            completion = _completion_with_special_tokens(
                tokenizer, conversation[last]["token_ids"]
            )
    judge_prompt = meta.get("judge_prompt")
    prompt_text = format_conversation(
        judge_prompt if judge_prompt is not None else prompt_messages
    )
    return prompt_text, completion, meta.get("judge_images")


def probs_from_token_logprobs(token_logprobs: dict[str, float]) -> dict[str, float]:
    """Softmax-normalize the first token's top-logprobs over ``TARGET_TOKENS``.

    Returns a probability for each of ``{"1".."5"}``; tokens absent from the
    top-logprobs are treated as ``-inf`` (probability 0). An all-absent payload
    returns all zeros (the response then scores 0.0 — treated as degenerate).
    """
    target_logprobs = {
        tok: token_logprobs.get(tok, -math.inf) for tok in TARGET_TOKENS
    }
    exp_values = [math.exp(lp) for lp in target_logprobs.values()]
    total = sum(exp_values)
    if total == 0:
        return {tok: 0.0 for tok in TARGET_TOKENS}
    return {tok: v / total for tok, v in zip(target_logprobs.keys(), exp_values)}


def expected_score(probs: dict[str, float]) -> float:
    """Expected value ``Σ tokenᵢ·pᵢ`` over ``TARGET_TOKENS`` → a score in [1, 5] (0 if all-zero)."""
    return sum(int(tok) * probs.get(tok, 0.0) for tok in TARGET_TOKENS)


# =======================================================
# Judge interface + factory
# =======================================================
@runtime_checkable
class Judge(Protocol):
    """The only judge contract the online-DPO driver depends on.

    ``score`` returns one scalar per (prompt, completion); higher = better. The
    driver ranks a prompt's ``R`` completions by this score (chosen = argmax,
    rejected = argmin). ``images[i]`` (optional) is a list of image references
    (URLs or ``data:`` URIs) for multimodal judging.

    This is the v1 *pointwise* contract: each (prompt, completion) is scored
    independently. A *pairwise* backend (which must see a prompt's R completions
    together) or a reward-model backend (which may want token ids) will need a
    richer signature — e.g. grouped completions or the raw message logs — so
    extend this protocol rather than overloading the flat string form.
    """

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        images: Optional[list[Optional[list[str]]]] = None,
    ) -> list[float]: ...


class UltraFeedbackJudge:
    """ActiveUltraFeedback LLM-as-judge over an OpenAI-compatible endpoint.

    Scores each completion by the logprob expected-value of the judge's first
    ``1``-``5`` token, averaged (optionally weighted via ``aspect_weights``) over
    the enabled aspects. All judge calls in a batch fire concurrently (one
    ``AsyncOpenAI`` event loop, bounded by a semaphore). API/parse failures yield
    0.0 and are logged, never fatal — the driver masks all-zero (degenerate)
    prompt groups.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str],
        api_key: Optional[str],
        model: Optional[str],
        aspects: tuple[str, ...] = ("helpfulness",),
        system_prompt: Optional[str] = None,
        aspect_prompts: Optional[dict[str, str]] = None,
        aspect_weights: Optional[dict[str, float]] = None,
        max_concurrency: int = 512,
        request_timeout: float = 60.0,
        top_logprobs: int = 20,
        enable_thinking: bool = False,
    ) -> None:
        if not aspects:
            raise ValueError("UltraFeedbackJudge needs at least one aspect")
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.aspects = tuple(aspects)
        self.system_prompt = system_prompt or PREFERENCE_ANNOTATION_SYSTEM_PROMPT
        # Per-aspect weights for the cross-aspect mean; None -> every aspect weighs 1.0
        # (a plain average). Lets a shaping axis (e.g. thinking_appropriateness) count for
        # less than a primary one without a separate judge. Aspects absent from the map keep
        # weight 1.0; the score stays in [1, 5] (a weighted mean of [1, 5] values).
        if aspect_weights:
            # Fail loud on a weight key that isn't an enabled aspect (usually a typo) — it
            # would otherwise be silently ignored and the intended re-weighting never apply.
            unknown_weights = set(aspect_weights) - set(self.aspects)
            if unknown_weights:
                raise ValueError(
                    f"aspect_weights references aspects that are not enabled: "
                    f"{sorted(unknown_weights)}; enabled aspects: {sorted(self.aspects)}"
                )
        self.aspect_weights = aspect_weights
        self.max_concurrency = max_concurrency
        self.request_timeout = request_timeout
        self.top_logprobs = top_logprobs
        self.enable_thinking = enable_thinking

        overrides = aspect_prompts or {}
        resolved: dict[str, str] = {}
        for aspect in self.aspects:
            template = overrides.get(aspect) or DEFAULT_ASPECT_PROMPTS.get(aspect)
            if template is None:
                raise ValueError(
                    f"Unknown judge aspect {aspect!r} with no template override; "
                    f"known aspects: {sorted(DEFAULT_ASPECT_PROMPTS)}"
                )
            resolved[aspect] = template
        self.aspect_prompts = resolved

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "UltraFeedbackJudge":
        """Build from a ``judge`` config dict; connection fields fall back to env vars.

        Only keys actually present in ``cfg`` are forwarded, so ``__init__`` stays the
        single source of every default (no defaults duplicated here).
        """
        # Optional keys whose default lives in __init__; forwarded (coerced) only when set.
        coercions = {
            "aspects": tuple,
            "system_prompt": str,
            "aspect_prompts": dict,
            "aspect_weights": dict,
            "max_concurrency": int,
            "request_timeout": float,
            "top_logprobs": int,
            "enable_thinking": bool,
        }
        kwargs = {
            key: coerce(cfg[key])
            for key, coerce in coercions.items()
            if cfg.get(key) is not None
        }
        return cls(
            base_url=cfg.get("base_url") or os.environ.get("JUDGE_BASE_URL"),
            api_key=cfg.get("api_key") or os.environ.get("JUDGE_API_KEY"),
            model=cfg.get("model") or os.environ.get("JUDGE_MODEL"),
            **kwargs,
        )

    def _build_user_content(
        self, user_prompt: str, sample_images: Optional[list[str]]
    ) -> Any:
        """Text content, or OpenAI multimodal content-parts when images are supplied."""
        if not sample_images:
            return user_prompt
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in sample_images:
            content.append({"type": "image_url", "image_url": {"url": image}})
        return content

    @staticmethod
    def _extract_token_logprobs(res: Any) -> dict[str, float]:
        """Pull the first generated token's ``{token: logprob}`` map from an OpenAI response."""
        first_token_logprobs = res.choices[0].logprobs.content[0].top_logprobs
        return {lp.token: lp.logprob for lp in first_token_logprobs}

    async def _judge_aspect(
        self,
        client: Any,
        aspect: str,
        formatted_input: str,
        completion: str,
        sample_images: Optional[list[str]],
        semaphore: "asyncio.Semaphore",
    ) -> float:
        user_prompt = self.aspect_prompts[aspect].format(
            prompt=formatted_input, completion=completion
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_content(user_prompt, sample_images)},
        ]
        async with semaphore:
            try:
                res = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                    },
                )
            # Judge failures must be non-fatal: a dead endpoint or a malformed
            # reply scores 0.0 (the prompt group degenerates and is masked).
            except Exception as exc:  # noqa: BLE001
                logger.warning("judge aspect=%s call failed: %s", aspect, exc)
                return 0.0
        try:
            token_logprobs = self._extract_token_logprobs(res)
        except (AttributeError, IndexError, TypeError) as exc:
            logger.warning("judge aspect=%s could not parse logprobs: %s", aspect, exc)
            return 0.0
        return expected_score(probs_from_token_logprobs(token_logprobs))

    async def _score_async(
        self,
        prompts: list[str],
        completions: list[str],
        images: Optional[list[Optional[list[str]]]],
    ) -> list[float]:
        # Imported lazily so the module imports even where openai/httpx are absent
        # (e.g. constructing the judge in a no-network unit test path).
        import httpx
        from openai import AsyncOpenAI

        if images is None:
            images = [None] * len(prompts)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        limits = httpx.Limits(max_connections=self.max_concurrency)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout), limits=limits
        ) as http_client:
            client = AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key, http_client=http_client
            )
            tasks = []
            owners = []  # (sample_idx, aspect) per task, to reduce aspect scores per sample
            for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
                for aspect in self.aspects:
                    tasks.append(
                        self._judge_aspect(
                            client, aspect, prompt, completion, images[idx], semaphore
                        )
                    )
                    owners.append((idx, aspect))
            aspect_scores = await asyncio.gather(*tasks)

        # Reduce aspect scores back per sample as a (optionally weighted) mean. With no
        # aspect_weights every aspect weighs 1.0 — the plain average — so this is a no-op
        # for the default config; a weight map down-weights a shaping axis (e.g.
        # thinking_appropriateness) relative to the primary one.
        weights = self.aspect_weights or {}
        weighted_sums = [0.0] * len(prompts)
        weight_totals = [0.0] * len(prompts)
        for (owner, aspect), score in zip(owners, aspect_scores):
            weight = weights.get(aspect, 1.0)
            weighted_sums[owner] += weight * score
            weight_totals[owner] += weight
        return [
            weighted_sums[i] / weight_totals[i] if weight_totals[i] else 0.0
            for i in range(len(prompts))
        ]

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        images: Optional[list[Optional[list[str]]]] = None,
    ) -> list[float]:
        if len(prompts) != len(completions):
            raise ValueError(
                f"prompts ({len(prompts)}) and completions ({len(completions)}) length mismatch"
            )
        if not self.base_url or not self.model:
            raise RuntimeError(
                "Judge endpoint not configured: set judge.base_url/judge.model in the "
                "config or the JUDGE_BASE_URL/JUDGE_MODEL environment variables "
                "(JUDGE_MODEL must equal the server's served-model-name)."
            )
        if not prompts:
            return []
        return asyncio.run(self._score_async(prompts, completions, images))


# Judge backends keyed by ``judge.type``. Add new entries (pairwise, reward_model,
# http_custom, ...) here; nothing else in the pipeline changes.
JUDGE_REGISTRY: dict[str, Any] = {
    "ultrafeedback": UltraFeedbackJudge.from_config,
}


def build_judge(cfg: dict[str, Any]) -> Judge:
    """Instantiate the judge backend named by ``cfg['type']`` (default: ``ultrafeedback``)."""
    judge_type = cfg.get("type", "ultrafeedback")
    if judge_type not in JUDGE_REGISTRY:
        raise ValueError(
            f"Unknown judge.type {judge_type!r}; registered: {sorted(JUDGE_REGISTRY)}"
        )
    return JUDGE_REGISTRY[judge_type](cfg)
