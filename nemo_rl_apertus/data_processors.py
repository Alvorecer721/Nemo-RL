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
"""Extensible preference-data processors for (online/offline) DPO.

NeMo-RL's data path has a tokenization seam — the *processor* — a callable
``(datum_dict, task_data_spec, tokenizer, max_seq_length, idx) -> PreferenceDatumSpec``
resolved by name from ``PROCESSOR_REGISTRY``. The stock processors are flat
functions that inline both the format parsing and the tokenization, so adding a
new on-disk format (or a *pretokenized* one that skips tokenization) means
copying a whole function.

This module adds a small class-based base, :class:`PreferenceDataProcessor`, that
keeps that registry contract (instances are callable with the same signature) but
splits the work via Template Method so a new format is just one ``parse`` method:

* ``parse(datum_dict)``        — *abstract, format-specific*: raw row -> (chosen, rejected)
                                  full conversations of canonical turns.
* ``build_message_log(...)``   — *overridable tokenization strategy*: default tokenizes
                                  text via ``get_formatted_message_log``; a pretokenized
                                  subclass returns token ids directly to skip tokenization.
* ``__call__`` / ``_assemble`` — *shared*: build both logs, apply max-seq masking, and
                                  assemble the ``PreferenceDatumSpec``.

Concrete examples below: :class:`RankedCompletionsPreferenceProcessor` (text, parity
with the stock ``preference_preprocessor``) and
:class:`PretokenizedPreferenceProcessor` (consumes per-turn token ids, no tokenizer).
Register an instance with ``MyProcessor(...).register("my_name")`` and point a recipe
at it via ``data.processor: my_name``.

Import-light on purpose (lazy ``nemo_rl`` imports + ``TYPE_CHECKING``) so the base and
the pretokenized path are unit-testable without the full runtime.
"""

from __future__ import annotations

import abc
import json
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from nemo_rl.data.interfaces import (
        LLMMessageLogType,
        PreferenceDatumSpec,
        TaskDataSpec,
        TokenizerType,
    )

# Canonical turn: {role, content} for text, or {role, token_ids[, content]} pretokenized.
Turn = dict[str, Any]


class PreferenceDataProcessor(abc.ABC):
    """Registry-compatible base for offline/online-DPO & RM preference processors.

    An instance *is* a ``TaskDataProcessFnCallable`` (it implements ``__call__`` with
    the registry signature), so ``MyProcessor().register("name")`` plugs it into
    ``PROCESSOR_REGISTRY`` and ``data.processor: name`` selects it. Subclasses
    implement :meth:`parse` for their on-disk format; everything else is shared.
    Override :meth:`build_message_log` to change the tokenization strategy.
    """

    @abc.abstractmethod
    def parse(
        self, datum_dict: dict[str, Any]
    ) -> tuple[list[Turn], list[Turn]]:
        """Parse one raw row into ``(chosen_messages, rejected_messages)``.

        Each is a *full* conversation (shared context + the chosen/rejected final
        turn) as canonical turns: ``{role, content}`` for the default (tokenizing)
        path, or ``{role, token_ids[, content]}`` when paired with a pretokenized
        :meth:`build_message_log`. The chosen conversation is the preferred one.
        """

    def build_message_log(
        self,
        messages: list[Turn],
        tokenizer: "TokenizerType",
        task_data_spec: "TaskDataSpec",
    ) -> "LLMMessageLogType":
        """Turn canonical turns into a tokenized ``message_log`` (per-turn ``token_ids``).

        Default: tokenize text via ``get_formatted_message_log`` (applies the chat
        template, adds BOS/EOS) — identical to the stock preference processor.
        Override to consume pretokenized turns and skip tokenization entirely.
        """
        # Lazy import: pulls the full data stack (and decord) only on the text path.
        from nemo_rl.data.llm_message_utils import get_formatted_message_log

        return get_formatted_message_log(messages, tokenizer, task_data_spec)

    def __call__(
        self,
        datum_dict: dict[str, Any],
        task_data_spec: "TaskDataSpec",
        tokenizer: "TokenizerType",
        max_seq_length: Optional[int],
        idx: int,
    ) -> "PreferenceDatumSpec":
        chosen_messages, rejected_messages = self.parse(datum_dict)
        chosen_log = self.build_message_log(chosen_messages, tokenizer, task_data_spec)
        rejected_log = self.build_message_log(rejected_messages, tokenizer, task_data_spec)
        return self._assemble(chosen_log, rejected_log, max_seq_length, idx)

    def _assemble(
        self,
        chosen_log: "LLMMessageLogType",
        rejected_log: "LLMMessageLogType",
        max_seq_length: Optional[int],
        idx: int,
    ) -> "PreferenceDatumSpec":
        """Shared spec assembly + over-length masking (mirrors ``preference_preprocessor``)."""
        length_chosen = sum(len(m["token_ids"]) for m in chosen_log)
        length_rejected = sum(len(m["token_ids"]) for m in rejected_log)

        loss_multiplier = 1.0
        if max_seq_length is not None and max(length_chosen, length_rejected) > max_seq_length:
            # Truncate to a token stub and mask the pair out (loss_multiplier=0) so it
            # keeps the batch shape but contributes nothing — same as the stock processor.
            for log in (chosen_log, rejected_log):
                cap = min(4, max_seq_length // max(len(log), 1))
                for message in log:
                    message["token_ids"] = message["token_ids"][:cap]
            loss_multiplier = 0.0
            length_chosen = sum(len(m["token_ids"]) for m in chosen_log)
            length_rejected = sum(len(m["token_ids"]) for m in rejected_log)

        return {
            "message_log_chosen": chosen_log,
            "message_log_rejected": rejected_log,
            "length_chosen": length_chosen,
            "length_rejected": length_rejected,
            "loss_multiplier": loss_multiplier,
            "idx": idx,
        }

    def register(
        self, name: str, *, idempotent: bool = False
    ) -> "PreferenceDataProcessor":
        """Register this instance in ``PROCESSOR_REGISTRY`` under ``name`` (returns self).

        Default raises on a duplicate ``name`` (stock ``register_processor`` semantics) to
        catch accidental clobbers. ``idempotent=True`` uses ``setdefault`` instead (no-op if
        already registered) for the auto-called registration hooks that may run every launch.
        """
        from nemo_rl.data.processors import PROCESSOR_REGISTRY, register_processor

        if idempotent:
            PROCESSOR_REGISTRY.setdefault(name, self)
        else:
            register_processor(name, self)
        return self


class RankedCompletionsPreferenceProcessor(PreferenceDataProcessor):
    """Text format ``{context, completions:[{rank, completion}]×2}`` (stock parity).

    Lower ``rank`` = preferred (chosen). Mirrors the stock ``preference_preprocessor``
    and uses the default (tokenizing) :meth:`build_message_log`.
    """

    def parse(self, datum_dict: dict[str, Any]) -> tuple[list[Turn], list[Turn]]:
        completions = datum_dict["completions"]
        if len(completions) != 2:
            raise ValueError("preference data supports exactly two completions")
        rank0, rank1 = completions[0]["rank"], completions[1]["rank"]
        if rank0 == rank1:
            raise NotImplementedError("ties are not supported (equal ranks)")
        chosen, rejected = (
            (completions[0], completions[1])
            if rank0 < rank1
            else (completions[1], completions[0])
        )
        context = datum_dict["context"]
        return context + chosen["completion"], context + rejected["completion"]


class PretokenizedPreferenceProcessor(PreferenceDataProcessor):
    """Pretokenized format: rows already carry per-turn token ids — no tokenizer call.

    Each of the ``chosen``/``rejected`` row keys is a list of turns
    ``{role, token_ids[, content]}`` forming the full conversation (context + final
    turn). The chat template / special tokens must already be baked into ``token_ids``.
    Key names are configurable so it adapts to different on-disk schemas.
    """

    def __init__(
        self,
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        role_key: str = "role",
        token_ids_key: str = "token_ids",
        content_key: str = "content",
    ) -> None:
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.role_key = role_key
        self.token_ids_key = token_ids_key
        self.content_key = content_key

    def parse(self, datum_dict: dict[str, Any]) -> tuple[list[Turn], list[Turn]]:
        return datum_dict[self.chosen_key], datum_dict[self.rejected_key]

    def build_message_log(
        self,
        messages: list[Turn],
        tokenizer: "TokenizerType",
        task_data_spec: "TaskDataSpec",
    ) -> "LLMMessageLogType":
        # Skip tokenization: wrap the supplied per-turn ids into the message_log shape.
        return [
            {
                "role": turn[self.role_key],
                "token_ids": torch.as_tensor(turn[self.token_ids_key], dtype=torch.long),
                "content": turn.get(self.content_key, ""),
            }
            for turn in messages
        ]


# ---------------------------------------------------------------------------
# Tools + thinking aware text processor (correct Apertus developer block + BOS/EOS)
# ---------------------------------------------------------------------------
def _conversation_uses_tools(messages: list[Turn]) -> bool:
    """True if any turn is a tool result or carries tool calls (a field or a content block)."""
    for message in messages:
        if message.get("role") == "tool" or message.get("tool_calls"):
            return True
        content = message.get("content")
        if isinstance(content, dict):
            for block in content.get("blocks", []):
                if isinstance(block, dict) and block.get("type") == "tool_calls":
                    return True
    return False


def _conversation_has_thoughts(messages: list[Turn]) -> bool:
    """True if any turn carries thinking as a structured ``thoughts`` content block.

    Assumes correctly-formatted data: thinking is expressed as a ``{"type": "thoughts"}`` content
    block (or flagged explicitly via ``datum[thinking_key]``). Inline ``<|inner_prefix|>`` strings
    baked into plain-text content are NOT auto-detected — set the flag explicitly for those.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, dict):
            for block in content.get("blocks", []):
                if isinstance(block, dict) and block.get("type") == "thoughts":
                    return True
    return False


def render_preference_pair_with_template(
    messages: list[Turn],
    tokenizer: "TokenizerType",
    tools: Optional[list] = None,
    enable_thinking: Optional[bool] = None,
) -> "LLMMessageLogType":
    """Render a full conversation with the chat template and split into ``[prompt, response]``.

    Renders the WHOLE conversation once via ``apply_chat_template`` (so per-turn tool calls,
    tool results, and thinking *and* the developer block are all rendered by the template),
    passing ``tools`` (→ the 'Tool Capabilities' developer block + definitions) and
    ``enable_thinking`` (→ 'Deliberation: enabled'). It tokenizes the rendered text with
    ``add_special_tokens=False`` — the chat template already emits the BOS (e.g. Apertus
    ``{{ bos_token }}``), so this avoids a duplicate; no EOS is added since the template's own
    turn terminator (e.g. Apertus ``<|assistant_end|>``) is the boundary (a BOS is prepended only
    if the template omitted one). It splits into a 2-entry message log: ``prompt`` (everything up
    to the final turn's generation prompt, masked) and ``response`` (the final assistant turn's
    generated tokens, the only span DPO trains), verifying ``prompt`` is an exact **token-prefix**
    of the full render so no special token is duplicated or dropped.

    The final completion turn must be a SINGLE assistant message (which may itself bundle
    thoughts / tool_calls / tool_outputs / response content blocks). A completion whose final
    reply follows a tool result in a *separate* message (assistant tool_calls → tool → assistant)
    fuses into one assistant block in the Apertus template, has no clean prompt/response split,
    and raises — pretokenize those.
    """
    full_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        enable_thinking=enable_thinking,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tools=tools,
        enable_thinking=enable_thinking,
        tokenize=False,
        add_generation_prompt=True,
    )
    # add_special_tokens=False: the template already emits the BOS (Apertus ``{{ bos_token }}``),
    # so this avoids a duplicate; its ``<|assistant_end|>`` is the turn terminator (no EOS added).
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    # Robustness fallback for tokenizers that do NOT emit the BOS themselves but request one
    # (add_bos_token truthy): prepend exactly one to both sides, keeping the prefix relation. The
    # Apertus template emits its own BOS and leaves add_bos_token unset, so this never fires there.
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if (
        bos_id is not None
        and getattr(tokenizer, "add_bos_token", False)
        and (full_ids.shape[0] == 0 or int(full_ids[0]) != bos_id)
    ):
        bos = torch.tensor([bos_id], dtype=full_ids.dtype)
        full_ids = torch.cat([bos, full_ids])
        prompt_ids = torch.cat([bos, prompt_ids])
    n = int(prompt_ids.shape[0])
    if full_ids.shape[0] <= n or not bool(torch.equal(full_ids[:n], prompt_ids)):
        raise ValueError(
            "render_preference_pair_with_template: the prompt is not a token-prefix of the full "
            "render — the final assistant turn does not append cleanly. This happens when a "
            "completion's final reply follows a tool result in a SEPARATE message (assistant "
            "tool_calls → tool → assistant), which the Apertus template fuses into one assistant "
            "block. Bundle the tool round-trip into the final assistant message's content blocks "
            "(thoughts/tool_calls/tool_outputs/response), or use PretokenizedPreferenceProcessor."
        )
    return [
        {"role": "context", "content": prompt_text, "token_ids": prompt_ids.to(torch.long)},
        {
            "role": str(messages[-1].get("role", "assistant")),
            "content": full_text[len(prompt_text):],
            "token_ids": full_ids[n:].clone().to(torch.long),
        },
    ]


class ToolThinkingPreferenceProcessor(RankedCompletionsPreferenceProcessor):
    """Offline-DPO text processor that renders tools + thinking into the Apertus developer block.

    Same ``{context, completions:[{rank, completion}]×2}`` format as the parent, but it passes the
    tool schemas (``datum[tools_key]``) and an ``enable_thinking`` flag to the chat template so the
    rendered prompt's developer message is correct: ``Tool Capabilities:`` + the tool definitions
    when tools are supplied, and ``Deliberation: enabled`` when the trace contains thinking.
    ``enable_thinking`` comes from ``datum[thinking_key]`` when present, else is auto-detected from a
    ``thoughts`` content block in either completion (inline ``<|inner_prefix|>`` strings are not
    auto-detected — set the flag for those). Tool schemas in ``datum[tools_key]`` may be a list or a
    JSON string (decoded). Tokenization renders the whole conversation and
    splits at the final assistant turn (:func:`render_preference_pair_with_template`), so BOS / turn
    tokens are never duplicated and no spurious EOS is added.

    Raises if the trace uses tools (a ``tool_calls`` field, a ``tool`` turn, or a tool_calls block)
    but no ``datum[tools_key]`` schemas are supplied — otherwise the developer block would declare
    tools disabled while the trace calls them.
    """

    def __init__(self, tools_key: str = "tools", thinking_key: str = "enable_thinking") -> None:
        self.tools_key = tools_key
        self.thinking_key = thinking_key

    def __call__(
        self,
        datum_dict: dict[str, Any],
        task_data_spec: "TaskDataSpec",
        tokenizer: "TokenizerType",
        max_seq_length: Optional[int],
        idx: int,
    ) -> "PreferenceDatumSpec":
        chosen_messages, rejected_messages = self.parse(datum_dict)
        tools = datum_dict.get(self.tools_key) or None
        if isinstance(tools, str):
            # Tool schemas are commonly stored as a JSON string: Arrow/HF datasets cannot unify
            # arbitrary ``parameters`` JSON-schemas across rows into one struct (it would merge every
            # tool's properties into a single polluted struct), so on-disk preference sets serialize
            # ``tools`` per row. Decode to the list ``apply_chat_template`` expects.
            tools = json.loads(tools) or None
        if tools is None and (
            _conversation_uses_tools(chosen_messages)
            or _conversation_uses_tools(rejected_messages)
        ):
            raise ValueError(
                f"preference datum {idx} uses tool calls but carries no tool schemas under "
                f"datum['{self.tools_key}']; add them so the developer block renders 'Tool "
                "Capabilities' + the definitions (else the prompt declares tools disabled)."
            )
        enable_thinking = self._resolve_enable_thinking(
            datum_dict, chosen_messages, rejected_messages
        )
        chosen_log = render_preference_pair_with_template(
            chosen_messages, tokenizer, tools, enable_thinking
        )
        rejected_log = render_preference_pair_with_template(
            rejected_messages, tokenizer, tools, enable_thinking
        )
        return self._assemble(chosen_log, rejected_log, max_seq_length, idx)

    def _resolve_enable_thinking(
        self,
        datum_dict: dict[str, Any],
        chosen_messages: list[Turn],
        rejected_messages: list[Turn],
    ) -> bool:
        flag = datum_dict.get(self.thinking_key)
        if flag is not None:
            return bool(flag)
        return _conversation_has_thoughts(chosen_messages) or _conversation_has_thoughts(
            rejected_messages
        )


def register_offline_dpo_processors() -> None:
    """Register the tools+thinking preference processor additively (idempotent).

    Plugs :class:`ToolThinkingPreferenceProcessor` into ``PROCESSOR_REGISTRY`` under
    ``apertus_tool_thinking_preference`` so an offline-DPO recipe can select it with
    ``data.processor`` (needs ``setup_preference_data`` to honor ``data.processor``).
    """
    ToolThinkingPreferenceProcessor().register(
        "apertus_tool_thinking_preference", idempotent=True
    )
