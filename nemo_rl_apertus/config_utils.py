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
"""Small config helpers shared by the Apertus entry points."""

from __future__ import annotations

from typing import Any


def default_tokenizer_to_model(policy_cfg: dict[str, Any]) -> None:
    """Default ``policy.tokenizer.name`` to ``policy.model_name`` when unset (mutates in place).

    Most HF checkpoints bundle their tokenizer, so this lets a recipe omit the ``tokenizer`` block
    and load it from the model dir (``get_tokenizer`` itself requires ``tokenizer.name``). It prints
    a **warning** when it triggers: for Apertus the checkpoint may ship a *stale* chat template (e.g.
    ``<think>`` thinking markers and no ``tool.function`` unwrap, so OpenAI-nested tool specs fail),
    so the recipes pin a fixed tokenizer explicitly — rely on this fallback only when the
    checkpoint's bundled tokenizer is known-good.
    """
    tokenizer_cfg = policy_cfg.setdefault("tokenizer", {})
    if not tokenizer_cfg.get("name"):
        tokenizer_cfg["name"] = policy_cfg["model_name"]
        print(
            "⚠ policy.tokenizer.name unset — defaulting to policy.model_name "
            f"({tokenizer_cfg['name']}); using the checkpoint's bundled tokenizer/chat-template. "
            "For Apertus this may be a stale template — pin the fixed tokenizer snapshot to be safe."
        )


def _prepend_empty_system(conversation: list) -> list:
    """Prepend an empty system turn unless the conversation already starts with one."""
    if (
        conversation
        and isinstance(conversation[0], dict)
        and conversation[0].get("role") == "system"
    ):
        return conversation
    return [{"role": "system", "content": ""}, *conversation]


def disable_default_system_prompt(tokenizer: Any) -> Any:
    """Wrap ``apply_chat_template`` to inject an empty system turn when none is present.

    Apertus' chat template auto-injects a default system prompt ("You are Apertus 1.5 Omni …")
    whenever the conversation has no leading ``system`` message. This wrapper instead supplies an
    explicit **empty** system message (``{"role": "system", "content": ""}``) in that case, so the
    template emits an empty ``<|system_start|><|system_end|>`` block rather than the default text.
    Conversations that already start with a system turn are left untouched.

    It wraps the bound method on the tokenizer instance, so every data path that goes through
    ``apply_chat_template`` (the stock ``get_formatted_message_log`` used by binary/preference data,
    the offline tools/thinking processor, and the online prompt processor + rollout prompts) is
    covered uniformly. Idempotent; mutates the tokenizer in place and returns it. Off by default —
    enable per recipe with ``policy.tokenizer.disable_default_system_prompt: true``.
    """
    if getattr(tokenizer, "_apertus_empty_system_wrapped", False):
        return tokenizer
    inner = tokenizer.apply_chat_template

    def _is_conversation(value: Any) -> bool:
        # a single conversation is a list of message dicts (empty list = nothing to do)
        return isinstance(value, list) and (len(value) == 0 or isinstance(value[0], dict))

    def wrapper(conversation: Any = None, *args: Any, **kwargs: Any) -> Any:
        if _is_conversation(conversation):
            conversation = _prepend_empty_system(conversation)
        elif isinstance(conversation, list):  # batched: a list of conversations
            conversation = [
                _prepend_empty_system(c) if _is_conversation(c) else c for c in conversation
            ]
        return inner(conversation, *args, **kwargs)

    tokenizer.apply_chat_template = wrapper
    tokenizer._apertus_empty_system_wrapped = True
    return tokenizer
