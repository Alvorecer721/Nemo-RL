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
"""Prompt-only dataset + processor for online DPO (additive; no upstream edits).

Online DPO starts from a *prompt-only* set: the policy generates ``R`` rollouts per
prompt and the judge ranks them, so there is no chosen/rejected column on disk. The
stock NeMo-RL loaders don't fit such data:

* ``ResponseDataset`` requires both ``input_key`` and ``output_key`` (a response).
* ``openai_format`` asserts the last message is ``assistant``.
* ``math_hf_data_processor`` reads ``messages[1]`` as the math ground truth, so it
  *crashes* on a single-turn prompt and ignores earlier turns of a multi-turn one.

This module adds the missing pieces, matching the reference SwissAI online-PO data
(a parquet whose ``prompt`` column is a list of ``{role, content}`` turns, e.g.
``train_maxmin_online_full.parquet``):

* :func:`prompt_to_messages` — normalize a ``prompt`` cell (str or message list) to turns.
* :class:`PromptOnlyDataset` (built lazily) — a ``RawDataset`` that loads a local
  parquet/jsonl/HF set and emits the canonical ``{messages, task_name}`` row.
* :func:`online_prompt_processor` — tokenize the *full* prompt conversation for
  generation (``add_generation_prompt=True``) and stash the raw prompt in
  ``extra_env_info["judge_prompt"]`` so the judge scores a clean prompt (not the
  chat-template-rendered string — see :mod:`nemo_rl_apertus.online_judge`).
* :func:`register_online_dpo_data` — register both into the stock registries via
  ``setdefault`` (idempotent, no upstream registry edits), like the judge env does.

Import-light (lazy ``nemo_rl`` imports + ``TYPE_CHECKING``) so the pure helpers and the
processor are unit-testable without the full data stack.
"""

from __future__ import annotations

import functools
import random
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec, TokenizerType

# Registry names; select them from a recipe via ``data.*.dataset_name`` / ``data.*.processor``.
PROMPT_ONLY_DATASET_NAME = "prompt_only"
ONLINE_PROMPT_PROCESSOR_NAME = "online_prompt_processor"
DEFAULT_TASK_NAME = "online_prompts"

# Per-prompt policy-thinking modes (the ``online_dpo.thinking.mode`` knob). "default"
# leaves the chat template's own default (prior behavior); "random" decides per prompt.
THINKING_MODES = ("default", "on", "off", "random")
# Row column carrying an explicit per-prompt override (mirrors the offline processor's key).
ENABLE_THINKING_KEY = "enable_thinking"


def _resolve_enable_thinking(
    datum_dict: dict[str, Any], idx: int, thinking_cfg: Optional[dict[str, Any]]
) -> Optional[bool]:
    """Resolve the per-prompt ``enable_thinking`` flag for a rollout prompt.

    Precedence: an explicit per-row ``datum_dict["enable_thinking"]`` wins (mirrors the
    offline ``ToolThinkingPreferenceProcessor``); otherwise the configured
    ``thinking_cfg["mode"]`` decides — ``"on"``/``"off"`` are constant, ``"random"`` is a
    ``Bernoulli(probability)`` drawn deterministically from ``idx`` (+ ``seed``) so it is
    reproducible across runs/resumes, and ``"default"`` (or no config) returns ``None``.

    ``None`` means "omit the ``enable_thinking`` kwarg" → the chat template's own default,
    i.e. the prior behavior. The decision is made once per prompt (the driver later repeats
    a prompt into ``R`` rollouts), so a prompt's ``R`` rollouts share one thinking mode and
    stay an apples-to-apples preference group.
    """
    flag = datum_dict.get(ENABLE_THINKING_KEY)
    if flag is not None:
        return bool(flag)
    if not thinking_cfg:
        return None
    mode = thinking_cfg.get("mode", "default")
    if mode == "default":
        return None
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode == "random":
        probability = float(thinking_cfg.get("probability", 0.5))
        seed = thinking_cfg.get("seed", 0)
        # Seed a per-prompt RNG from (seed, idx) so the draw is stable for a given prompt.
        return random.Random(f"{seed}:{idx}").random() < probability
    raise ValueError(
        f"online_dpo.thinking.mode must be one of {THINKING_MODES}, got {mode!r}"
    )


def prompt_to_messages(prompt: Any) -> list[dict[str, str]]:
    """Normalize a dataset ``prompt`` cell into a list of ``{role, content}`` turns.

    Accepts a bare string (wrapped as a single ``user`` turn) or a list of message
    dicts (the reference parquet form). Roles/contents are coerced to ``str``.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, (list, tuple)):
        return [
            {"role": str(turn.get("role", "user")), "content": str(turn.get("content", ""))}
            for turn in prompt
        ]
    raise TypeError(
        f"prompt must be a str or a list of {{role, content}} dicts, got {type(prompt).__name__}"
    )


def _task_name_from_path(data_path: str) -> str:
    """Derive a task name from the last two path components (mirrors stock ``ResponseDataset``).

    Joins the final dir + file stem with ``-`` and strips a *single* leading ``-`` (the same
    one-character strip stock does — not ``lstrip``), falling back to ``DEFAULT_TASK_NAME``.
    """
    task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
    if task_name and task_name[0] == "-":
        task_name = task_name[1:]
    return task_name or DEFAULT_TASK_NAME


def online_prompt_processor(
    datum_dict: dict[str, Any],
    task_data_spec: "TaskDataSpec",
    tokenizer: "TokenizerType",
    max_seq_length: Optional[int],
    idx: int,
    *,
    thinking_cfg: Optional[dict[str, Any]] = None,
) -> "DatumSpec":
    """Tokenize a prompt-only conversation for generation; stash a clean judge prompt.

    Mirrors ``math_hf_data_processor``'s output shape (a single rendered prompt entry
    in the ``message_log``, over-length pairs masked to ``loss_multiplier=0``), but
    tokenizes the *whole* prompt conversation (multi-turn safe) and sets
    ``extra_env_info["judge_prompt"]`` to the raw turns so the judge env scores a clean
    prompt rather than the chat-template-rendered string.

    ``thinking_cfg`` (the ``online_dpo.thinking`` block, bound onto the registered
    processor by :func:`register_online_dpo_data`) selects the policy's per-prompt
    reasoning toggle: when it resolves to a concrete bool, ``enable_thinking=`` is passed
    to ``apply_chat_template`` (Apertus → the developer block's ``Deliberation:
    enabled/disabled``) and stashed in ``extra_env_info`` for rollout inspection. When it
    resolves to ``None`` the kwarg is omitted (prior behavior). See
    :func:`_resolve_enable_thinking`.
    """
    messages = datum_dict["messages"]

    message_list: list[dict[str, str]] = []
    if getattr(task_data_spec, "system_prompt", None):
        message_list.append({"role": "system", "content": task_data_spec.system_prompt})
    message_list.extend(
        {"role": turn["role"], "content": turn["content"]} for turn in messages
    )

    enable_thinking = _resolve_enable_thinking(datum_dict, idx, thinking_cfg)
    # Only forward the kwarg when a decision was made, so "default" mode (and the no-config
    # path) renders exactly as before — and stays compatible with templates that don't take it.
    template_kwargs: dict[str, Any] = (
        {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    )
    rendered: str = tokenizer.apply_chat_template(
        message_list,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
        **template_kwargs,
    )
    token_ids = tokenizer(
        rendered, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    message_log = [{"role": "user", "content": rendered, "token_ids": token_ids}]

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        # Over the cap: shrink to a token stub and mask (kept for batch shape, ignored by loss).
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    # The judge env prefers this clean prompt over the rendered turn content.
    extra_env_info: dict[str, Any] = {"judge_prompt": messages}
    if enable_thinking is not None:
        # Travels with the rollout so build_rollout_log can record the per-prompt mode;
        # the judge ignores it (it scores the rendered completion, which holds the trace).
        extra_env_info[ENABLE_THINKING_KEY] = enable_thinking

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict.get("task_name", DEFAULT_TASK_NAME),
    }
    return output


# Built once on first use. PromptOnlyDataset subclasses the *runtime* RawDataset, so it
# can't live at module top without making this module import-heavy; defining it in a
# function keeps the pure helpers + processor unit-testable off-cluster.
_PROMPT_ONLY_DATASET_CLS: Optional[type] = None


def _build_prompt_only_dataset_cls() -> type:
    """Build (once, cached) and return the :class:`PromptOnlyDataset` class."""
    global _PROMPT_ONLY_DATASET_CLS
    if _PROMPT_ONLY_DATASET_CLS is not None:
        return _PROMPT_ONLY_DATASET_CLS

    from nemo_rl.data.datasets.raw_dataset import RawDataset
    from nemo_rl.data.datasets.utils import load_dataset_from_path

    class PromptOnlyDataset(RawDataset):
        """Prompt-only loader for online RL: parquet/jsonl/HF with a ``prompt`` column.

        ``prompt`` may be a string or a list of ``{role, content}`` turns (the reference
        ``train_maxmin_online_full.parquet`` form). Emits the canonical
        ``{messages, task_name}`` row that ``online_prompt_processor`` tokenizes.
        """

        def __init__(
            self,
            data_path: str,
            prompt_key: str = "prompt",
            subset: Optional[str] = None,
            split: Optional[str] = None,
            split_validation_size: float = 0,
            seed: int = 42,
            thinking_key: str = ENABLE_THINKING_KEY,
            **kwargs: Any,
        ) -> None:
            self.prompt_key = prompt_key
            self.thinking_key = thinking_key
            self.task_name = _task_name_from_path(data_path)

            self.dataset = load_dataset_from_path(data_path, subset, split)
            self.dataset = self.dataset.map(
                self.format_data, remove_columns=self.dataset.column_names
            )
            # `val_dataset` is set (not None) only if this set is split for validation.
            self.val_dataset = None
            self.split_train_validation(split_validation_size, seed)

        def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
            row: dict[str, Any] = {
                "messages": prompt_to_messages(data[self.prompt_key]),
                "task_name": self.task_name,
            }
            # Carry an optional per-row thinking override (the configurable source column
            # `thinking_key` -> the fixed `enable_thinking` key the processor reads); absent
            # column -> the config mode decides (see _resolve_enable_thinking).
            if data.get(self.thinking_key) is not None:
                row[ENABLE_THINKING_KEY] = bool(data[self.thinking_key])
            return row

    _PROMPT_ONLY_DATASET_CLS = PromptOnlyDataset
    return PromptOnlyDataset


def register_online_dpo_data(thinking_cfg: Optional[dict[str, Any]] = None) -> None:
    """Register the prompt-only dataset + processor additively (idempotent).

    Uses ``setdefault`` into the stock ``DATASET_REGISTRY`` / ``PROCESSOR_REGISTRY`` so
    a recipe can select ``data.*.dataset_name: prompt_only`` and
    ``data.*.processor: online_prompt_processor`` without any upstream registry edit.

    ``thinking_cfg`` (the ``online_dpo.thinking`` block, if any) is bound onto the
    registered processor so each rollout prompt's ``enable_thinking`` is resolved
    per-prompt; ``None`` registers the bare processor (the prior behavior).
    """
    from nemo_rl.data.datasets.response_datasets import DATASET_REGISTRY
    from nemo_rl.data.processors import PROCESSOR_REGISTRY

    processor = (
        functools.partial(online_prompt_processor, thinking_cfg=thinking_cfg)
        if thinking_cfg
        else online_prompt_processor
    )
    DATASET_REGISTRY.setdefault(PROMPT_ONLY_DATASET_NAME, _build_prompt_only_dataset_cls())
    PROCESSOR_REGISTRY.setdefault(ONLINE_PROMPT_PROCESSOR_NAME, processor)
