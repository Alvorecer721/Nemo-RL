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
"""Unit tests for the prompt-only online-DPO data path.

The module is import-light (lazy nemo_rl imports), so the pure prompt normalization
and the processor (which only needs a tokenizer) are testable without the data stack.
A fake tokenizer renders messages to text and tokens. The dataset class + registry
wiring need the real nemo_rl.data stack and are skipped where it isn't importable.
"""

import pytest
import torch

from nemo_rl_apertus.online_data import (
    DEFAULT_TASK_NAME,
    ENABLE_THINKING_KEY,
    _resolve_enable_thinking,
    _task_name_from_path,
    online_prompt_processor,
    prompt_to_messages,
)


class _FakeTok:
    """Renders a message list to text and tokenizes by whitespace.

    Reflects ``enable_thinking`` into the rendered text (``<think>``/``<nothink>``) so a
    test can assert whether — and how — the kwarg was forwarded; the kwarg is only present
    when the caller passes it (default mode omits it, matching the real template seam).
    """

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
        **kwargs,
    ):
        text = " ".join(m["content"] for m in messages)
        if "enable_thinking" in kwargs:
            text += " <think>" if kwargs["enable_thinking"] else " <nothink>"
        return text + (" <gen>" if add_generation_prompt else "")

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([[7] * len(text.split())])}


class _Spec:
    system_prompt = None
    prompt = None


# ---------------------------------------------------------------------------
# prompt_to_messages
# ---------------------------------------------------------------------------
def test_prompt_to_messages_str():
    assert prompt_to_messages("hi there") == [{"role": "user", "content": "hi there"}]


def test_prompt_to_messages_list_multi_turn():
    out = prompt_to_messages(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
    )
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[-1]["content"] == "u2"


def test_prompt_to_messages_coerces_and_defaults_role():
    # missing role defaults to user; content coerced to str
    out = prompt_to_messages([{"content": 123}])
    assert out == [{"role": "user", "content": "123"}]


def test_prompt_to_messages_bad_type_raises():
    with pytest.raises(TypeError):
        prompt_to_messages(42)


# ---------------------------------------------------------------------------
# _task_name_from_path (mirrors stock ResponseDataset's single-'-' strip)
# ---------------------------------------------------------------------------
def test_task_name_from_path():
    assert _task_name_from_path("/data/dir/train_set.parquet") == "dir-train_set"
    assert _task_name_from_path("/foo.parquet") == "foo"  # single leading '-' stripped (not lstrip)
    assert _task_name_from_path("/.parquet") == DEFAULT_TASK_NAME  # empty stem -> default


# ---------------------------------------------------------------------------
# online_prompt_processor
# ---------------------------------------------------------------------------
def test_processor_basic_shape_and_clean_judge_prompt():
    messages = [{"role": "user", "content": "solve x"}]
    datum = {"messages": messages, "task_name": "maxmin"}
    out = online_prompt_processor(datum, _Spec(), _FakeTok(), max_seq_length=100, idx=5)

    assert out["idx"] == 5
    assert out["loss_multiplier"] == 1.0
    assert out["task_name"] == "maxmin"
    # the judge prompt is the RAW turns (clean), not the rendered string
    assert out["extra_env_info"]["judge_prompt"] is messages
    entry = out["message_log"][0]
    assert entry["role"] == "user"
    assert entry["token_ids"].dtype == torch.long
    assert entry["content"].endswith("<gen>")  # add_generation_prompt applied
    assert out["length"] == len(entry["token_ids"])


def test_processor_system_prompt_prepended_but_not_in_judge_prompt():
    class SysSpec:
        system_prompt = "BE TERSE"
        prompt = None

    messages = [{"role": "user", "content": "q"}]
    out = online_prompt_processor({"messages": messages}, SysSpec(), _FakeTok(), 100, 0)
    # system prompt is rendered into the generation context...
    assert "BE TERSE" in out["message_log"][0]["content"]
    # ...but the judge prompt stays the clean original turns (no injected system prompt)
    assert out["extra_env_info"]["judge_prompt"] == messages


def test_processor_overlength_is_masked():
    messages = [{"role": "user", "content": "a b c d e f g h"}]
    out = online_prompt_processor({"messages": messages}, _Spec(), _FakeTok(), max_seq_length=2, idx=0)
    assert out["loss_multiplier"] == 0.0
    # token_ids are truncated to a stub: min(4, max_seq_length // n_messages) = min(4, 2) = 2.
    # (length stays the pre-truncation value, matching math_hf_data_processor.)
    assert len(out["message_log"][0]["token_ids"]) <= 2


def test_processor_default_task_name_when_absent():
    out = online_prompt_processor({"messages": [{"role": "user", "content": "q"}]}, _Spec(), _FakeTok(), 100, 0)
    assert out["task_name"] == DEFAULT_TASK_NAME


def test_processor_no_max_seq_length():
    out = online_prompt_processor(
        {"messages": [{"role": "user", "content": "q"}]}, _Spec(), _FakeTok(), max_seq_length=None, idx=0
    )
    assert out["loss_multiplier"] == 1.0  # None disables the length gate


# ---------------------------------------------------------------------------
# _resolve_enable_thinking (per-prompt policy thinking toggle)
# ---------------------------------------------------------------------------
def test_resolve_thinking_no_config_is_none():
    # No config and no per-row override -> None (omit the kwarg; prior behavior).
    assert _resolve_enable_thinking({}, idx=0, thinking_cfg=None) is None
    assert _resolve_enable_thinking({}, idx=0, thinking_cfg={"mode": "default"}) is None


def test_resolve_thinking_on_off():
    assert _resolve_enable_thinking({}, idx=0, thinking_cfg={"mode": "on"}) is True
    assert _resolve_enable_thinking({}, idx=0, thinking_cfg={"mode": "off"}) is False


def test_resolve_thinking_row_override_wins():
    # An explicit per-row flag beats the configured mode, both ways.
    assert _resolve_enable_thinking({ENABLE_THINKING_KEY: True}, 0, {"mode": "off"}) is True
    assert _resolve_enable_thinking({ENABLE_THINKING_KEY: False}, 0, {"mode": "on"}) is False
    # ...and beats "no config" too (truthy/falsy coerced to bool).
    assert _resolve_enable_thinking({ENABLE_THINKING_KEY: 1}, 0, None) is True
    assert _resolve_enable_thinking({ENABLE_THINKING_KEY: 0}, 0, None) is False


def test_resolve_thinking_random_deterministic_and_bounded():
    cfg = {"mode": "random", "probability": 0.5, "seed": 7}
    # Deterministic: same (seed, idx) -> same decision across calls.
    first = [_resolve_enable_thinking({}, i, cfg) for i in range(50)]
    second = [_resolve_enable_thinking({}, i, cfg) for i in range(50)]
    assert first == second
    assert all(isinstance(v, bool) for v in first)
    # Not all-equal for p=0.5 over 50 samples (sanity that it actually varies).
    assert any(first) and not all(first)


def test_resolve_thinking_random_probability_extremes():
    on = {"mode": "random", "probability": 1.0, "seed": 0}
    off = {"mode": "random", "probability": 0.0, "seed": 0}
    assert all(_resolve_enable_thinking({}, i, on) for i in range(20))
    assert not any(_resolve_enable_thinking({}, i, off) for i in range(20))


def test_resolve_thinking_seed_changes_draw():
    # Different seeds give (generally) different per-idx draws.
    a = [_resolve_enable_thinking({}, i, {"mode": "random", "probability": 0.5, "seed": 1}) for i in range(50)]
    b = [_resolve_enable_thinking({}, i, {"mode": "random", "probability": 0.5, "seed": 2}) for i in range(50)]
    assert a != b


def test_resolve_thinking_bad_mode_raises():
    with pytest.raises(ValueError, match="online_dpo.thinking.mode"):
        _resolve_enable_thinking({}, 0, {"mode": "sometimes"})


# ---------------------------------------------------------------------------
# online_prompt_processor thinking integration (kwarg passthrough + extra_env_info)
# ---------------------------------------------------------------------------
def test_processor_default_mode_omits_thinking_kwarg():
    out = online_prompt_processor(
        {"messages": [{"role": "user", "content": "q"}]}, _Spec(), _FakeTok(), 100, 0,
        thinking_cfg={"mode": "default"},
    )
    content = out["message_log"][0]["content"]
    assert "<think>" not in content and "<nothink>" not in content  # kwarg omitted
    assert ENABLE_THINKING_KEY not in out["extra_env_info"]  # nothing stashed


def test_processor_thinking_on_passes_kwarg_and_stashes_flag():
    out = online_prompt_processor(
        {"messages": [{"role": "user", "content": "q"}]}, _Spec(), _FakeTok(), 100, 0,
        thinking_cfg={"mode": "on"},
    )
    assert "<think>" in out["message_log"][0]["content"]
    assert out["extra_env_info"][ENABLE_THINKING_KEY] is True
    # the clean judge prompt is unaffected by the thinking toggle
    assert out["extra_env_info"]["judge_prompt"] == [{"role": "user", "content": "q"}]


def test_processor_thinking_off_passes_false():
    out = online_prompt_processor(
        {"messages": [{"role": "user", "content": "q"}]}, _Spec(), _FakeTok(), 100, 0,
        thinking_cfg={"mode": "off"},
    )
    assert "<nothink>" in out["message_log"][0]["content"]
    assert out["extra_env_info"][ENABLE_THINKING_KEY] is False


def test_processor_row_override_beats_config_mode():
    out = online_prompt_processor(
        {"messages": [{"role": "user", "content": "q"}], ENABLE_THINKING_KEY: True},
        _Spec(), _FakeTok(), 100, 0, thinking_cfg={"mode": "off"},
    )
    assert "<think>" in out["message_log"][0]["content"]
    assert out["extra_env_info"][ENABLE_THINKING_KEY] is True


# ---------------------------------------------------------------------------
# registry wiring (needs the real nemo_rl.data stack)
# ---------------------------------------------------------------------------
def test_register_into_registries():
    try:
        from nemo_rl.data.datasets.response_datasets import DATASET_REGISTRY
        from nemo_rl.data.processors import PROCESSOR_REGISTRY
    except Exception:  # noqa: BLE001 - decord/full runtime absent off-cluster
        pytest.skip("nemo_rl.data stack not importable in this environment")
    from nemo_rl_apertus.online_data import (
        ONLINE_PROMPT_PROCESSOR_NAME,
        PROMPT_ONLY_DATASET_NAME,
        register_online_dpo_data,
    )

    register_online_dpo_data()
    assert PROMPT_ONLY_DATASET_NAME in DATASET_REGISTRY
    assert PROCESSOR_REGISTRY[ONLINE_PROMPT_PROCESSOR_NAME] is online_prompt_processor
    register_online_dpo_data()  # idempotent (setdefault) — must not raise


def test_register_binds_thinking_cfg():
    try:
        from nemo_rl.data.processors import PROCESSOR_REGISTRY
    except Exception:  # noqa: BLE001 - decord/full runtime absent off-cluster
        pytest.skip("nemo_rl.data stack not importable in this environment")
    import functools

    from nemo_rl_apertus.online_data import (
        ONLINE_PROMPT_PROCESSOR_NAME,
        register_online_dpo_data,
    )

    # Control registry state (setdefault won't replace an existing entry): pop, then restore.
    saved = PROCESSOR_REGISTRY.pop(ONLINE_PROMPT_PROCESSOR_NAME, None)
    try:
        register_online_dpo_data(thinking_cfg={"mode": "on"})
        proc = PROCESSOR_REGISTRY[ONLINE_PROMPT_PROCESSOR_NAME]
        # the bound processor is a partial carrying the thinking config...
        assert isinstance(proc, functools.partial)
        assert proc.keywords["thinking_cfg"] == {"mode": "on"}
        # ...and invoking it with the 5-positional registry signature applies thinking
        out = proc({"messages": [{"role": "user", "content": "q"}]}, _Spec(), _FakeTok(), 100, 0)
        assert "<think>" in out["message_log"][0]["content"]
    finally:
        if saved is None:
            PROCESSOR_REGISTRY.pop(ONLINE_PROMPT_PROCESSOR_NAME, None)
        else:
            PROCESSOR_REGISTRY[ONLINE_PROMPT_PROCESSOR_NAME] = saved
