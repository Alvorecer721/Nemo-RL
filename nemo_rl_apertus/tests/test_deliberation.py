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
"""Per-pair deliberation rule: ``enable_thinking`` keyed on CHOSEN, both sides.

CONTRACT under test: the deliberation decision is PER-PAIR, keyed on the
CHOSEN response only — both sides of a pair render under the same mode, so
prompt conditioning is identical across the pair and the DPO logprob
difference is attributable to the responses alone. Rejected-with-think under
``Deliberation: disabled`` is intentional negative signal, not an error.

Pending-CI note (2026-06-13): this module needs the uv-locked runtime — under
bare ~/miniconda3 the nemo_rl import chain fails at ``import decord``
(nemo_rl/data/multimodal_utils.py). Compile-checked locally; the template
mechanics (partial-bound ``enable_thinking`` flips ``Deliberation:``, explicit
False == undefined, outer partial overrides inner) were verified standalone
against the real tokenizer snapshot.
"""

import pytest
import torch

from nemo_rl_apertus.omni_preference import (
    OmniPreferenceDataset,
    has_think_markers,
    omni_preference_preprocessor,
)
from nemo_rl_apertus.tests.toy_store import build_toy_dataset


# ---------------------------------------------------------------------------
# pure marker check (no tokenizer)
# ---------------------------------------------------------------------------


def test_has_think_markers_mirrors_sft_marker_set():
    # Marker set mirrored from vision_tokenization/discrete/sft_segments.py
    # (_THINKING_OPEN_MARKERS) — openers only, unclosed traces still count.
    assert has_think_markers("<think>chain</think> answer")
    assert has_think_markers("x <|channel>thought\nchain")
    assert has_think_markers("<thought>chain</thought>")
    assert has_think_markers("partial <think> trace, never closed")
    assert not has_think_markers("no reasoning markers here")
    assert not has_think_markers("an orphan </think> close does not count")


# ---------------------------------------------------------------------------
# per-pair rendering (real tokenizer; toy store only for the media reader)
# ---------------------------------------------------------------------------


@pytest.fixture()
def ds(tmp_path):
    build_toy_dataset(tmp_path / "ds")
    return OmniPreferenceDataset(tmp_path / "ds", split="train")


def _pair(chosen: str, rejected: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": "Question?"}],
        "chosen": chosen,
        "rejected": rejected,
        "prompt_media_refs": [],
        "chosen_media_refs": [],
        "rejected_media_refs": [],
    }


def _process(ds, tokenizer, datum_dict):
    return omni_preference_preprocessor(
        datum_dict, ds.task_spec, tokenizer, 8192, 0,
        media=ds.media, image_marker_id=ds.image_marker_id,
    )


# The three-row table: the decision is keyed on CHOSEN only.
THREE_ROWS = [
    ("<think>because</think> Yes.", "Plain refusal.", "enabled"),
    ("Plain answer.", "<think>sneaky</think> Nope.", "disabled"),
    ("Plain answer.", "Another plain answer.", "disabled"),
]


@pytest.mark.parametrize("chosen,rejected,mode", THREE_ROWS)
def test_per_pair_deliberation_table(tokenizer, ds, chosen, rejected, mode):
    datum = _process(ds, tokenizer, _pair(chosen, rejected))
    other = "disabled" if mode == "enabled" else "enabled"
    for side in ("chosen", "rejected"):
        text = "".join(m["content"] for m in datum[f"message_log_{side}"])
        assert f"Deliberation: {mode}" in text, (side, mode)
        assert f"Deliberation: {other}" not in text, (side, mode)


@pytest.mark.parametrize("chosen,rejected,mode", THREE_ROWS)
def test_same_conditioning_across_sides(tokenizer, ds, chosen, rejected, mode):
    """Prompt token ids are identical across the two sides' renders.

    The interesting row is rejected-only-think: even there, the rejected side
    renders under the chosen-keyed mode, so the shared prompt is bit-identical.
    """
    datum = _process(ds, tokenizer, _pair(chosen, rejected))
    prompt_chosen = torch.cat(
        [m["token_ids"] for m in datum["message_log_chosen"][:-1]]
    )
    prompt_rejected = torch.cat(
        [m["token_ids"] for m in datum["message_log_rejected"][:-1]]
    )
    assert torch.equal(prompt_chosen, prompt_rejected)


def test_parity_holds_under_enabled_mode(tokenizer, ds):
    """Gate-1 check 2, extended to enabled mode.

    The concatenated per-message render must equal the one-shot template
    render with the same ``enable_thinking`` (modulo the BOS/EOS that
    ``get_formatted_message_log`` documents adding). Gate 1 itself covers the
    unconditioned render, which this template defines as disabled.
    """
    chosen = "<think>because</think> Yes."
    datum = _process(ds, tokenizer, _pair(chosen, "No."))
    got = "".join(m["content"] for m in datum["message_log_chosen"])

    messages = [
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": chosen},
    ]
    expected = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=True
    )
    if not expected.startswith(tokenizer.bos_token):
        expected = tokenizer.bos_token + expected
    if not expected.rstrip("\n").endswith(tokenizer.eos_token):
        expected = expected + tokenizer.eos_token
    assert got == expected


def test_binding_is_restored_after_processing(tokenizer, ds):
    """The per-pair partial must not leak past the pair (instance dict clean)."""
    before = tokenizer.__dict__.get("apply_chat_template")
    _process(ds, tokenizer, _pair("<think>t</think> a", "b"))
    assert tokenizer.__dict__.get("apply_chat_template") is before
