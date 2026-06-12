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
"""Gate 1: bit-exact splice parity against an independent reference assembly.

Oracle: render the same messages through the stock ``get_formatted_message_log``
and replace marker ids with blocks BY HAND, then compare bit-exact with the
adapter's output. Two oracle-independent assertions kill the common-mode
failure where the oracle and the SUT share a broken assumption:

  1. marker atomicity is re-verified against the *current* tokenizer snapshot
     by directly encoding the one-shot template render — bypassing
     ``get_formatted_message_log`` entirely;
  2. the concatenated per-message render must equal the one-shot
     ``apply_chat_template(tokenize=False)`` string (catches drift in the
     incremental per-turn rendering both oracle and SUT rely on).
"""

import torch

from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl_apertus.omni_preference import (
    OmniPreferenceDataset,
    omni_preference_preprocessor,
)
from nemo_rl_apertus.tests.toy_store import build_toy_dataset


def test_gate1_splice_parity(tokenizer, tmp_path):
    meta = build_toy_dataset(tmp_path / "ds")
    ds = OmniPreferenceDataset(tmp_path / "ds", split="train")
    datum = omni_preference_preprocessor(
        ds[1], ds.task_spec, tokenizer, 8192, 1,
        media=ds.media, image_marker_id=ds.image_marker_id
    )
    got = torch.cat([m["token_ids"] for m in datum["message_log_chosen"]])

    row = ds[1]
    messages = list(row["prompt"]) + [{"role": "assistant", "content": row["chosen"]}]

    # Check 1: the prompt's two "<|image|>" encode to exactly two ids 131079.
    # Plain encode — independent of the SUT's render path and template API.
    one_shot_ids = tokenizer(
        tokenizer.apply_chat_template(messages, tokenize=False),
        add_special_tokens=False,
    ).input_ids
    assert list(one_shot_ids).count(ds.image_marker_id) == 2

    # Reference render (same stock path the adapter uses).
    ref_log = get_formatted_message_log(
        messages, tokenizer, ds.task_spec, add_bos_token=True, add_eos_token=True
    )

    # Check 2: concatenated per-message render == one-shot template render,
    # modulo the BOS/EOS that get_formatted_message_log documents adding.
    expected_str = tokenizer.apply_chat_template(messages, tokenize=False)
    if not expected_str.startswith(tokenizer.bos_token):
        expected_str = tokenizer.bos_token + expected_str
    if not expected_str.rstrip("\n").endswith(tokenizer.eos_token):
        expected_str = expected_str + tokenizer.eos_token
    assert "".join(m["content"] for m in ref_log) == expected_str

    # ORACLE: independent splice by hand, then bit-exact comparison.
    ref_flat = torch.cat([m["token_ids"] for m in ref_log]).tolist()
    blocks = iter([meta["blocks"]["img1"], meta["blocks"]["img2"]])
    expected = []
    for token in ref_flat:
        if token == ds.image_marker_id:
            expected.extend(next(blocks))
        else:
            expected.append(token)
    assert got.tolist() == expected, "bit-exact splice parity failed"
