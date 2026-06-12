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
"""Shared fixtures and the nemo_rl runtime pin.

Runtime pin: the adapter targets the LOCKED runtime at /opt/nemo-rl (v0.6.0 —
every upstream signature was verified there). This repository also contains a
*different* (newer) ``nemo_rl`` source tree, and pytest prepends the repo root
to ``sys.path`` to import ``nemo_rl_apertus`` — accidentally shadowing the
locked ``nemo_rl`` with the unverified one (which additionally needs deps the
locked venv does not ship, e.g. soundfile). This conftest restores the
contract by importing ``nemo_rl`` while /opt/nemo-rl is first on ``sys.path``:
once the top package is cached in ``sys.modules``, its ``__path__`` pins every
submodule to the locked tree, immune to pytest's later sys.path prepends.
``nemo_rl_apertus`` still resolves to this repo (absent under /opt).

The real tokenizer takes ~200 s to LOAD — it is loaded at most once per pytest
session (session scope) and only when a test requests it, so format/manifest
tests stay fast.
"""

import sys
from pathlib import Path

import pytest

LOCKED_NEMO_RL_REPO = "/opt/nemo-rl"
if Path(LOCKED_NEMO_RL_REPO, "nemo_rl").is_dir():
    sys.path.insert(0, LOCKED_NEMO_RL_REPO)
    import nemo_rl

    assert nemo_rl.__file__ is not None and nemo_rl.__file__.startswith(
        LOCKED_NEMO_RL_REPO
    ), f"nemo_rl resolved outside the locked runtime: {nemo_rl.__file__}"

TOKENIZER_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed.snapshot-20260611"


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)
