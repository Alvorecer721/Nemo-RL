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
"""Shared fixtures for the Apertus extension tests.

Pytest imports ``nemo_rl`` from the checkout under test. The real tokenizer
(~200 s load) is session-scoped and loaded only on request.
"""

import os
import sys

import pytest

TOKENIZER_PATH = (
    "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Apertus-v1.5-8B-official"
)


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


_exit_status = 0


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    global _exit_status
    _exit_status = int(exitstatus)


def pytest_unconfigure(config):
    """Bypass CPython finalization, which segfaults after green runs here.

    With the ~136k-added-token tokenizer plus the torch/decord native stack
    loaded, interpreter teardown crashes (exit 139) AFTER the full summary has
    been printed, turning green suites red. All reporting is done by the time
    unconfigure runs; exit with pytest's real status instead of risking the
    native teardown.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status)
