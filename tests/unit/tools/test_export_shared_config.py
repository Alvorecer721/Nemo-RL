# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "export_shared_config", REPO_ROOT / "tools" / "export_shared_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_shared_config"] = module
    spec.loader.exec_module(module)
    return module


def test_scrub_replaces_site_paths_and_keeps_basename(tool):
    cfg = {
        "policy": {
            "model_name": "/capstor/store/models/ap1p5-70b",
            "megatron_cfg": {"env_vars": {"HF_HOME": "/iopsstor/scratch/u/.cache"}},
            "train_micro_batch_size": 1,
        },
        "logger": {"wandb": {"entity": "swissai", "project": "nemo-rl-apertus"}},
        "note": "/opt/nemo-rl stays",
    }
    out = tool.scrub(cfg, ("/capstor", "/iopsstor"))
    assert out["policy"]["model_name"] == "<site>/ap1p5-70b"
    assert out["policy"]["megatron_cfg"]["env_vars"]["HF_HOME"] == "<site>/.cache"
    assert out["policy"]["train_micro_batch_size"] == 1
    assert out["logger"]["wandb"] == {
        "entity": "<wandb-entity>",
        "project": "nemo-rl-apertus",
    }
    assert out["note"] == "/opt/nemo-rl stays"
    assert cfg["policy"]["model_name"].startswith("/capstor"), (
        "input must not be mutated"
    )


def test_fork_only_paths_uses_identifier_set(tool):
    cfg = {
        "grpo": {"num_prompts_per_step": 48, "cot_think_token_ids": [32, 33]},
        "async_rl": {"sampler": {"name": "windowed", "max_staleness_versions": 2}},
    }
    known = {"grpo", "num_prompts_per_step", "async_rl", "sampler", "name"}
    assert tool.fork_only_paths(cfg, known) == [
        "async_rl.sampler.max_staleness_versions",
        "grpo.cot_think_token_ids",
    ]


def test_render_header_lists_provenance_and_fork_only_keys(tool):
    text = tool.render(
        {"b": 1, "a": {"x": "<site>/m"}},
        {
            "Reference run": "70B GSM8K",
            "Source commit": "7197ac71505b",
            "Slurm job": "3352055",
        },
        ["a.x"],
    )
    head, body = text.split("\n\n", 1)
    assert head.splitlines()[0] == "# Reference run: 70B GSM8K"
    assert "# Source commit: 7197ac71505b" in head
    assert "# Fork-only keys (absent from the upstream ref):" in head
    assert "#   - a.x" in head
    assert yaml.safe_load(body) == {"a": {"x": "<site>/m"}, "b": 1}
    assert body.index("a:") < body.index("b:")
