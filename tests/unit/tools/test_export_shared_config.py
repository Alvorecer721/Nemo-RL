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
import subprocess
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


def _init_repo(path, files):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    for name, text in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )


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


def test_scrub_requires_prefix_boundary(tool):
    cfg = {"a": "/usersfoo/x", "b": "/users/x/y"}
    out = tool.scrub(cfg, ("/users",))
    assert out["a"] == "/usersfoo/x"
    assert out["b"] == "<site>/y"


def test_scrub_walks_lists(tool):
    cfg = {"paths": ["/capstor/a", "/opt/b"]}
    out = tool.scrub(cfg, ("/capstor", "/iopsstor"))
    assert out["paths"] == ["<site>/a", "/opt/b"]


def test_fork_only_paths_uses_identifier_set(tool):
    cfg = {
        "grpo": {"num_prompts_per_step": 48, "cot_think_token_ids": [32, 33]},
        "async_rl": {"sampler": {"name": "windowed"}, "weight_sync_period": 2},
    }
    known = {"grpo", "num_prompts_per_step", "async_rl", "sampler", "name"}
    assert tool.fork_only_paths(cfg, known) == [
        "async_rl.weight_sync_period",
        "grpo.cot_think_token_ids",
    ]


def test_fork_only_paths_walks_lists(tool):
    cfg = {"data": {"train": [{"name": "x", "shard_hint": 1}]}}
    known = {"data", "train", "name"}
    assert tool.fork_only_paths(cfg, known) == ["data.train[0].shard_hint"]


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


def test_load_resolved_recipe_follows_defaults_and_env(tool, tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text("policy:\n  model_name: base\n  train_micro_batch_size: 2\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("defaults: ./base.yaml\npolicy:\n  model_name: ${oc.env:AP_CKPT}\n")
    monkeypatch.setenv("AP_CKPT", "/capstor/models/ap")
    cfg = tool.load_resolved(config=None, recipe=leaf)
    assert cfg["policy"] == {
        "model_name": "/capstor/models/ap",
        "train_micro_batch_size": 2,
    }


def test_load_resolved_requires_exactly_one_input(tool, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        tool.load_resolved(config=None, recipe=None)


def test_load_resolved_rejects_both_inputs(tool, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("a: 1\n")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("a: 1\n")
    with pytest.raises(ValueError, match="exactly one"):
        tool.load_resolved(config=config, recipe=recipe)


def test_load_resolved_rejects_non_mapping_config(tool, tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        tool.load_resolved(config=empty, recipe=None)


def test_cli_writes_scrubbed_annotated_yaml(tool, tmp_path):
    resolved = tmp_path / "config.yaml"
    resolved.write_text(
        "grpo:\n  num_prompts_per_step: 48\n  cot_think_token_ids: [32, 33]\n"
        "policy:\n  model_name: /capstor/store/models/ap1p5-70b\n"
    )
    idents = tmp_path / "idents.txt"
    idents.write_text("grpo\nnum_prompts_per_step\npolicy\nmodel_name\n")
    out = tmp_path / "out.yaml"
    rc = tool.main(
        [
            "--config",
            str(resolved),
            "--upstream-identifiers",
            str(idents),
            "--source-commit",
            "7197ac71505b",
            "--job-id",
            "3352055",
            "--reference-run",
            "70B GSM8K",
            "--resolved-from",
            "checkpoint step_2/config.yaml",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text()
    head, body = text.split("\n\n", 1)
    assert "# Slurm job: 3352055" in head
    assert "#   - grpo.cot_think_token_ids" in head
    assert "# Scrubbed prefixes: /capstor, /iopsstor, /users" in head
    assert "<site>/ap1p5-70b" in body and "/capstor" not in body


def test_main_raises_on_leftover_site_prefix_in_body(tool, tmp_path):
    resolved = tmp_path / "config.yaml"
    resolved.write_text(
        'policy:\n  megatron_cfg:\n    env_vars:\n      PYTHONPATH: "/opt/x:/capstor/y"\n'
    )
    idents = tmp_path / "idents.txt"
    idents.write_text("policy\nmegatron_cfg\nenv_vars\nPYTHONPATH\n")
    out = tmp_path / "out.yaml"
    with pytest.raises(RuntimeError, match="unscrubbed site prefix"):
        tool.main(
            [
                "--config",
                str(resolved),
                "--upstream-identifiers",
                str(idents),
                "--source-commit",
                "7197ac71505b",
                "--job-id",
                "3352055",
                "--reference-run",
                "70B GSM8K",
                "--resolved-from",
                "checkpoint step_2/config.yaml",
                "--output",
                str(out),
            ]
        )


def test_main_raises_on_leftover_site_prefix_in_resolved_from(tool, tmp_path):
    resolved = tmp_path / "config.yaml"
    resolved.write_text("policy:\n  model_name: base\n")
    idents = tmp_path / "idents.txt"
    idents.write_text("policy\nmodel_name\n")
    out = tmp_path / "out.yaml"
    with pytest.raises(RuntimeError, match="unscrubbed site prefix"):
        tool.main(
            [
                "--config",
                str(resolved),
                "--upstream-identifiers",
                str(idents),
                "--source-commit",
                "7197ac71505b",
                "--job-id",
                "3352055",
                "--reference-run",
                "70B GSM8K",
                "--resolved-from",
                "checkpoint at /iopsstor/foo",
                "--output",
                str(out),
            ]
        )


def test_upstream_identifiers_from_git_reads_the_ref(tool, tmp_path):
    _init_repo(tmp_path, {"nemo_rl/a.py": "num_prompts_per_step = 1\n"})
    idents = tool.upstream_identifiers_from_git(tmp_path, "HEAD")
    assert "num_prompts_per_step" in idents and "cot_think_token_ids" not in idents


def test_upstream_identifiers_from_git_rejects_unknown_ref(tool, tmp_path):
    _init_repo(tmp_path, {"nemo_rl/a.py": "num_prompts_per_step = 1\n"})
    with pytest.raises(RuntimeError, match="git grep failed"):
        tool.upstream_identifiers_from_git(tmp_path, "no-such-ref")


def test_load_resolved_recipe_registers_nemo_rl_resolvers(tool, tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("policy:\n  train_mb_tokens: ${mul:2,3}\n")
    cfg = tool.load_resolved(config=None, recipe=recipe)
    assert cfg["policy"]["train_mb_tokens"] == 6


def test_upstream_identifiers_from_git_keeps_short_names(tool, tmp_path):
    _init_repo(tmp_path, {"nemo_rl/a.py": "lr = 1\n"})
    assert "lr" in tool.upstream_identifiers_from_git(tmp_path, "HEAD")


SNAPSHOT_DIR = REPO_ROOT / "docs" / "reference-configs"


@pytest.mark.parametrize("snapshot", sorted(SNAPSHOT_DIR.glob("*.yaml")) or [None])
def test_reference_snapshots_have_provenance_headers(snapshot):
    assert snapshot is not None, "no snapshots committed under docs/reference-configs"
    text = snapshot.read_text()
    head, body = text.split("\n\n", 1)
    assert head.startswith("# Reference run: ")
    assert "# Source commit: " in head and "# Slurm job: " in head
    assert "/capstor" not in body and "/iopsstor" not in body and "/users/" not in body
    assert isinstance(yaml.safe_load(body), dict)
