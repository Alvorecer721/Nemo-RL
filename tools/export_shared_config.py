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

"""Export a run's resolved NeMo-RL config as a scrubbed, annotated hand-off YAML."""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import yaml

DEFAULT_SCRUB_PREFIXES = ("/capstor", "/iopsstor", "/users")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def scrub(config: dict, prefixes: tuple[str, ...]) -> dict:
    def visit(value):
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        if isinstance(value, str) and value.startswith(prefixes):
            return "<site>/" + PurePosixPath(value).name
        return value

    out = visit(copy.deepcopy(config))
    wandb = (
        out.get("logger", {}).get("wandb")
        if isinstance(out.get("logger"), dict)
        else None
    )
    if isinstance(wandb, dict) and "entity" in wandb:
        wandb["entity"] = "<wandb-entity>"
    return out


def fork_only_paths(config: dict, upstream_identifiers: set[str]) -> list[str]:
    found: list[str] = []

    def visit(value, path):
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            dotted = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key not in upstream_identifiers:
                found.append(dotted)
            visit(child, dotted)

    visit(config, "")
    return sorted(found)


def render(config: dict, provenance: dict[str, str], fork_only: list[str]) -> str:
    lines = [f"# {k}: {v}" for k, v in provenance.items()]
    lines.append("# Fork-only keys (absent from the upstream ref):")
    lines.extend(f"#   - {path}" for path in fork_only)
    if not fork_only:
        lines.append("#   (none)")
    body = yaml.safe_dump(config, sort_keys=True, default_flow_style=False, width=100)
    return "\n".join(lines) + "\n\n" + body


def _load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolved_container(path: Path) -> dict:
    from omegaconf import OmegaConf

    tools_dir = Path(__file__).resolve().parent
    config_cli = _load_module("config_cli", tools_dir / "config_cli.py")
    nemo_rl_config = _load_module(
        "nemo_rl_utils_config", tools_dir.parent / "nemo_rl" / "utils" / "config.py"
    )
    nemo_rl_config.register_omegaconf_resolvers()
    return OmegaConf.to_container(config_cli.load_config(str(path)), resolve=True)


def load_resolved(config: Path | None, recipe: Path | None) -> dict:
    if (config is None) == (recipe is None):
        raise ValueError("pass exactly one of --config or --recipe")
    if config is not None:
        return yaml.safe_load(config.read_text())
    return _resolved_container(recipe)


def upstream_identifiers_from_git(repo: Path, ref: str) -> set[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "grep",
            "-h",
            "-o",
            "-w",
            "-E",
            _IDENTIFIER.pattern,
            ref,
            "--",
            "nemo_rl/*.py",
            "examples/configs/*.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git grep failed for {ref} in {repo}: {result.stderr.strip()}"
        )
    identifiers = set(result.stdout.split())
    if not identifiers:
        raise RuntimeError(
            f"no identifiers found at {ref} under nemo_rl/ or examples/configs/ in {repo}"
        )
    return identifiers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config", type=Path, help="resolved config.yaml saved with a checkpoint"
    )
    source.add_argument(
        "--recipe",
        type=Path,
        help="recipe YAML; its defaults chain and env interpolations are resolved",
    )
    idents = parser.add_mutually_exclusive_group(required=True)
    idents.add_argument(
        "--upstream-ref", help="git ref of the upstream snapshot to annotate against"
    )
    idents.add_argument(
        "--upstream-identifiers",
        type=Path,
        help="newline-separated identifier list (tests)",
    )
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--resolved-from", required=True)
    parser.add_argument("--scrub-prefix", action="append", dest="scrub_prefixes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    resolved = load_resolved(args.config, args.recipe)
    if args.upstream_identifiers is not None:
        identifiers = set(args.upstream_identifiers.read_text().split())
        upstream_label = str(args.upstream_identifiers)
    else:
        identifiers = upstream_identifiers_from_git(args.repo, args.upstream_ref)
        upstream_label = args.upstream_ref
    prefixes = tuple(args.scrub_prefixes or DEFAULT_SCRUB_PREFIXES)
    scrubbed = scrub(resolved, prefixes)
    provenance = {
        "Reference run": args.reference_run,
        "Source commit": args.source_commit,
        "Slurm job": args.job_id,
        "Resolved from": args.resolved_from,
        "Upstream ref for annotation": upstream_label,
        "Scrubbed prefixes": ", ".join(prefixes),
        "Note": "reference of what ran on the fork; not loadable by upstream as-is",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(scrubbed, provenance, fork_only_paths(scrubbed, identifiers))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
