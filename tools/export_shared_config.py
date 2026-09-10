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

import copy
import re
from pathlib import PurePosixPath

import yaml

DEFAULT_SCRUB_PREFIXES = ("/capstor", "/iopsstor", "/users")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


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
