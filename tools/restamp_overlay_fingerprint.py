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

"""Restamp an overlay only when the submodules it inherits match the source.

Uses only stdlib so it can run before dependency synchronization or nemo_rl imports.
The requested fingerprint comes from NEMO_RL_BUILD_FINGERPRINT_B64, as in the
release build. Dependency hashes may change. A submodule the lock installs as
editable source may move too when the overlay ships its tree; any other
submodule is baked into the base image and must not.
"""

import argparse
import base64
import json
import os
import tomllib
from pathlib import Path


def submodule_pins(fingerprint: object, *, label: str) -> dict[str, str]:
    """Read a nonempty submodule map without silently dropping invalid pins."""
    if not isinstance(fingerprint, dict):
        raise ValueError(f"{label} fingerprint has no submodule map")
    pins = {
        key: value
        for key, value in fingerprint.items()
        if key.startswith("submodules/")
    }
    if not pins or any(
        key == "submodules/" or not isinstance(value, str) or not value.strip()
        for key, value in pins.items()
    ):
        raise ValueError(f"{label} fingerprint has missing or invalid submodule pins")
    return pins


def editable_submodules(pins: dict[str, str], lock: Path) -> set[str]:
    """Pins of submodules that hold an editable source of the lock."""
    packages = tomllib.loads(lock.read_text())["package"]
    sources = [package.get("source", {}).get("editable") for package in packages]
    return {
        key
        for key in pins
        for source in sources
        if source and f"{source}/".startswith(f"{key.removeprefix('submodules/')}/")
    }


def moved_pins(inherited: dict[str, str], requested: dict[str, str]) -> set[str]:
    return {
        key
        for key in inherited.keys() | requested.keys()
        if inherited.get(key) != requested.get(key)
    }


def requested_fingerprint() -> bytes:
    return base64.b64decode(os.environ["NEMO_RL_BUILD_FINGERPRINT_B64"], validate=True)


def restamp(args: argparse.Namespace) -> None:
    """Validate the inherited pins before overwriting either fingerprint."""
    try:
        payload = requested_fingerprint()
        requested = submodule_pins(json.loads(payload), label="Requested")
        inherited = submodule_pins(
            json.loads(args.container_fingerprint.read_bytes()), label="Inherited"
        )
        shipped = {f"submodules/{path}" for path in args.shipped}
        refused = moved_pins(inherited, requested) - (
            shipped & editable_submodules(requested, args.lock)
        )
        if refused:
            raise ValueError(
                "Inherited submodule pins differ from the requested source and "
                "the overlay does not ship them as editable source; rebuild the "
                "base image:\n"
                + "\n".join(
                    f"  {key}: inherited={inherited.get(key, 'missing')} "
                    f"requested={requested.get(key, 'missing')}"
                    for key in sorted(refused)
                )
            )
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Overlay fingerprint validation failed: {error}") from error

    args.container_fingerprint.write_bytes(payload)
    args.source_fingerprint.write_bytes(payload)


def shipped_submodules(args: argparse.Namespace) -> None:
    """Print the editable submodules whose pin moved since the base release."""
    requested = submodule_pins(json.loads(requested_fingerprint()), label="Requested")
    inherited = submodule_pins(
        json.loads(args.inherited_fingerprint.read_bytes()), label="Inherited"
    )
    moved = moved_pins(inherited, requested) & editable_submodules(requested, args.lock)
    for key in sorted(moved):
        print(key.removeprefix("submodules/"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stamp = commands.add_parser("restamp")
    stamp.add_argument("container_fingerprint", type=Path)
    stamp.add_argument("source_fingerprint", type=Path)
    stamp.add_argument("--shipped", nargs="*", default=[])
    stamp.set_defaults(run=restamp)
    shipped = commands.add_parser("shipped-submodules")
    shipped.add_argument("inherited_fingerprint", type=Path)
    shipped.set_defaults(run=shipped_submodules)
    for command in (stamp, shipped):
        command.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
