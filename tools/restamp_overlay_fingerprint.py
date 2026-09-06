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

"""Restamp an overlay only when its inherited submodule pins match the source.

Uses only stdlib so it can run before dependency synchronization or nemo_rl imports.
The requested fingerprint comes from NEMO_RL_BUILD_FINGERPRINT_B64, as in the
release build. Dependency hashes may change; inherited submodules may not.
"""

import argparse
import base64
import json
import os
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


def main() -> None:
    """Validate the inherited pins before overwriting either fingerprint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container_fingerprint", type=Path)
    parser.add_argument("source_fingerprint", type=Path)
    args = parser.parse_args()
    try:
        payload = base64.b64decode(
            os.environ["NEMO_RL_BUILD_FINGERPRINT_B64"], validate=True
        )
        requested = submodule_pins(json.loads(payload), label="Requested")
        inherited = submodule_pins(
            json.loads(args.container_fingerprint.read_bytes()), label="Inherited"
        )
        differences = [
            f"  {key}: inherited={inherited.get(key, 'missing')} "
            f"requested={requested.get(key, 'missing')}"
            for key in sorted(inherited.keys() | requested.keys())
            if inherited.get(key) != requested.get(key)
        ]
        if differences:
            raise ValueError(
                "Inherited submodule pins differ from the requested source; "
                "rebuild the base image:\n" + "\n".join(differences)
            )
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Overlay fingerprint validation failed: {error}") from error

    args.container_fingerprint.write_bytes(payload)
    args.source_fingerprint.write_bytes(payload)


if __name__ == "__main__":
    main()
