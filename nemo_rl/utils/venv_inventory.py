# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record and verify the result of an image worker's successful frozen sync.

Run directly with the worker interpreter, without importing NeMo-RL. This checks
installation metadata against the producer's receipt, not UV's reinstall plan:
upstream wheel tags, dynamic versions and cache freshness can trigger a reinstall
even immediately after a successful frozen sync. Source fingerprints and native
imports are checked separately by image qualification.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from importlib.metadata import distributions
from pathlib import Path

INVENTORY_FILENAME = "NEMO_RL_VENV_PACKAGES.json"


def snapshot(extras: list[str]) -> dict:
    packages = {}
    for dist in distributions():
        raw_name = dist.metadata.get("Name")
        if not raw_name:
            raise ValueError("Installed distribution is missing its package name")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        if name == "nemo-rl":
            continue  # Source-only overlays may replace the project itself.
        if name in packages:
            raise ValueError(f"Duplicate installed distribution: {name}")
        digest = hashlib.sha256()
        for filename in ("METADATA", "WHEEL", "RECORD", "direct_url.json"):
            content = dist.read_text(filename)
            if not content and filename != "direct_url.json":
                raise ValueError(f"{name}: missing or empty {filename}")
            digest.update(filename.encode() + b"\0")
            digest.update((content or "").encode() + b"\0")
        packages[name] = digest.hexdigest()
    if not packages:
        raise ValueError("Worker contains no installed dependencies")
    return {
        "schema": 1,
        "python": sys.version,
        "prefix": str(Path(sys.prefix).resolve()),
        "extras": sorted(set(extras)),
        "packages": packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "verify"))
    parser.add_argument("--extra", action="append", default=[])
    args = parser.parse_args()
    current = snapshot(args.extra)
    path = Path(sys.prefix) / INVENTORY_FILENAME
    if args.action == "record":
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(current, stream, sort_keys=True)
                stream.write("\n")
            temporary.chmod(0o644)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    else:
        expected = json.loads(path.read_text())
        if expected != current:
            raise RuntimeError(
                f"Worker installation differs from its frozen build inventory: {path}"
            )
    print(f"Worker inventory {args.action}: {len(current['packages'])} packages")


if __name__ == "__main__":
    main()
