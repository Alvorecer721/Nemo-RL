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

"""Subprocess tests for the stdlib-only overlay fingerprint gate."""

import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "tools/restamp_overlay_fingerprint.py"
BASE = {
    "pyproject.toml": "old-project",
    "uv.lock": "old-lock",
    "submodules/3rdparty/Bridge": "abc123",
    "submodules/3rdparty/Gym": "def456",
    "submodules/3rdparty/kernels": "789abc",
}
LOCK = """
[[package]]
name = "project"
source = { editable = "." }

[[package]]
name = "core"
source = { editable = "3rdparty/Bridge/3rdparty/LM" }

[[package]]
name = "gym"
source = { editable = "3rdparty/Gym" }

[[package]]
name = "wheel"
source = { registry = "https://pypi.org/simple" }
"""


class OverlayFingerprintTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.container = Path(temporary.name) / "container-fingerprint"
        self.source = Path(temporary.name) / "source-fingerprint"
        self.lock = Path(temporary.name) / "uv.lock"
        self.lock.write_text(LOCK)
        self.original = json.dumps(BASE).encode()
        self.container.write_bytes(self.original)
        self.source.write_bytes(self.original)

    def _run(self, requested, command=None):
        payload = json.dumps(requested).encode()
        command = command or ["restamp", str(self.container), str(self.source)]
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *command, "--lock", str(self.lock)],
            env={
                **os.environ,
                "NEMO_RL_BUILD_FINGERPRINT_B64": base64.b64encode(payload).decode(),
            },
            capture_output=True,
            text=True,
        )
        return result, payload

    def test_matching_submodules_allow_changed_dependency_hashes(self):
        requested = {**BASE, "pyproject.toml": "new-project", "uv.lock": "new-lock"}
        result, payload = self._run(requested)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.container.read_bytes(), payload)
        self.assertEqual(self.source.read_bytes(), payload)

    def test_changed_or_missing_requested_pins_preserve_both_fingerprints(self):
        cases = [
            {**BASE, "submodules/3rdparty/kernels": "different"},
            {key: value for key, value in BASE.items() if not key.endswith("kernels")},
            {**BASE, "submodules/3rdparty/New": "new-pin"},
            {"pyproject.toml": "new-project", "uv.lock": "new-lock"},
            {**BASE, "submodules/3rdparty/Bridge": None},
        ]
        for requested in cases:
            with self.subTest(requested=requested):
                result, _ = self._run(requested)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("submodule", result.stderr.lower())
                self.assertEqual(self.container.read_bytes(), self.original)
                self.assertEqual(self.source.read_bytes(), self.original)

    def test_shipped_editable_submodules_may_move(self):
        requested = {
            **BASE,
            "submodules/3rdparty/Bridge": "moved-with-nested-core",
            "submodules/3rdparty/Gym": "moved",
        }
        listed, _ = self._run(requested, ["shipped-submodules", str(self.container)])
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(listed.stdout.split(), ["3rdparty/Bridge", "3rdparty/Gym"])

        restamp = ["restamp", str(self.container), str(self.source), "--shipped"]
        refused, _ = self._run(requested, [*restamp, "3rdparty/Gym"])
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("3rdparty/Bridge", refused.stderr)
        self.assertEqual(self.container.read_bytes(), self.original)

        result, payload = self._run(requested, [*restamp, *listed.stdout.split()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.container.read_bytes(), payload)

    def test_shipping_a_baked_submodule_does_not_excuse_its_pin(self):
        requested = {**BASE, "submodules/3rdparty/kernels": "different"}
        result, _ = self._run(
            requested,
            [
                "restamp",
                str(self.container),
                str(self.source),
                "--shipped",
                "3rdparty/kernels",
            ],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.container.read_bytes(), self.original)

    def test_missing_inherited_pins_preserve_both_fingerprints(self):
        for inherited in ({"uv.lock": "old-lock"}, {}, []):
            with self.subTest(inherited=inherited):
                before = json.dumps(inherited).encode()
                self.container.write_bytes(before)
                result, _ = self._run(BASE)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("submodule", result.stderr.lower())
                self.assertEqual(self.container.read_bytes(), before)
                self.assertEqual(self.source.read_bytes(), self.original)

    def test_missing_inherited_fingerprint_is_not_created(self):
        self.container.unlink()
        result, _ = self._run(BASE)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.container.exists())
        self.assertEqual(self.source.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
