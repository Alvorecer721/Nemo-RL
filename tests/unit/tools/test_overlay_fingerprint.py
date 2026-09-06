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
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[3] / "tools/restamp_overlay_fingerprint.py"
BASE = {
    "pyproject.toml": "old-project",
    "uv.lock": "old-lock",
    "submodules/3rdparty/Bridge": "abc123",
    "submodules/3rdparty/Gym": "def456",
}


class OverlayFingerprintTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.container = Path(temporary.name) / "container-fingerprint"
        self.source = Path(temporary.name) / "source-fingerprint"
        self.original = json.dumps(BASE).encode()
        self.container.write_bytes(self.original)
        self.source.write_bytes(self.original)

    def _run(self, requested):
        payload = json.dumps(requested).encode()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.container), str(self.source)],
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
            {**BASE, "submodules/3rdparty/Bridge": "different"},
            {key: value for key, value in BASE.items() if not key.endswith("Bridge")},
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
