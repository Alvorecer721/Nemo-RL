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

"""Exercise generated Ray sidecar behavior without starting Ray or Slurm."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SharedShutdownTest(unittest.TestCase):
    def test_ended_uses_graceful_shared_shutdown(self):
        source = (Path(__file__).resolve().parents[3] / "ray.sub").read_text()
        blocks = source.split("monitor-sidecar() {")[1:]
        self.assertEqual(len(blocks), 2)
        for role, block in zip(("head", "worker"), blocks):
            fragment = (
                "monitor-sidecar() {"
                + block.split("# Background process to sync ray logs", 1)[0]
            )
            for shared in ("0", "1"):
                with (
                    self.subTest(role=role, shared=shared),
                    tempfile.TemporaryDirectory() as d,
                ):
                    root = Path(d)
                    (root / "ENDED").touch()
                    # Record the fatal sidecar request instead of broadcasting SIGTERM.
                    script = (
                        """
exit-dramatically() { touch "$LOG_DIR/fatal_requested"; exit 1; }
sleep() { command sleep 0.02; }
"""
                        + fragment
                        + """
command sleep 0.15
wait || true
"""
                    )
                    subprocess.run(
                        ["bash", "-c", script],
                        env={**os.environ, "LOG_DIR": d, "RAY_SINGLE_SRUN": shared},
                        check=True,
                        timeout=5,
                        capture_output=True,
                    )
                    self.assertEqual((root / "fatal_requested").exists(), shared == "0")


if __name__ == "__main__":
    unittest.main()
