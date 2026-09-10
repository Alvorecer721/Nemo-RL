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

"""Exercise the image qualification command using real isolated interpreters."""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[3] / "tools/check_image_workers.py"
ROOT = SCRIPT.parent.parent


class ImageWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "actors.tsv"
        self.report = self.root / "report.json"

    def run_check(self, actor="json.JSONDecoder", create=True):
        self.manifest.write_text(f"{actor}\tdeps\t\n")
        if create:
            venv.EnvBuilder(with_pip=False).create(self.root / actor)
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(self.manifest),
                "--venv-root",
                str(self.root),
                "--output",
                str(self.report),
            ],
            capture_output=True,
            text=True,
        )

    def test_real_interpreter_import_and_offline_environment(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(self.report.read_text())
        self.assertTrue(report["passed"])
        worker = report["workers"][0]
        self.assertEqual(worker["actor"], "json.JSONDecoder")
        self.assertEqual(worker["prefix"], str(self.root / "json.JSONDecoder"))
        self.assertEqual(worker["offline"], "1")

    def test_missing_interpreter_is_a_failed_report(self):
        result = self.run_check(create=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.report.exists(), result.stderr)
        self.assertFalse(json.loads(self.report.read_text())["passed"])

    def test_unimportable_actor_fails_even_with_valid_interpreter(self):
        result = self.run_check("json.NonexistentActor")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.report.exists(), result.stderr)
        self.assertIn("NonexistentActor", self.report.read_text())

    def test_empty_manifest_cannot_pass(self):
        self.manifest.write_text("")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(self.manifest),
                "--venv-root",
                str(self.root),
                "--output",
                str(self.report),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_qualification_removes_runtime_bypass_variables(self):
        actor = "envprobe.Worker"
        prefix = self.root / actor
        venv.EnvBuilder(with_pip=False).create(prefix)
        site = next((prefix / "lib").glob("python*/site-packages"))
        (site / "envprobe.py").write_text(
            "import os\n"
            "for key in ('NRL_IGNORE_VERSION_MISMATCH', 'NRL_FORCE_REBUILD_VENVS', "
            "'NEMO_RL_PY_EXECUTABLES_SYSTEM'):\n"
            "    if key in os.environ: raise RuntimeError('unsafe override: ' + key)\n"
            "class Worker: pass\n"
        )
        with patch.dict(
            os.environ,
            {
                "NRL_IGNORE_VERSION_MISMATCH": "1",
                "NRL_FORCE_REBUILD_VENVS": "true",
                "NEMO_RL_PY_EXECUTABLES_SYSTEM": "1",
            },
        ):
            result = self.run_check(actor, create=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class NemoWorkerTests(unittest.TestCase):
    """Use a tiny NeMo-shaped package and the real readiness/fingerprint functions."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.actor = "nemo_rl.fixture.Worker"
        self.prefix = self.root / self.actor
        venv.EnvBuilder(with_pip=False).create(self.prefix)
        self.source = next((self.prefix / "lib").glob("python*/site-packages"))
        for package in ("nemo_rl", "nemo_rl/distributed", "nemo_rl/utils"):
            path = self.source / package
            path.mkdir()
            (path / "__init__.py").write_text("")
        (self.source / "nemo_rl/fixture.py").write_text("class Worker: pass\n")
        (self.source / "nemo_rl/distributed/actor_environments.py").write_text(
            "ACTOR_ENVIRONMENTS = {'nemo_rl.fixture.Worker': []}\n"
        )
        self.command = f"uv run --locked --directory {self.source}"
        (
            self.source / "nemo_rl/distributed/ray_actor_environment_registry.py"
        ).write_text(f"def get_actor_python_env(actor): return {self.command!r}\n")
        tree = ast.parse((ROOT / "nemo_rl/utils/venvs.py").read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "_normalized_worker_command",
                "_dependency_fingerprint",
                "venv_is_current",
            }
        ]
        readiness = (
            "import hashlib, os, shlex\nfrom pathlib import Path\n"
            "from functools import lru_cache\n"
            f"git_root = {str(self.source)!r}\n"
            + ast.unparse(ast.Module(body=functions, type_ignores=[]))
        )
        (self.source / "nemo_rl/utils/venvs.py").write_text(readiness)
        self.project = self.source / "pyproject.toml"
        self.project.write_text(
            '[project]\nname = "qualification-fixture"\nversion = "1.0"\n'
            'requires-python = ">=3.11"\ndependencies = []\n'
            "[project.optional-dependencies]\ntrtllm = []\n"
        )
        self.environment = {
            **os.environ,
            "UV": shutil.which("uv"),
            "UV_CACHE_DIR": str(self.root / "uv-cache"),
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_OFFLINE": "1",
        }
        subprocess.run(
            [
                self.environment["UV"],
                "--no-config",
                "lock",
                "--directory",
                str(self.source),
            ],
            env=self.environment,
            capture_output=True,
            text=True,
            check=True,
        )
        namespace = {}
        exec(readiness, namespace)
        self.marker = self.prefix / "NEMO_RL_VENV_READY"
        self.marker.write_text(namespace["_dependency_fingerprint"](self.command))
        (self.source / "tools").mkdir()
        shutil.copyfile(
            ROOT / "tools/generate_fingerprint.py",
            self.source / "tools/generate_fingerprint.py",
        )
        self.fingerprint_path = self.root / "container-fingerprint.json"
        self.fingerprint_path.write_text(
            subprocess.check_output(
                [sys.executable, str(self.source / "tools/generate_fingerprint.py")],
                text=True,
            )
        )
        spec = importlib.util.spec_from_file_location("image_workers", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def check(self, extras="", **kwargs):
        with (
            chdir(self.source),
            patch.dict(os.environ, self.environment),
            patch.object(
                self.module,
                "CONTAINER_FINGERPRINT_PATH",
                self.fingerprint_path,
                create=True,
            ),
        ):
            return self.module.check_worker(self.actor, extras, self.root, 30, **kwargs)

    def test_current_marker_and_matching_fingerprint_pass(self):
        report = self.check()
        self.assertTrue(report["passed"], report)

    def test_stale_readiness_marker_fails(self):
        self.marker.write_text("old-but-nonempty-fingerprint")
        report = self.check()
        self.assertFalse(report["passed"], report)
        self.assertIn("readiness", report["error"].lower())

    def test_missing_malformed_or_mismatching_container_fingerprint_fails(self):
        valid = json.loads(self.fingerprint_path.read_text())
        for value in (None, "{", "[]", "{}", json.dumps({**valid, "uv.lock": "other"})):
            with self.subTest(value=value):
                if value is None:
                    self.fingerprint_path.unlink(missing_ok=True)
                else:
                    self.fingerprint_path.write_text(value)
                report = self.check()
                self.assertFalse(report["passed"], report)
                self.assertIn("fingerprint", report["error"].lower())

    def test_missing_deferred_trt_backend_cannot_pass_actor_import(self):
        report = self.check("trtllm")
        self.assertFalse(report["passed"], report)
        self.assertIn("tensorrt_llm", report["error"])

    def test_trt_native_import_is_explicitly_deferred_on_cpu(self):
        (self.source / "tensorrt_llm.py").write_text(
            "raise RuntimeError('native initializer requires GPU qualification')\n"
        )
        report = self.check("trtllm")
        self.assertTrue(report["passed"], report)
        self.assertFalse(report["gpu_qualified"])
        self.assertIn("tensorrt_llm", report["deferred_checks"])

    def test_wrong_installed_version_fails_without_repairing_environment(self):
        dependency = self.source / "locked-dep"
        dependency.mkdir()
        (dependency / "pyproject.toml").write_text(
            '[project]\nname = "locked-dep"\nversion = "1.0"\n'
            'requires-python = ">=3.11"\n'
            '[build-system]\nrequires = []\nbuild-backend = "fixture_backend"\n'
        )
        self.project.write_text(
            self.project.read_text().replace(
                "dependencies = []", 'dependencies = ["locked-dep==1.0"]'
            )
            + '\n[tool.uv.sources]\nlocked-dep = {path = "locked-dep", editable = true}\n'
        )
        subprocess.run(
            [self.environment["UV"], "lock", "--directory", str(self.source)],
            env=self.environment,
            capture_output=True,
            text=True,
            check=True,
        )
        namespace = {}
        exec((self.source / "nemo_rl/utils/venvs.py").read_text(), namespace)
        self.marker.write_text(namespace["_dependency_fingerprint"](self.command))
        self.fingerprint_path.write_text(
            subprocess.check_output(
                [sys.executable, str(self.source / "tools/generate_fingerprint.py")],
                text=True,
            )
        )
        metadata = self.source / "locked_dep-2.0.dist-info"
        metadata.mkdir()
        metadata_file = metadata / "METADATA"
        original = "Metadata-Version: 2.1\nName: locked-dep\nVersion: 2.0\n"
        metadata_file.write_text(original)
        (metadata / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": dependency.as_uri(),
                    "dir_info": {"editable": True},
                }
            )
        )
        report = self.check()
        self.assertFalse(report["passed"], report)
        self.assertIn("Locked dependency check failed", report["error"])
        self.assertEqual(metadata_file.read_text(), original)

    def test_cpu_check_does_not_claim_gpu_qualification(self):
        report = self.check()
        self.assertTrue(report["passed"], report)
        self.assertIs(report.get("gpu_qualified"), False)

    def test_gpu_requirement_fails_without_cuda(self):
        report = self.check(require_gpu=True)
        self.assertFalse(report["passed"], report)


if __name__ == "__main__":
    unittest.main()
