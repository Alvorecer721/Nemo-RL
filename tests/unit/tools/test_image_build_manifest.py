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

"""Host-only tests for dependency cache identities and selection."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[3] / "tools/image_build_manifest.py"


class _ManifestFixture(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), "Generated build manifest helper is missing")
        spec = importlib.util.spec_from_file_location("image_build_manifest", TOOL)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        files = {
            "pyproject.toml": '[project]\nname = "fixture"\n',
            "uv.lock": "version = 1\n",
            "tools/generate_fingerprint.py": (
                "import hashlib, json\nfrom pathlib import Path\n"
                "root = Path(__file__).resolve().parents[1]\n"
                "print(json.dumps({p: hashlib.md5((root/p).read_bytes()).hexdigest() "
                "for p in ['pyproject.toml', 'uv.lock']}))\n"
            ),
            "nemo_rl/distributed/actor_environments.py": (
                "import sys\nprofile = sys.argv[sys.argv.index('--profile') + 1]\n"
                "print(f'fixture.{profile}Worker\\tdeps\\t--extra mcore')\n"
            ),
            "tools/build-deps.sh": "echo build\n",
            "tools/release-only.sh": "echo release\n",
            "nemo_rl/application.py": "VALUE = 1\n",
            "research/project/pyproject.toml": '[project]\nname = "research"\n',
            "research/project/train.py": "print('train')\n",
            "docker/Dockerfile": (
                "ARG BASE_IMAGE\nFROM ${BASE_IMAGE} AS base\nRUN echo base\n"
                "FROM base AS hermetic\n"
                "COPY --from=nemo-rl tools/build-deps.sh ./tools/\n"
                "COPY --from=nemo-rl research/ ./research/\n"
                "RUN bash tools/build-deps.sh\n"
                "FROM hermetic AS release-core\n"
                "COPY --from=nemo-rl . /opt/nemo-rl\n"
                "RUN bash tools/release-only.sh\n"
                "FROM release-core AS release\nRUN echo release\n"
            ),
        }
        for relative, content in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        subprocess.run(["git", "init", "--quiet", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        self.options = {
            "base_image": "registry/base@sha256:" + "a" * 64,
            "platform": "linux/arm64",
            "profile": "apertus",
            "build_args": {"NVTE_CUDA_ARCHS": "90", "NVTE_WITH_NCCL_EP": "0"},
        }

    def manifest(self, **overrides):
        return self.tool.generate_manifest(self.repo, **(self.options | overrides))


class ImageBuildManifestTests(_ManifestFixture):
    def test_identity_ignores_application_source_release_recipe_and_mtime(self):
        first = self.manifest()
        (self.repo / "nemo_rl/application.py").write_text("VALUE = 2\n")
        (self.repo / "research/project/train.py").write_text("print('new train')\n")
        dockerfile = self.repo / "docker/Dockerfile"
        dockerfile.write_text(
            dockerfile.read_text().replace("RUN echo release", "RUN echo newer")
        )
        (self.repo / "uv.lock").touch()
        self.assertEqual(first, self.manifest())
        self.assertEqual(first["cache_key"], self.tool.manifest_digest(first["inputs"]))

    def test_dependency_recipe_helpers_metadata_and_actor_changes_invalidate(self):
        original = self.manifest()["cache_key"]
        for name in [
            "uv.lock",
            "pyproject.toml",
            "tools/build-deps.sh",
            "research/project/pyproject.toml",
            "nemo_rl/distributed/actor_environments.py",
        ]:
            with self.subTest(name=name):
                path = self.repo / name
                content = path.read_text()
                path.write_text(content + "\n# changed\n")
                self.assertNotEqual(original, self.manifest()["cache_key"])
                path.write_text(content)
        path = self.repo / "docker/Dockerfile"
        path.write_text(
            path.read_text().replace("RUN echo base", "RUN echo changed-base")
        )
        self.assertNotEqual(original, self.manifest()["cache_key"])

    def test_recipe_comments_do_not_declare_missing_build_inputs(self):
        path = self.repo / "docker/Dockerfile"
        path.write_text(
            "# Historical implementation: nemo_rl/utils/removed.py\n" + path.read_text()
        )
        self.assertTrue(self.manifest()["cache_key"])

    def test_configuration_choices_invalidate(self):
        original = self.manifest()["cache_key"]
        for override in [
            {"base_image": "registry/base@sha256:" + "b" * 64},
            {"platform": "linux/amd64"},
            {"profile": "full"},
            {"build_args": {"NVTE_CUDA_ARCHS": "100"}},
            {"build_args": {"NVTE_WITH_NCCL_EP": "1"}},
            {"build_args": {"SKIP_TRTLLM_BUILD": "1"}},
        ]:
            with self.subTest(override=override):
                self.assertNotEqual(original, self.manifest(**override)["cache_key"])

    def test_nested_submodule_pin_changes_invalidate_and_moved_pins_are_rejected(self):
        with tempfile.TemporaryDirectory() as external:
            external = Path(external)
            inner = external / "inner"
            outer = external / "outer"
            for repository in [inner, outer]:
                repository.mkdir()
                subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
                (repository / "source.txt").write_text("first\n")
                subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
                self.commit(repository)
            self.add_submodule(outer, inner, "nested")
            self.commit(outer)
            self.add_submodule(self.repo, outer, "3rdparty/outer")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo),
                    "-c",
                    "protocol.file.allow=always",
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                ],
                check=True,
                capture_output=True,
            )
            first = self.manifest()
            self.assertIn("3rdparty/outer/nested", first["inputs"]["submodules"])
            nested = self.repo / "3rdparty/outer/nested"
            (nested / "source.txt").write_text("second\n")
            subprocess.run(["git", "-C", str(nested), "add", "."], check=True)
            self.commit(nested)
            with self.assertRaisesRegex(ValueError, "recorded pin"):
                self.manifest()
            subprocess.run(
                ["git", "-C", str(nested.parent), "add", "nested"], check=True
            )
            self.commit(nested.parent)
            subprocess.run(
                ["git", "-C", str(self.repo), "add", "3rdparty/outer"], check=True
            )
            self.assertNotEqual(first["cache_key"], self.manifest()["cache_key"])

    @staticmethod
    def commit(repository):
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )

    @staticmethod
    def add_submodule(repository, source, destination):
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "--quiet",
                str(source),
                destination,
            ],
            check=True,
            capture_output=True,
        )

    def test_mutable_base_and_missing_recipe_boundary_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "digest"):
            self.manifest(base_image="registry/base:latest")
        (self.repo / "docker/Dockerfile").write_text("FROM scratch AS wrong\n")
        with self.assertRaisesRegex(ValueError, "release"):
            self.manifest()

    def test_atomic_write_preserves_previous_manifest_on_replace_failure(self):
        destination = self.repo / "manifest.json"
        destination.write_text('{"old": true}\n')
        with patch.object(self.tool.os, "replace", side_effect=OSError("disk failed")):
            with self.assertRaisesRegex(OSError, "disk failed"):
                self.tool.write_manifest(destination, self.manifest())
        self.assertEqual(json.loads(destination.read_text()), {"old": True})
        self.tool.write_manifest(destination, self.manifest())
        self.assertEqual(json.loads(destination.read_text()), self.manifest())
        self.assertEqual(list(self.repo.glob(".manifest.json.*")), [])

    def test_embedded_manifest_must_match_inputs_and_its_own_digest(self):
        expected = self.manifest()
        self.tool.verify_manifest(expected, expected)
        altered = json.loads(json.dumps(expected))
        altered["inputs"]["profile"] = "full"
        with self.assertRaisesRegex(ValueError, "digest"):
            self.tool.verify_manifest(expected, altered)
        different = self.manifest(profile="full")
        with self.assertRaisesRegex(ValueError, "match"):
            self.tool.verify_manifest(expected, different)
        with self.assertRaises(ValueError):
            self.tool.verify_manifest(expected, {})

    def test_cache_selection_never_reuses_missing_or_mismatched_explicit_pin(self):
        key = "a" * 64
        for mode, exists, target in [
            ("auto", False, "hermetic"),
            ("auto", True, "release-core"),
            ("rebuild", True, "hermetic"),
            (key, True, "release-core"),
        ]:
            self.assertEqual(
                self.tool.select_cache(mode, key, available=exists), target
            )
        for mode, exists in [(key, False), ("b" * 64, True), ("old-manual-tag", True)]:
            with self.subTest(mode=mode, exists=exists), self.assertRaises(ValueError):
                self.tool.select_cache(mode, key, available=exists)


class BuilderFlowTests(_ManifestFixture):
    """Exercise the real launcher, replacing only container/storage services."""

    def setUp(self):
        super().setUp()
        self.services = tempfile.TemporaryDirectory()
        self.addCleanup(self.services.cleanup)
        self.external = Path(self.services.name)
        (self.repo / "tools/image_build_manifest.py").write_bytes(TOOL.read_bytes())
        launcher = TOOL.parents[1] / "infra/slurm/cscs/build_nemo_rl_image.slurm"
        helper = "#!/bin/sh\nexit 0\n"
        self.launcher = self.external / "build-image.slurm"
        # Exercise the real checksum guard against a small downloaded fixture.
        self.launcher.write_text(
            launcher.read_text().replace(
                "82fed736197b2a881a822e5357b488796f654e8371ce8573a1592331510a0133",
                hashlib.sha256(helper.encode()).hexdigest(),
            )
        )
        service_script = self.external / "service"
        service_script.write_text(
            f"#!{sys.executable}\n"
            + """import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['FAKE_LOG'], 'a') as log:
    log.write(json.dumps([name, *args]) + '\\n')
if name == 'curl':
    if '--output' in args and any('fuse-overlayfs/releases/' in arg for arg in args):
        payload = os.environ['FAKE_FUSE_HELPER']
        if os.environ.get('FAKE_FUSE_CORRUPT'):
            payload += 'corrupted'
        Path(args[args.index('--output') + 1]).write_text(payload)
    elif '--write-out' in args:
        print(os.environ.get('FAKE_HTTP_STATUS', '404'), end='')
elif name == 'podman':
    if args[0] == 'build':
        raise SystemExit(int(os.environ.get('FAKE_BUILD_EXIT', '0')))
    elif args[0] == 'info':
        print(os.environ.get('FAKE_GRAPH_ROOT', os.environ['PODMAN_STORAGE_BASE'] + '/graphroot'))
    elif args[0] == 'run' and '/bin/cat' in args:
        payload = json.loads((Path(os.environ['PODMAN_STORAGE_BASE']) / 'hermetic-manifest.json').read_text())
        if os.environ.get('FAKE_EMBEDDED_MODE') == 'mismatch':
            payload['inputs']['profile'] = 'wrong'
        print(json.dumps(payload))
elif name == 'enroot':
    if args[0] == 'import':
        Path(args[args.index('-o') + 1]).write_text('fixture squashfs')
    else:
        sys.stdin.read()
"""
        )
        service_script.chmod(0o755)
        for name in ["podman", "curl", "lfs", "enroot"]:
            (self.external / name).symlink_to(service_script)
        self.environment = dict(os.environ) | {
            "PATH": f"{self.external}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "fixture",
            "SLURM_SUBMIT_DIR": str(self.repo),
            "REPO_DIR": str(self.repo),
            "HOST_PYTHON": sys.executable,
            "CACHE_DIR": str(self.external / "cache"),
            "OUTPUT_DIR": str(self.external / "images"),
            "PODMAN_STORAGE_BASE": str(self.external / "private-store"),
            "REGISTRY_IMAGE_ARCHIVE": str(self.external / "absent.oci"),
            "GIT_CREDENTIALS_FILE": str(self.external / "absent.credentials"),
            "HERMETIC_CACHE_TAG": "auto",
            "NRL_IMAGE_PROFILE": "apertus",
            "BUILD_TRTLLM": "0",
            "FAKE_LOG": str(self.external / "calls.jsonl"),
            "FAKE_FUSE_HELPER": helper,
        }
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        self.bash = shutil.which("bash")

    def launch(self, **environment):
        return subprocess.run(
            [self.bash, str(self.launcher)],
            env=self.environment | environment,
            text=True,
            capture_output=True,
        )

    def calls(self):
        path = self.external / "calls.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def test_auto_miss_publishes_hermetic_and_stops_before_assembly(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("HERMETIC BUILD COMPLETE", result.stdout)
        builds = [call for call in self.calls() if call[:2] == ["podman", "build"]]
        self.assertEqual(len(builds), 1)
        self.assertIn("--target=hermetic", builds[0])
        self.assertTrue(any(call[:2] == ["podman", "push"] for call in self.calls()))
        self.assertFalse(any(call[0] == "enroot" for call in self.calls()))
        self.assertTrue(list((self.external / "cache/manifests").glob("*.json")))

    def test_auto_hit_assembles_and_exports_after_embedded_manifest_verification(self):
        result = self.launch(FAKE_HTTP_STATUS="200")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        builds = [call for call in self.calls() if call[:2] == ["podman", "build"]]
        self.assertEqual(len(builds), 2)
        self.assertIn("--target=release-core", builds[0])
        self.assertIn("--target=release", builds[1])
        self.assertTrue(list((self.external / "images").glob("*.sqsh")))
        report = next((self.external / "images").glob("*.timings.log")).read_text()
        self.assertIn("stage=squashfs-export", report)
        self.assertIn("stage=release-core-build", report)

    def test_mismatched_embedded_image_is_rejected_before_build(self):
        result = self.launch(FAKE_HTTP_STATUS="200", FAKE_EMBEDDED_MODE="mismatch")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("digest", result.stderr)
        self.assertFalse(any(call[:2] == ["podman", "build"] for call in self.calls()))

    def test_registry_errors_are_not_interpreted_as_cache_misses(self):
        result = self.launch(FAKE_HTTP_STATUS="500")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HTTP 500", result.stderr)
        self.assertFalse(any(call[:2] == ["podman", "build"] for call in self.calls()))

    def test_wrong_graph_stops_before_any_mutating_podman_command(self):
        result = self.launch(FAKE_GRAPH_ROOT="/home/shared/containers")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected Podman graph", result.stderr)
        self.assertEqual(
            [call[:2] for call in self.calls() if call[0] == "podman"],
            [["podman", "info"]],
        )
        self.assertFalse(any(call[0] == "enroot" for call in self.calls()))

    def test_corrupted_overlay_helper_is_rejected_before_container_operations(self):
        result = self.launch(FAKE_FUSE_CORRUPT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAILED", result.stdout)
        self.assertFalse(any(call[0] in {"podman", "enroot"} for call in self.calls()))

    def test_existing_store_and_unsupported_profile_combination_fail_early(self):
        Path(self.environment["PODMAN_STORAGE_BASE"]).mkdir()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("new absolute private directory", result.stderr)
        self.assertEqual(self.calls(), [])
        result = self.launch(BUILD_TRTLLM="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires NRL_IMAGE_PROFILE=full", result.stderr)

    def test_rebuild_then_explicit_pin_complete_in_separate_private_stores(self):
        result = self.launch(HERMETIC_CACHE_TAG="rebuild", FAKE_HTTP_STATUS="200")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("HERMETIC BUILD COMPLETE", result.stdout)
        manifest = json.loads(
            next((self.external / "cache/manifests").glob("*.json")).read_text()
        )
        result = self.launch(
            HERMETIC_CACHE_TAG=manifest["cache_key"],
            FAKE_HTTP_STATUS="200",
            PODMAN_STORAGE_BASE=str(self.external / "second-private-store"),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BUILD COMPLETE:", result.stdout)
        self.assertTrue(list((self.external / "images").glob("*.sqsh")))

    def test_dirty_source_is_rejected_before_container_operations(self):
        (self.repo / "nemo_rl/application.py").write_text("dirty = True\n")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("clean committed repository", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_failed_build_is_timed_under_its_actual_stage(self):
        result = self.launch(FAKE_BUILD_EXIT="23")
        self.assertEqual(result.returncode, 23)
        report = next((self.external / "images").glob("*.timings.log")).read_text()
        self.assertRegex(
            report, r"stage=hermetic-build elapsed_seconds=\d+ exit_code=23"
        )
        self.assertFalse(any(call[:2] == ["podman", "push"] for call in self.calls()))


if __name__ == "__main__":
    unittest.main()
