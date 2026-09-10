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

import importlib.util
import json
import subprocess
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
                "for p in ['pyproject.toml', 'uv.lock', 'nemo_rl/distributed/actor_environments.py']}))\n"
            ),
            "nemo_rl/distributed/actor_environments.py": (
                "import argparse\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--actors', default='')\n"
                "p.add_argument('stage', nargs='?', default='all')\n"
                "p.add_argument('skip', nargs='*')\n"
                "a=p.parse_args()\n"
                "for actor in a.actors.split() or ['fixture.Worker']:\n"
                " print(f'{actor}\\tdeps\\t--extra mcore')\n"
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
    def test_explicit_workers_determine_manifest_and_cache_identity(self):
        first = self.manifest(
            profile="custom",
            build_args={"NRL_ACTORS": "fixture.First fixture.Second"},
        )
        second = self.manifest(
            profile="custom", build_args={"NRL_ACTORS": "fixture.Second"}
        )
        self.assertEqual(
            first["inputs"]["actor_rows"],
            [
                "fixture.First\tdeps\t--extra mcore",
                "fixture.Second\tdeps\t--extra mcore",
            ],
        )
        self.assertNotEqual(first["cache_key"], second["cache_key"])

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


if __name__ == "__main__":
    unittest.main()
