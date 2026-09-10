# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSCS image builder tests with isolated container and storage services."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.unit.tools.test_image_build_manifest import TOOL, _ManifestFixture


class _BuilderFixture(_ManifestFixture):
    """Exercise the real launcher, replacing only container/storage services."""

    def setUp(self):
        super().setUp()
        self.services = tempfile.TemporaryDirectory()
        self.addCleanup(self.services.cleanup)
        self.external = Path(self.services.name)
        site = self.repo / "infra/slurm/cscs"
        site.mkdir(parents=True)
        source_site = TOOL.parents[1] / "infra/slurm/cscs"
        (self.repo / "tools/image_build_manifest.py").write_bytes(TOOL.read_bytes())
        (site / "image_release_receipt.py").write_bytes(
            (source_site / "image_release_receipt.py").read_bytes()
        )
        shutil.copytree(source_site / "profiles", site / "profiles")
        launcher = TOOL.parents[1] / "infra/slurm/cscs/build_nemo_rl_image.slurm"
        helper = "#!/bin/sh\nexit 0\n"
        self.launcher = self.external / "build-image.slurm"
        # Exercise the real checksum guard against a small downloaded fixture.
        self.launcher.write_text(launcher.read_text())
        (self.repo / "infra/slurm/cscs/image_storage.sh").write_text(
            (launcher.parent / "image_storage.sh")
            .read_text()
            .replace(
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
        if os.environ.get('FAKE_ASSEMBLY'):
            raise SystemExit('Assembly must never invoke build')
        raise SystemExit(int(os.environ.get('FAKE_BUILD_EXIT', '0')))
    elif args[0] == 'push' and '--digestfile' in args:
        Path(args[args.index('--digestfile') + 1]).write_text('sha256:' + 'c' * 64)
    elif args[0] == 'pull' and '@sha256:' in args[-1]:
        raise SystemExit(int(os.environ.get('FAKE_PULL_EXIT', '0')))
    elif args[:2] == ['image', 'inspect']:
        receipt = json.loads(Path(os.environ['RELEASE_RECEIPT']).read_text())
        print(json.dumps([{'Id': 'd' * 64, 'Os': 'linux', 'Architecture': 'arm64',
            'RepoDigests': [receipt['image_ref']], 'Labels': {
                'org.opencontainers.image.revision': receipt['source_commit'],
                'org.opencontainers.image.source-inputs': receipt['build_inputs_sha256']}}]))
    elif args[0] == 'info':
        print(os.environ.get('FAKE_GRAPH_ROOT', os.environ['PODMAN_STORAGE_BASE'] + '/graphroot'))
    elif args[0] == 'run' and '/bin/cat' in args:
        if os.environ.get('FAKE_ASSEMBLY'):
            receipt = json.loads(Path(os.environ['RELEASE_RECEIPT']).read_text())
            payload = receipt['hermetic_manifest']
            if args[-1] == '/opt/nemo_rl_container_fingerprint':
                payload = payload['inputs']['dependency_fingerprint']
                if os.environ.get('FAKE_FINGERPRINT_MISMATCH'):
                    payload = {}
        else:
            payload = json.loads((Path(os.environ['PODMAN_STORAGE_BASE']) / 'hermetic-manifest.json').read_text())
        if os.environ.get('FAKE_EMBEDDED_MODE') == 'mismatch':
            payload['inputs']['profile'] = 'wrong'
        print(json.dumps(payload))
elif name == 'enroot':
    if args[0] == 'import':
        Path(args[args.index('-o') + 1]).write_text('fixture squashfs')
        raise SystemExit(int(os.environ.get('FAKE_EXPORT_EXIT', '0')))
    else:
        raise SystemExit('Qualification must be a separate phase')
elif name == 'unsquashfs':
    raise SystemExit(int(os.environ.get('FAKE_EXPORT_INVALID', '0')))
elif name in {'uv', 'pip', 'gcc', 'nvcc'}:
    raise SystemExit('Assembly must never install or compile')
"""
        )
        service_script.chmod(0o755)
        for name in [
            "podman",
            "curl",
            "lfs",
            "enroot",
            "unsquashfs",
            "uv",
            "pip",
            "gcc",
            "nvcc",
        ]:
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


class BuilderFlowTests(_BuilderFixture):
    def test_selected_workers_match_manifest_and_docker_build(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads(
            next((self.external / "cache/manifests").glob("*.json")).read_text()
        )
        actors = (
            (self.repo / "infra/slurm/cscs/profiles/apertus.actors")
            .read_text()
            .rstrip("\n")
        )
        self.assertEqual(manifest["inputs"]["build_args"]["NRL_ACTORS"], actors)
        self.assertEqual(
            [row.split("\t")[0] for row in manifest["inputs"]["actor_rows"]],
            actors.split(),
        )
        build = next(call for call in self.calls() if call[:2] == ["podman", "build"])
        self.assertIn("NRL_ACTORS=" + actors, build)

    def test_full_profile_selects_all_registered_workers(self):
        result = self.launch(NRL_IMAGE_PROFILE="full")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads(
            next((self.external / "cache/manifests").glob("*.json")).read_text()
        )
        self.assertEqual(manifest["inputs"]["build_args"]["NRL_ACTORS"], "")
        build = next(call for call in self.calls() if call[:2] == ["podman", "build"])
        self.assertIn("NRL_ACTORS=", build)

    def test_empty_site_profile_fails_before_container_operations(self):
        (self.repo / "infra/slurm/cscs/profiles/apertus.actors").write_text("\n\t\n")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Empty Apertus actor profile", result.stderr)
        self.assertEqual(self.calls(), [])

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

    def test_auto_hit_prepares_release_after_embedded_manifest_verification(self):
        result = self.launch(FAKE_HTTP_STATUS="200")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        builds = [call for call in self.calls() if call[:2] == ["podman", "build"]]
        self.assertEqual(len(builds), 2)
        self.assertIn("--target=release-core", builds[0])
        self.assertIn("--target=release", builds[1])
        self.assertTrue(list((self.external / "images").glob("*.release.json")))
        report = next((self.external / "images").glob("*.timings.log")).read_text()
        self.assertNotIn("stage=squashfs-export", report)
        self.assertIn("stage=release-core-build", report)

    def test_build_publishes_receipt_without_exporting_squashfs(self):
        result = self.launch(FAKE_HTTP_STATUS="200")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any(call[0] == "enroot" for call in self.calls()))
        self.assertTrue(list((self.external / "images").glob("*.release.json")))
        self.assertFalse(list((self.external / "images").glob("*.sqsh")))

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
        self.assertIn("PREPARED IMAGE BUILD COMPLETE:", result.stdout)
        self.assertTrue(list((self.external / "images").glob("*.release.json")))

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


class SiteProfileTests(unittest.TestCase):
    def test_apertus_profile_preserves_the_six_worker_environments(self):
        root = TOOL.parents[1]
        actors = (root / "infra/slurm/cscs/profiles/apertus.actors").read_text()
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(root / "nemo_rl/distributed/actor_environments.py"),
                "--actors",
                actors,
            ],
            cwd="/tmp",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("\t") for line in result.stdout.splitlines()]
        self.assertEqual(
            {actor.rsplit(".", 1)[1]: (stage, flags) for actor, stage, flags in rows},
            {
                **{
                    actor: ("deps", "--extra vllm")
                    for actor in (
                        "AsyncTrajectoryCollector",
                        "ReplayBuffer",
                        "SyncRolloutActor",
                    )
                },
                # Token capture imports nemo_gym inside the vLLM workers (#4009).
                **{
                    actor: ("deps", "--extra vllm --extra nemo_gym")
                    for actor in ("VllmGenerationWorker", "VllmAsyncGenerationWorker")
                },
                "MegatronPolicyWorker": ("deps", "--extra mcore"),
            },
        )


if __name__ == "__main__":
    unittest.main()
