# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run real build/assembly scripts with external container services replaced."""

import json
import subprocess

from tests.unit.infra.test_cscs_image_build import TOOL, _BuilderFixture


class AssemblyFlowTests(_BuilderFixture):
    def setUp(self):
        super().setUp()
        result = self.launch(FAKE_HTTP_STATUS="200")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.receipt = next((self.external / "images").glob("*.release.json"))
        self.assembler = self.external / "assemble_nemo_rl_image.slurm"
        source = TOOL.parents[1] / "infra/slurm/cscs/assemble_nemo_rl_image.slurm"
        self.assertTrue(source.is_file(), "Assembly needs its own entry point")
        self.assembler.write_text(source.read_text())
        (self.repo / "infra/slurm/cscs/enroot_podman_cleanup.sh").write_bytes(
            (source.parent / "enroot_podman_cleanup.sh").read_bytes()
        )
        (self.external / "calls.jsonl").unlink()
        self.environment.update(
            {
                "RELEASE_RECEIPT": str(self.receipt),
                "FAKE_ASSEMBLY": "1",
                "PODMAN_STORAGE_BASE": str(self.external / "assembly-store"),
            }
        )

    def assemble(self, **env):
        return subprocess.run(
            [self.bash, str(self.assembler)],
            env=self.environment | env,
            text=True,
            capture_output=True,
        )

    def test_assembly_consumes_exact_digest_and_only_exports(self):
        result = self.assemble()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertFalse(any(call[:2] == ["podman", "build"] for call in calls))
        self.assertFalse(any(call[0] in {"uv", "pip", "gcc", "nvcc"} for call in calls))
        ref = json.loads(self.receipt.read_text())["image_ref"]
        self.assertIn(["podman", "pull", "--tls-verify=false", ref], calls)
        self.assertTrue(
            any(
                call[0] == "enroot" and call[-1] == "podman://" + "d" * 64
                for call in calls
            )
        )
        self.assertTrue(list((self.external / "images").glob("*.sqsh")))
        self.assertFalse(list((self.external / "images").glob("*.partial*")))

    def test_missing_receipt_stops_before_container_operations(self):
        result = self.assemble(RELEASE_RECEIPT=str(self.external / "missing.json"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_missing_image_has_no_build_fallback(self):
        result = self.assemble(FAKE_PULL_EXIT="125")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(
            any(
                call[:2] == ["podman", "build"] or call[0] == "enroot"
                for call in self.calls()
            )
        )

    def test_mismatched_fingerprint_stops_before_export(self):
        result = self.assemble(FAKE_FINGERPRINT_MISMATCH="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fingerprint", result.stderr)
        self.assertFalse(any(call[0] == "enroot" for call in self.calls()))

    def test_bad_export_never_publishes_final_image(self):
        result = self.assemble(FAKE_EXPORT_EXIT="1", FAKE_EXPORT_INVALID="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(list((self.external / "images").glob("*.sqsh")))

    def test_nonzero_export_is_rejected_even_with_a_valid_superblock(self):
        result = self.assemble(FAKE_EXPORT_EXIT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(list((self.external / "images").glob("*.sqsh")))

    def test_zero_export_status_does_not_override_failed_data_validation(self):
        result = self.assemble(FAKE_EXPORT_INVALID="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(list((self.external / "images").glob("*.sqsh")))

    def test_existing_image_is_not_overwritten(self):
        output = self.receipt.with_name(
            self.receipt.name.replace(".release.json", ".sqsh")
        )
        output.write_text("existing artifact")
        result = self.assemble()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "existing artifact")
        self.assertEqual(self.calls(), [])


def test_enroot_cleanup_works_from_deleted_directory(tmp_path):
    import os

    helper = TOOL.parents[1] / "infra/slurm/cscs/enroot_podman_cleanup.sh"
    assert helper.is_file(), "Enroot cleanup must leave its deleted working directory"
    fake = tmp_path / "podman"
    fake.write_text(
        '#!/bin/sh\n[ "$PWD" = / ] || exit 32\nprintf "%s\\n" "$@"\nexit "${FAKE_CLEANUP_EXIT:-0}"\n'
    )
    fake.chmod(0o755)
    deleted = tmp_path / "deleted"
    deleted.mkdir()
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'rmdir "$PWD"; exec bash "$1" rm -f -v enroot.fixture',
            "test",
            str(helper),
        ],
        cwd=deleted,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["rm", "-f", "-v", "--", "enroot.fixture"]
    rejected = subprocess.run(
        ["bash", str(helper), "run", "unrelated"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    failed = subprocess.run(
        ["bash", str(helper), "rm", "-f", "-v", "enroot.fixture"],
        env=env | {"FAKE_CLEANUP_EXIT": "23"},
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 23
