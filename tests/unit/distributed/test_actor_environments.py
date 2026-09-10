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

"""Exercise the build/runtime actor contract without importing Ray or GPU packages."""

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import venv
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "nemo_rl/distributed/actor_environments.py"
REGISTRY = ROOT / "nemo_rl/distributed/ray_actor_environment_registry.py"
CONTROLLERS = {"AsyncTrajectoryCollector", "ReplayBuffer", "SyncRolloutActor"}
LEGACY_GROUPS = {
    "--extra vllm": CONTROLLERS | {"VllmGenerationWorker", "VllmAsyncGenerationWorker"},
    "--extra sglang": {"SGLangGenerationWorker"},
    "--extra fsdp": {"DTensorPolicyWorker"},
    "--extra automodel": {"DTensorPolicyWorkerV2", "DTensorValueWorkerV2"},
    "--extra mcore": {"MegatronPolicyWorker", "MegatronValueWorker"},
    "--extra trtllm": {"TrtllmAsyncGenerationWorker"},
    "--extra nemo_gym": {"NemoGym"},
    "--extra modelopt --extra vllm": {
        "VllmQuantGenerationWorker",
        "VllmQuantAsyncGenerationWorker",
    },
    "--extra modelopt --extra automodel": {
        "DTensorQuantPolicyWorker",
        "DTensorQuantPolicyWorkerV2",
    },
    "--extra modelopt --extra mcore": {"MegatronQuantPolicyWorker"},
}
SYSTEM_CLASSES = {
    "BracketMathEnvironment",
    "DynamoVllmWorker",
    "MathEnvironment",
    "MathMultiRewardEnvironment",
    "VLMEnvironment",
    "SingleTurnVerifierEnvironment",
    "CodeEnvironment",
    "RewardModelEnvironment",
    "CodeJaccardEnvironment",
    "SlidingPuzzleEnv",
    "RAGEnvironment",
}


def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_registry(*, system: bool = False, table=None) -> types.ModuleType:
    """Load real registries/constants while isolating Ray and package init side effects."""
    virtual_path = ROOT / "nemo_rl/distributed/virtual_cluster.py"
    tree = ast.parse(virtual_path.read_text())
    constants = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PY_EXECUTABLES"
    )
    virtual = types.ModuleType("nemo_rl.distributed.virtual_cluster")
    virtual.git_root = str(ROOT)
    virtual.sys = sys
    exec(
        compile(
            ast.Module(body=[constants], type_ignores=[]), str(virtual_path), "exec"
        ),
        virtual.__dict__,
    )
    modules = {}
    for name in ("nemo_rl", "nemo_rl.distributed", "nemo_rl.modelopt"):
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / name.replace(".", "/"))]
        modules[name] = package
    modules[virtual.__name__] = virtual
    if table is not None:
        modules["nemo_rl.distributed.actor_environments"] = table
    with (
        patch.dict(sys.modules, modules),
        patch.dict(
            os.environ, {"NEMO_RL_PY_EXECUTABLES_SYSTEM": "1" if system else "0"}
        ),
    ):
        modelopt = load_module(
            "nemo_rl.modelopt.registry", ROOT / "nemo_rl/modelopt/registry.py"
        )
        with patch.dict(sys.modules, {modelopt.__name__: modelopt}):
            return load_module("_isolated_actor_registry", REGISTRY)


class ActorEnvironmentTests(unittest.TestCase):
    def run_cli(
        self, *args: str, script: Path = MANIFEST
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-I", "-S", str(script), *args],
            capture_output=True,
            text=True,
            cwd="/tmp",
        )

    def rows(self, *args: str) -> dict[str, tuple[str, str]]:
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("\t") for line in result.stdout.splitlines()]
        self.assertTrue(rows)
        self.assertTrue(all(len(row) == 3 for row in rows))
        self.assertEqual(rows, sorted(rows))
        self.assertEqual(len(rows), len({row[0] for row in rows}))
        return {actor: (stage, flags) for actor, stage, flags in rows}

    def test_explicit_actor_selection_is_independent_of_site_profiles(self) -> None:
        selected = {
            "nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker",
            "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
        }
        rows = self.rows("--actors", " ".join(sorted(selected)))
        self.assertEqual(set(rows), selected)
        rows = self.rows("--actors", " ".join(sorted(selected)), "all", "vllm")
        self.assertEqual(
            {name.rsplit(".", 1)[1] for name in rows}, {"MegatronValueWorker"}
        )

    def test_unknown_explicit_actor_is_rejected_before_output(self) -> None:
        result = self.run_cli("--actors", "missing.Worker")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("unregistered actors", result.stderr)

    def test_default_selection_preserves_current_worker_environments(self) -> None:
        expected = {
            name: ("trtllm" if flags == "--extra trtllm" else "deps", flags)
            for flags, names in LEGACY_GROUPS.items()
            for name in names
        }
        rows = self.rows()
        self.assertEqual(
            {name.rsplit(".", 1)[1]: row for name, row in rows.items()}, expected
        )
        self.assertEqual(rows, self.rows("--actors", ""))

    def test_skip_extras_reaches_controllers_and_quant_workers(self) -> None:
        rows = self.rows("all", "vllm", "automodel")
        names = {actor.rsplit(".", 1)[1] for actor in rows}
        self.assertFalse(names & CONTROLLERS)
        self.assertNotIn("VllmQuantGenerationWorker", names)
        self.assertNotIn("DTensorQuantPolicyWorker", names)
        self.assertIn("MegatronPolicyWorker", names)
        self.assertTrue(
            all(
                "vllm" not in flags and "automodel" not in flags
                for _, flags in rows.values()
            )
        )

    def test_stage_selection_emits_only_trtllm(self) -> None:
        rows = self.rows("trtllm")
        self.assertEqual(
            rows,
            {
                "nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker": (
                    "trtllm",
                    "--extra trtllm",
                )
            },
        )

    def test_invalid_selection_fails_before_emitting_rows(self) -> None:
        for args in (("badstage",), ("all", "notanextra")):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr)

    def test_script_runs_from_dependency_layer_without_package(self) -> None:
        self.assertTrue(MANIFEST.is_file(), "actor manifest is missing")
        with tempfile.TemporaryDirectory() as directory:
            standalone = Path(directory) / "actor_environments.py"
            shutil.copyfile(MANIFEST, standalone)
            result = self.run_cli(script=standalone)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            len(result.stdout.splitlines()), sum(map(len, LEGACY_GROUPS.values()))
        )

    def test_docker_manifest_invocation_without_python_on_path(self) -> None:
        selected = {
            "nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker",
            "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
        }
        docker_lines = (
            (ROOT / "docker/Dockerfile").read_text().replace("\\\n", "").splitlines()
        )
        invocation = next(
            line
            for line in docker_lines
            if "nemo_rl/distributed/actor_environments.py" in line
            and not line.startswith("COPY")
        ).replace("/opt/actor_venvs.tsv", '"$manifest_output"')
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            prefix = fixture / "driver venv"
            venv.EnvBuilder(with_pip=False).create(prefix)
            empty_path = fixture / "empty-path"
            empty_path.mkdir()
            output = fixture / "actors.tsv"
            result = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    "SKIP_EXTRAS=(trtllm)\n" + invocation,
                ],
                cwd=ROOT,
                env={
                    "PATH": str(empty_path),
                    "UV_PROJECT_ENVIRONMENT": str(prefix),
                    "NRL_ACTORS": "\n".join(sorted(selected)),
                    "manifest_output": str(output),
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [line.split("\t") for line in output.read_text().splitlines()]
        self.assertEqual({row[0] for row in rows}, selected)
        self.assertTrue(all(row[1] == "deps" for row in rows))

    def test_registry_keeps_legacy_commands_and_system_override(self) -> None:
        for system in (False, True):
            with self.subTest(system=system):
                registry = load_registry(system=system)
                expected = {name: sys.executable for name in SYSTEM_CLASSES}
                for flags, names in LEGACY_GROUPS.items():
                    for name in names:
                        override = (
                            system
                            and name not in CONTROLLERS
                            and (
                                flags
                                in (
                                    "--extra vllm",
                                    "--extra sglang",
                                    "--extra mcore",
                                    "--extra trtllm",
                                )
                                or "modelopt" in flags
                            )
                        )
                        expected[name] = (
                            sys.executable
                            if override
                            else f"uv run --locked {flags} --directory {ROOT}"
                        )
                actual = {
                    actor.rsplit(".", 1)[1]: command
                    for actor, command in registry.ACTOR_ENVIRONMENT_REGISTRY.items()
                }
                self.assertEqual(actual, expected)
                with self.assertRaisesRegex(
                    ValueError, "No actor environment registered"
                ):
                    registry.get_actor_python_env("missing.Actor")

    def test_runtime_and_build_share_all_actor_definitions(self) -> None:
        self.assertTrue(MANIFEST.is_file(), "actor manifest is missing")
        table = load_module("_actor_manifest", MANIFEST)
        registry = load_registry(table=table)
        self.assertEqual(
            set(registry.ACTOR_ENVIRONMENT_REGISTRY), set(table.ACTOR_ENVIRONMENTS)
        )
        self.assertEqual(
            set(self.rows()),
            {
                actor
                for actor, extras in table.ACTOR_ENVIRONMENTS.items()
                if extras is not None
            },
        )
        table.ACTOR_ENVIRONMENTS["nemo_rl.test.NewWorker"] = ["mcore"]
        refreshed = load_registry(table=table)
        self.assertEqual(
            refreshed.get_actor_python_env("nemo_rl.test.NewWorker"),
            f"uv run --locked --extra mcore --directory {ROOT}",
        )

    def test_registry_rejects_undeclared_extra_at_import(self) -> None:
        self.assertTrue(MANIFEST.is_file(), "actor manifest is missing")
        table = load_module("_invalid_actor_manifest", MANIFEST)
        table.ACTOR_ENVIRONMENTS["nemo_rl.test.NewWorker"] = ["not_a_declared_extra"]
        with self.assertRaisesRegex(ValueError, "not_a_declared_extra"):
            load_registry(table=table)


class ActorFingerprintTests(unittest.TestCase):
    def test_actor_manifest_change_invalidates_fingerprint(self) -> None:
        module = load_module(
            "_generate_fingerprint", ROOT / "tools/generate_fingerprint.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            (fixture / "pyproject.toml").write_text("[project]\n")
            (fixture / "uv.lock").write_text("version = 1\n")
            actor_path = fixture / "nemo_rl/distributed/actor_environments.py"
            actor_path.parent.mkdir(parents=True)
            actor_path.write_text("ACTOR_ENVIRONMENTS = {'Worker': ['vllm']}\n")
            with patch.object(module, "get_repo_root", return_value=fixture):
                before = module.generate_fingerprint()
                actor_path.write_text("ACTOR_ENVIRONMENTS = {'Worker': ['mcore']}\n")
                after = module.generate_fingerprint()
        self.assertNotEqual(before, after)
        self.assertEqual(before["pyproject.toml"], after["pyproject.toml"])
        self.assertEqual(before["uv.lock"], after["uv.lock"])
        self.assertNotEqual(
            before["nemo_rl/distributed/actor_environments.py"],
            after["nemo_rl/distributed/actor_environments.py"],
        )


if __name__ == "__main__":
    unittest.main()
