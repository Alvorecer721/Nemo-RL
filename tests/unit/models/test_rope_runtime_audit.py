"""Reject a requested factor 32 that never reaches the model's tensors."""

import ast
import asyncio
import importlib
import math
import os
from pathlib import Path
import runpy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

audit = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "nemo_rl/models/rope_runtime_audit.py")
)
check_cache = audit["check_cache"]
check_frequencies = audit["check_frequencies"]


def reference(factor):
    values = []
    for j in range(64):
        frequency = 4_000_000 ** (-2 * j / 128)
        wavelength = 2 * math.pi / frequency
        if wavelength > 8192:
            frequency /= factor
        elif wavelength >= 2048:
            smooth = (8192 / wavelength - 1) / 3
            frequency *= smooth + (1 - smooth) / factor
        values.append(frequency)
    return torch.tensor(values, dtype=torch.float32)


class TestRopeRuntimeAudit(unittest.TestCase):
    def test_actual_async_hook_runs_even_when_hf_overrides_are_missing(self):
        # Load the actual method and its standard-library imports without
        # importing Ray/vLLM or starting an engine in this CPU unit test.
        source = (
            Path(__file__).resolve().parents[3]
            / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
        )
        tree = ast.parse(source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "VllmAsyncGenerationWorkerImpl"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "post_init_async"
        )
        namespace = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("asyncio", "os"):
                        namespace[alias.asname or alias.name] = importlib.import_module(
                            alias.name
                        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        rpc = AsyncMock()
        worker = SimpleNamespace(
            llm=SimpleNamespace(collective_rpc=rpc),
            report_device_id_async=AsyncMock(return_value=["gpu0"]),
            cfg={"vllm_cfg": {}, "vllm_kwargs": {"hf_overrides": {}}},
            _sparse_refit_receiver=None,
            _mtp_load_from_disk=False,
        )
        with patch.dict(
            os.environ,
            {
                "NRL_ROPE_AUDIT_DIR": "/tmp/audit-hook-test",
                "NRL_ROPE_AUDIT_FACTOR": "32",
            },
        ):
            asyncio.run(namespace["post_init_async"](worker))
        rpc.assert_any_await("verify_rope_runtime", args=(32.0,))

    def test_actual_factor32_passes(self):
        check_frequencies(reference(32), 32)

    def test_factor8_rejected_despite_requested32(self):
        with self.assertRaisesRegex(ValueError, "frequencies"):
            check_frequencies(reference(8), 32)

    def test_missing_or_nonfinite_frequency_rejected(self):
        for bad in [torch.ones(32), torch.full((64,), float("nan"))]:
            with self.assertRaises(ValueError):
                check_frequencies(bad, 32)

    def test_actual_cache_checked_not_only_recomputed_frequencies(self):
        positions = torch.tensor([1, 2048, 8192, 12287], dtype=torch.float32)
        for dtype in [torch.float32, torch.bfloat16]:
            phases = positions[:, None] * reference(32)[None, :]
            cache = torch.cat([phases.cos(), phases.sin()], dim=-1).to(dtype)
            check_cache(cache, positions, 32)
            wrong = positions[:, None] * reference(8)[None, :]
            stale_cache = torch.cat([wrong.cos(), wrong.sin()], dim=-1).to(dtype)
            with self.assertRaisesRegex(ValueError, "cache"):
                check_cache(stale_cache, positions, 32)


if __name__ == "__main__":
    unittest.main()
