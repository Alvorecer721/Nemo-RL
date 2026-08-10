# Copyright (c) 2026, the Apertus project.
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
"""Contract tests for the vLLM XIELU compile-safety plugin.

The plugin must be inert wherever the CUDA kernel or vLLM is absent, and where
it does apply it must reach *every* XIELU instance -- patching ``forward_native``
alone is not enough, because ``CustomOp.__init__`` snapshots ``_forward_method``
before the plugin can run.

The non-CUDA branch of the patched forward is what these tests drive: it routes
to the stock Python implementation, which distinguishes patched from unpatched
without needing a GPU or the kernel.
"""

import sys
import types

import pytest

from nemo_rl.models.generation.vllm import xielu_patch


class _CpuInput:
    is_cuda = False


def _make_xielu_class():
    """A fresh stand-in per test: vLLM's CustomOp snapshots forward at __init__."""

    class FakeXIELU:
        def __init__(self):
            self._forward_method = self.forward_native
            self.alpha_p = "alpha_p"
            self.alpha_n = "alpha_n"
            self._beta_scalar = 0.5
            self._eps_scalar = -1e-6
            self.with_vector_loads = False

        def forward(self, *args, **kwargs):
            return self._forward_method(*args, **kwargs)

        def forward_native(self, x):
            return ("stock", x)

        def _xielu_python(self, x):
            return ("python", x)

    return FakeXIELU


@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a stand-in vllm.model_executor.layers.activation module."""
    activation = types.ModuleType("vllm.model_executor.layers.activation")
    activation.XIELU = _make_xielu_class()
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.activation", activation
    )
    monkeypatch.setattr(
        sys.modules["vllm.model_executor.layers"],
        "activation",
        activation,
        raising=False,
    )
    return activation


@pytest.fixture
def fake_kernel(monkeypatch):
    """Install a stand-in xielu.ops so the plugin proceeds past its guard."""
    for name in ("xielu", "xielu.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


@pytest.fixture
def no_op_registration(monkeypatch):
    """Count registrations without touching the real torch.library."""
    calls = []
    monkeypatch.setattr(
        xielu_patch, "_register_custom_op", lambda: calls.append(1)
    )
    return calls


def test_noop_without_kernel(monkeypatch, fake_vllm, no_op_registration):
    """No CUDA kernel installed -> vLLM must be left exactly as found."""
    monkeypatch.setitem(sys.modules, "xielu.ops", None)
    monkeypatch.delitem(sys.modules, "xielu", raising=False)
    original = fake_vllm.XIELU.forward_native

    xielu_patch.apply()

    assert fake_vllm.XIELU.forward_native is original
    assert not getattr(fake_vllm.XIELU, "_nemo_rl_compile_safe", False)
    assert no_op_registration == []
    assert fake_vllm.XIELU().forward(_CpuInput())[0] == "stock"


def test_noop_without_vllm(monkeypatch, fake_kernel, no_op_registration):
    """vLLM absent (e.g. a training-only process) -> return quietly."""
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.activation", None)

    xielu_patch.apply()  # must not raise

    assert no_op_registration == []


def test_patch_reaches_an_instance_built_before_it_ran(
    fake_vllm, fake_kernel, no_op_registration
):
    """The snapshot trap that made v0.2.1 ineffective in the real engine."""
    early = fake_vllm.XIELU()
    assert early.forward(_CpuInput())[0] == "stock"

    xielu_patch.apply()

    # forward() resolves the method at call time, so the stale snapshot on the
    # already-constructed instance is bypassed.
    assert early.forward(_CpuInput())[0] == "python"
    assert fake_vllm.XIELU().forward(_CpuInput())[0] == "python"
    assert no_op_registration == [1]


def test_idempotent(fake_vllm, fake_kernel, no_op_registration):
    """vLLM warns plugins may be loaded repeatedly; re-running must be safe."""
    xielu_patch.apply()
    first = fake_vllm.XIELU.forward_native

    xielu_patch.apply()

    assert fake_vllm.XIELU.forward_native is first
    assert fake_vllm.XIELU._nemo_rl_compile_safe is True
    assert no_op_registration == [1], "the op must be registered exactly once"


def test_forward_cuda_is_patched_too(fake_vllm, fake_kernel, no_op_registration):
    """vLLM dispatches through forward_cuda on GPU platforms."""
    xielu_patch.apply()

    instance = fake_vllm.XIELU()
    assert instance.forward_cuda(_CpuInput())[0] == "python"


def test_entry_point_is_declared():
    """The plugin only reaches vLLM's EngineCore process via its entry point."""
    import importlib.metadata as md

    targets = {ep.value for ep in md.entry_points(group="vllm.general_plugins")}
    assert "nemo_rl.models.generation.vllm.xielu_patch:apply" in targets, (
        "vllm.general_plugins entry point missing -- re-sync the project so the "
        "installed dist metadata picks up pyproject.toml's [project.entry-points]."
    )
