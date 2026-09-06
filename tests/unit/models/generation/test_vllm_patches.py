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

"""Guards for the vLLM source patches that had no coverage.

The two port patches ship their own suites. These cover the remaining patches:

* ``_patch_vllm_tool_parser_namespace_tool`` is the most load-bearing patch in
  the repo -- it is the only thing that makes vLLM 0.25.1 importable against
  the pinned ``openai==2.6.1``. If upstream reorders that import block the
  patch logs a warning and returns, and every engine then dies on
  ``import vllm.tool_parsers``. So the anchor needs pinning.
* ``_patch_vllm_glm_decoder_sequence_parallel_moe`` restores the vLLM 0.24
  decoder boundary for GLM-5.1/5.2 while leaving MoE-local SP enabled.
* the ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`` merge replaced the old
  ``ADDITIONAL_ENV_VARS`` file patch and is what now carries
  ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` and every user ``extra_env_vars`` to the
  Ray workers. Being additive rather than clobbering is the whole point of the
  rewrite, and it is pure string handling, so it is cheap to pin.
"""

import ast
import logging
import os
import sys
import types

import pytest

from nemo_rl.models.generation.vllm import patches
from tests.unit.models.generation.vllm_patch_source_utils import (
    write_unpatched_copy,
)

_TOOL_PARSER_SOURCE = "tool_parsers/utils.py"
_PATCH_FN = "_patch_vllm_tool_parser_namespace_tool"
_MARKER = "except ImportError:  # openai < 2.25.0 predates namespace tools"
_RADIO_SOURCE = "model_executor/models/radio.py"
_RADIO_PATCH_FN = "_patch_vllm_radio_layerscale_loader"
_RADIO_MARKER = "initializer_factor = self.config.initializer_factor"
_GLM_DSA_SOURCE = "model_executor/models/deepseek_v2.py"
_GLM_DSA_PATCH_FN = "_patch_vllm_glm_decoder_sequence_parallel_moe"
_GLM_DSA_MARKER = 'getattr(config, "model_type", None) != "glm_moe_dsa"'
_XIELU_SOURCE = "model_executor/layers/activation.py"
_XIELU_PATCH_FN = "_patch_vllm_xielu_static_constants"
_XIELU_MARKER = '"beta", torch.tensor(beta, dtype=dtype), persistent=False'
_APERTUS_SOURCE = "model_executor/models/apertus.py"
_APERTUS_PATCH_FN = "_patch_vllm_apertus_static_xielu_loader"
_APERTUS_MARKER = "Apertus xIELU architecture constant"
_FLASHINFER_AR_SOURCE = "distributed/device_communicators/flashinfer_all_reduce.py"
_MNNVL_PATCH_FN = "_patch_vllm_invalid_mnnvl_workspace"
_MNNVL_MARKER = 'backend == "mnnvl" and not getattr(workspace, "mc_ptr", 0)'


@pytest.fixture
def patched_tool_parser_source(tmp_path, monkeypatch):
    """The installed tool_parsers/utils.py, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_TOOL_PARSER_SOURCE, _PATCH_FN, tmp_path / "utils.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_radio_source(tmp_path, monkeypatch):
    """The installed vLLM RADIO loader, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_RADIO_SOURCE, _RADIO_PATCH_FN, tmp_path / "radio.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_glm_dsa_source(tmp_path, monkeypatch):
    """The installed GLM/DeepSeek model source, unpatched then patched in tmp."""
    copied = write_unpatched_copy(
        _GLM_DSA_SOURCE, _GLM_DSA_PATCH_FN, tmp_path / "deepseek_v2.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_glm_decoder_sequence_parallel_moe(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_xielu_source(tmp_path, monkeypatch):
    copied = write_unpatched_copy(
        _XIELU_SOURCE, _XIELU_PATCH_FN, tmp_path / "activation.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_xielu_static_constants(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_apertus_source(tmp_path, monkeypatch):
    copied = write_unpatched_copy(
        _APERTUS_SOURCE, _APERTUS_PATCH_FN, tmp_path / "apertus.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_apertus_static_xielu_loader(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_flashinfer_ar_source(tmp_path, monkeypatch):
    copied = write_unpatched_copy(
        _FLASHINFER_AR_SOURCE,
        _MNNVL_PATCH_FN,
        tmp_path / "flashinfer_all_reduce.py",
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_invalid_mnnvl_workspace(logging.getLogger(__name__))
    return copied


@pytest.mark.vllm
def test_namespace_tool_patch_anchor_still_matches_installed_vllm(
    patched_tool_parser_source,
):
    """A source edit becomes a silent no-op if upstream reorders the import."""
    content = patched_tool_parser_source.read_text()
    assert _MARKER in content, (
        "the NamespaceTool compat patch did not apply to the installed vLLM; "
        "its anchor import block has probably changed upstream. Every vLLM "
        "engine will fail to import tool_parsers against the pinned openai."
    )
    ast.parse(content)  # the edit must leave valid Python


@pytest.mark.vllm
def test_namespace_tool_patch_is_idempotent(patched_tool_parser_source, monkeypatch):
    """Every worker on a node runs the patch against the same file."""
    before = patched_tool_parser_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_tool_parser_source)
    )
    patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    assert patched_tool_parser_source.read_text() == before


@pytest.mark.vllm
def test_namespace_tool_stub_never_matches(patched_tool_parser_source):
    """The stub must be a plain class, so isinstance() is always False.

    All upstream uses are ``isinstance(tool, NamespaceTool)`` guarding a
    namespace-tools branch, so degrading to "no namespace tools" is correct for
    a client that cannot construct them -- but only if nothing can be an
    instance of the stub.
    """
    namespace: dict = {}
    tree = ast.parse(patched_tool_parser_source.read_text())
    stub = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "NamespaceTool"
    )
    exec(compile(ast.Module(body=[stub], type_ignores=[]), "<stub>", "exec"), namespace)
    stub_cls = namespace["NamespaceTool"]
    for value in ({}, "tool", 0, None, object()):
        assert not isinstance(value, stub_cls)


@pytest.mark.vllm
def test_radio_layerscale_patch_anchor_still_matches_installed_vllm(
    patched_radio_source,
):
    """Pin the vLLM 0.25.1 RADIO loader shape used by the source patch."""
    content = patched_radio_source.read_text()
    assert _RADIO_MARKER in content
    assert "Skip layer-scale entries that vLLM doesn't use" not in content
    ast.parse(content)


@pytest.mark.vllm
def test_radio_layerscale_patch_loads_explicit_and_initializes_folded_weights(
    patched_radio_source,
):
    content = patched_radio_source.read_text()
    assert 'vllm_key = f"model.encoder.layers.{layer_idx}.{suffix}"' in content
    assert 'name.endswith((".ls1", ".ls2"))' in content
    assert "param.data.fill_(initializer_factor)" in content
    assert "loaded_params.add(name)" in content


@pytest.mark.vllm
def test_radio_layerscale_patch_is_idempotent(patched_radio_source, monkeypatch):
    before = patched_radio_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_radio_source)
    )

    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert patched_radio_source.read_text() == before


def test_radio_layerscale_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    radio_source = tmp_path / "radio.py"
    radio_source.write_text("class RadioModel:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(radio_source))

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert radio_source.read_text() == "class RadioModel:\n    pass\n"
    assert "vLLM 0.25.1 source shape was not found" in caplog.text


@pytest.mark.vllm
def test_xielu_constants_are_non_persistent(patched_xielu_source):
    content = patched_xielu_source.read_text()
    assert _XIELU_MARKER in content
    assert '"eps", torch.tensor(eps, dtype=dtype), persistent=False' in content
    assert 'if "beta" in self.state_dict() or "eps" in self.state_dict():' in content
    ast.parse(content)


@pytest.mark.vllm
def test_apertus_loader_validates_static_xielu(patched_apertus_source):
    content = patched_apertus_source.read_text()
    assert _APERTUS_MARKER in content
    assert 'getattr(self, "_nrl_xielu_static_buffers", None)' in content
    assert "if not torch.equal(expected, received):" in content
    ast.parse(content)


@pytest.mark.vllm
@pytest.mark.parametrize(
    "fixture_name,patch_fn_name",
    [
        ("patched_xielu_source", _XIELU_PATCH_FN),
        ("patched_apertus_source", _APERTUS_PATCH_FN),
    ],
)
def test_apertus_xielu_patches_are_idempotent(
    fixture_name, patch_fn_name, request, monkeypatch
):
    source = request.getfixturevalue(fixture_name)
    before = source.read_text()
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))
    getattr(patches, patch_fn_name)(logging.getLogger(__name__))
    assert source.read_text() == before


@pytest.mark.vllm
def test_invalid_mnnvl_workspace_patch_matches_installed_vllm(
    patched_flashinfer_ar_source,
):
    content = patched_flashinfer_ar_source.read_text()
    assert _MNNVL_MARKER in content
    assert "workspace.destroy()" in content
    assert "return None" in content
    ast.parse(content)


@pytest.mark.vllm
def test_glm_decoder_sp_moe_patch_anchor_still_matches_installed_vllm(
    patched_glm_dsa_source,
):
    """Pin the vLLM 0.25.1 decoder-level SP-MoE source shape."""
    content = patched_glm_dsa_source.read_text()
    assert _GLM_DSA_MARKER in content
    ast.parse(content)


@pytest.mark.vllm
def test_glm_decoder_sp_moe_patch_is_idempotent(patched_glm_dsa_source, monkeypatch):
    before = patched_glm_dsa_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_glm_dsa_source)
    )

    patches._patch_vllm_glm_decoder_sequence_parallel_moe(logging.getLogger(__name__))

    assert patched_glm_dsa_source.read_text() == before


def test_glm_decoder_sp_moe_patch_warns_on_unknown_source(
    monkeypatch, tmp_path, caplog
):
    model_source = tmp_path / "deepseek_v2.py"
    model_source.write_text("class DeepseekV2DecoderLayer:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(model_source))

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_glm_decoder_sequence_parallel_moe(
            logging.getLogger(__name__)
        )

    assert model_source.read_text() == "class DeepseekV2DecoderLayer:\n    pass\n"
    assert "vLLM 0.25.1 source shape was not found" in caplog.text


def test_source_compat_applies_all_independent_patches(monkeypatch):
    applied = []
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_logger = types.ModuleType("vllm.logger")
    fake_logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", fake_logger)

    names = [
        ("_patch_vllm_tool_parser_namespace_tool", "tool-parser"),
        ("_patch_vllm_xielu_static_constants", "xielu"),
        ("_patch_vllm_apertus_static_xielu_loader", "apertus"),
        ("_patch_vllm_radio_layerscale_loader", "radio"),
        ("_patch_vllm_glm_decoder_sequence_parallel_moe", "glm"),
        ("_patch_vllm_invalid_mnnvl_workspace", "mnnvl"),
    ]
    for function_name, label in names:
        monkeypatch.setattr(
            patches,
            function_name,
            lambda _logger, label=label: applied.append(label),
        )

    patches.ensure_vllm_source_compat()

    assert applied == [label for _, label in names]


@pytest.mark.parametrize(
    "existing,extra,expected",
    [
        (None, None, "RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        ("", ["MY_VAR"], "MY_VAR,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # A value the caller already set must survive, not be clobbered.
        ("PRESET", ["MY_VAR"], "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # Duplicates collapse and surrounding whitespace is stripped.
        (
            " PRESET , MY_VAR ",
            ["MY_VAR"],
            "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        ),
    ],
)
def test_ray_extra_env_vars_merge_is_additive(
    monkeypatch, tmp_path, existing, extra, expected
):
    """vLLM 0.25 replaced the ADDITIONAL_ENV_VARS source patch with this hook.

    It must add to whatever the caller already set rather than overwrite it --
    otherwise user ``extra_env_vars`` silently stop reaching the Ray workers.
    """
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    if existing is None:
        monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)
    else:
        monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", existing)

    patches._patch_vllm_init_workers_ray("py", extra)

    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == expected


def test_init_workers_ray_reports_a_missing_anchor(monkeypatch, tmp_path):
    """A reshaped call site must not be reported as a successful patch."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray_renamed(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))
    monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)

    assert patches._patch_vllm_init_workers_ray("py", None) is False
    # The env merge still has to happen; it is independent of the file patch.
    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == (
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV"
    )


def test_init_workers_ray_reports_success_and_is_idempotent(monkeypatch, tmp_path):
    """Patching twice against the same file still reports success."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    once = ray_executor.read_text()
    assert 'runtime_env={"py_executable": "py-exec"}' in once

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    assert ray_executor.read_text() == once


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("patch_fn_name", "message"),
    [
        (_XIELU_PATCH_FN, "Required vLLM xIELU static-constant patch"),
        (_APERTUS_PATCH_FN, "Required vLLM Apertus static xIELU loader patch"),
    ],
)
def test_apertus_xielu_patch_anchor_miss_is_fatal(
    patch_fn_name,
    message,
    tmp_path,
    monkeypatch,
):
    unrelated_source = tmp_path / "unexpected_vllm_source.py"
    unrelated_source.write_text("# incompatible vLLM source shape\n")
    monkeypatch.setattr(
        patches,
        "_get_vllm_file",
        lambda _relative: str(unrelated_source),
    )

    with pytest.raises(RuntimeError, match=message):
        getattr(patches, patch_fn_name)(logging.getLogger(__name__))


@pytest.mark.vllm
def test_invalid_mnnvl_workspace_patch_is_idempotent(
    patched_flashinfer_ar_source, monkeypatch
):
    """Multiple generation workers can patch the shared wheel concurrently."""
    before = patched_flashinfer_ar_source.read_text()
    monkeypatch.setattr(
        patches,
        "_get_vllm_file",
        lambda _relative: str(patched_flashinfer_ar_source),
    )
    patches._patch_vllm_invalid_mnnvl_workspace(logging.getLogger(__name__))
    assert patched_flashinfer_ar_source.read_text() == before
