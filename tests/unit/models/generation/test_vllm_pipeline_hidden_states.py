"""Execute the actual optimized vLLM model's pipeline handoff on CPU tensors."""

import ast
import sys
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.vllm import patches, pipeline_hidden_states


@pytest.fixture
def source_file(monkeypatch, tmp_path):
    original = Path(patches._get_vllm_file(pipeline_hidden_states._SOURCE)).read_text()
    for old, new in pipeline_hidden_states._SOURCE_EDITS:
        original = original.replace(new, old)
    original = original.replace(pipeline_hidden_states._MARKER, "")
    target = tmp_path / "model.py"
    target.write_text(original)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(target))
    monkeypatch.delitem(sys.modules, pipeline_hidden_states._MODULE, raising=False)
    return target


@pytest.fixture
def patched_source(source_file):
    original = source_file.read_text()
    pipeline_hidden_states.patch_pipeline_hidden_states()
    return original, source_file.read_text()


def _forward(source, class_name, namespace):
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    forward = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], forward],
        type_ignores=[],
    )
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(module), class_name, "exec"), scope)
    return scope["forward"]


class _Intermediate(dict):
    pass


class _Norm:
    def __init__(self, name):
        self.name = name

    def __call__(self, value, residual=None):
        combined = value if residual is None else value + residual
        normalized = combined * torch.rsqrt(
            combined.square().mean(-1, keepdim=True) + 1e-5
        )
        return normalized if residual is None else (normalized, combined)


class _Layer:
    def __call__(self, *args, **kwargs):
        return self.forward(self, *args, **kwargs)


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("tokens", [1, 5, 13])
@pytest.mark.parametrize("local_layers", [1, 2])
def test_pp_handoff_replicates_sum_and_normalizes_once(
    patched_source, tp_size, tokens, local_layers
):
    _, source = patched_source
    shape = (tokens, 16)
    base = torch.arange(tokens * 16, dtype=torch.float32).reshape(shape) / 128
    partials = [base * (rank + 1) + rank / 8 for rank in range(tp_size)]
    total = torch.stack(partials).sum(0)
    residual = torch.full(shape, 0.125)
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=False)
    calls = []

    def all_reduce(value):
        calls.append("sender")
        return total.clone()

    def fused_norm(value, residual, norm):
        calls.append(norm.name)
        return norm(value * tp_size, residual)

    namespace = {
        "torch": torch,
        "get_pp_group": lambda: pp,
        "tensor_model_parallel_all_reduce": all_reduce,
        "fused_allreduce_rms_norm": fused_norm,
        "IntermediateTensors": _Intermediate,
        "islice": islice,
    }
    model_forward = _forward(source, "DeepseekV32Model", namespace)
    layer_forward = _forward(source, "DeepseekV32DecoderLayer", namespace)
    positions = torch.arange(tokens)
    sent = []
    for partial in partials:
        model = SimpleNamespace(
            replicated_embed=False,
            embed_input_ids=lambda _: base,
            use_sequence_parallel=False,
            aux_hidden_state_layers=[],
            start_layer=0,
            end_layer=1,
            layers=[
                lambda *args, _partial=partial, **kwargs: (
                    _partial.clone(),
                    residual.clone(),
                )
            ],
        )
        sent.append(model_forward(model, torch.arange(tokens), positions))
    assert calls == ["sender"] * tp_size
    # Actual PP optimization: each TP rank sends a flattened slice, then the
    # receiver all-gathers. This is valid only for replicated sender tensors.
    gathered = torch.cat(
        [
            item["hidden_states"].reshape(tp_size, -1)[rank]
            for rank, item in enumerate(sent)
        ]
    ).reshape(shape)
    torch.testing.assert_close(gathered, total, rtol=0, atol=0)
    assert all(torch.equal(item["hidden_states"], total) for item in sent)

    pp.is_first_rank, pp.is_last_rank = False, True
    first_layer = 19  # Exercise a real uneven PP boundary, not a layer-zero shortcut.
    attention_inputs = []
    decoder_layers = {}
    for index in range(first_layer, first_layer + local_layers):
        layer = _Layer()
        layer.forward = layer_forward
        layer.use_sequence_parallel = False
        layer.input_layernorm = _Norm(f"input-{index}")
        layer.post_attention_layernorm = _Norm(f"post-{index}")

        def attention(*, positions, hidden_states):
            attention_inputs.append(hidden_states.clone())
            return torch.zeros_like(hidden_states)

        layer.self_attn = attention
        layer.mlp = lambda value: torch.full_like(value, 0.05)
        decoder_layers[index] = layer
    model = SimpleNamespace(
        use_sequence_parallel=False,
        aux_hidden_state_layers=[first_layer],
        start_layer=first_layer,
        end_layer=first_layer + local_layers,
        layers=[None] * first_layer + list(decoder_layers.values()),
        norm=_Norm("final"),
    )
    output, auxiliary = model_forward(
        model,
        None,
        positions,
        _Intermediate(hidden_states=gathered, residual=residual.clone()),
    )
    expected_input, combined = _Norm("reference")(total, residual)
    torch.testing.assert_close(attention_inputs[0], expected_input, rtol=0, atol=0)
    torch.testing.assert_close(auxiliary[0], total + residual, rtol=0, atol=0)
    expected_output = _Norm("reference")(combined + 0.05 * tp_size * local_layers)
    torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
    assert f"input-{first_layer}" not in calls
    assert calls.count("sender") == tp_size
    assert calls.count("final") == 1
    for index in range(first_layer, first_layer + local_layers):
        assert calls.count(f"post-{index}") == 1
        assert calls.count(f"input-{index}") == (index != first_layer)


def test_pp1_forward_is_unchanged(patched_source):
    original, updated = patched_source
    outputs = []
    for source in (original, updated):
        pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        namespace = {
            "torch": torch,
            "get_pp_group": lambda: pp,
            "fused_allreduce_rms_norm": lambda value, residual, norm: norm(
                value * 2, residual
            ),
            "IntermediateTensors": _Intermediate,
            "islice": islice,
        }
        layer = _Layer()
        layer.forward = _forward(source, "DeepseekV32DecoderLayer", namespace)
        layer.use_sequence_parallel = False
        layer.input_layernorm = _Norm("input")
        layer.post_attention_layernorm = _Norm("post")
        layer.self_attn = lambda *, positions, hidden_states: hidden_states / 2
        layer.mlp = lambda value: value / 2
        model = SimpleNamespace(
            replicated_embed=False,
            embed_input_ids=lambda _: torch.arange(32).float().reshape(2, 16),
            use_sequence_parallel=False,
            aux_hidden_state_layers=[],
            start_layer=0,
            end_layer=1,
            layers=[layer],
            norm=_Norm("final"),
        )
        outputs.append(
            _forward(source, "DeepseekV32Model", namespace)(
                model, torch.arange(2), torch.arange(2)
            )
        )
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("stage", [0, 1, 3])
def test_empty_pipeline_stage_is_rejected_at_construction(
    patched_source, monkeypatch, stage
):
    _, source = patched_source
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV32Model"
    )
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], cls],
        type_ignores=[],
    )
    scope = {
        "torch": torch,
        "get_pp_group": lambda: SimpleNamespace(
            is_first_rank=stage == 0, is_last_rank=stage == 3
        ),
        "make_input_embedding": lambda *a, **kw: torch.nn.Identity(),
        "has_full_vocab_on_rank": lambda _: False,
        "PPMissingLayer": torch.nn.Identity,
        "make_layers": lambda *a, **kw: (stage, stage, []),
        "RMSNorm": lambda *a, **kw: torch.nn.Identity(),
        "make_empty_intermediate_tensors_factory": lambda *a, **kw: None,
    }
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace(device_type="cpu")),
    )
    exec(compile(ast.fix_missing_locations(module), "DeepseekV32Model", "exec"), scope)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=4,
                index_topk=1,
                rms_norm_eps=1e-5,
            )
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=False,
            pipeline_parallel_size=4,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    with pytest.raises(ValueError, match="at least one decoder layer"):
        scope["DeepseekV32Model"](vllm_config=config)


def test_installer_is_idempotent(source_file):
    pipeline_hidden_states.patch_pipeline_hidden_states()
    once = source_file.read_bytes()
    pipeline_hidden_states.patch_pipeline_hidden_states()
    assert source_file.read_bytes() == once


@pytest.mark.parametrize("anchor", range(len(pipeline_hidden_states._SOURCE_EDITS)))
def test_installer_rejects_drift_without_partial_writes(source_file, anchor):
    old, _ = pipeline_hidden_states._SOURCE_EDITS[anchor]
    broken = source_file.read_text().replace(old, "# Source drift\n", 1)
    source_file.write_text(broken)
    with pytest.raises(RuntimeError, match="anchor mismatch"):
        pipeline_hidden_states.patch_pipeline_hidden_states()
    assert source_file.read_text() == broken


def test_installer_rejects_imported_unpatched_model(source_file, monkeypatch):
    before = source_file.read_bytes()
    monkeypatch.setitem(sys.modules, pipeline_hidden_states._MODULE, SimpleNamespace())
    with pytest.raises(RuntimeError, match="already imported"):
        pipeline_hidden_states.patch_pipeline_hidden_states()
    assert source_file.read_bytes() == before


def test_installer_rejects_unsupported_version(source_file, monkeypatch):
    before = source_file.read_bytes()
    monkeypatch.setattr(pipeline_hidden_states, "version", lambda _: "0.30.0")
    with pytest.raises(RuntimeError, match="requires vLLM 0.29.0"):
        pipeline_hidden_states.patch_pipeline_hidden_states()
    assert source_file.read_bytes() == before
