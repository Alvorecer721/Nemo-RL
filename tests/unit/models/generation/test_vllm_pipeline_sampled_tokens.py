# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute upstream PP feedback with distinct producer and consumer streams."""

import ast
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nemo_rl.models.generation.vllm import patches, pipeline_sampled_tokens as pipeline


@pytest.fixture
def source_file(tmp_path, monkeypatch):
    source = Path(patches._get_vllm_file(pipeline._SOURCE)).read_text()
    for old, new in pipeline._SOURCE_EDITS:
        source = source.replace(new, old)
    source = source.replace(pipeline._MARKER, "")
    path = tmp_path / "pp_utils.py"
    path.write_text(source)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(path))
    monkeypatch.delitem(sys.modules, pipeline._MODULE, raising=False)
    return path


class _Stream:
    def __init__(self):
        self.waited_events = []
        self.waited_streams = []

    def wait_event(self, event):
        self.waited_events.append(event)

    def wait_stream(self, stream):
        self.waited_streams.append(stream)


class _Tensor:
    dtype = "int64"

    def __init__(self):
        self.recorded = []

    def record_stream(self, stream):
        self.recorded.append(stream)

    def contiguous(self):
        return self


def _load_handler(source):
    tree = ast.parse(source)
    handler = next(
        x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "PPHandler"
    )
    current = [_Stream()]

    @contextmanager
    def use_stream(stream):
        prior = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = prior

    cuda = SimpleNamespace(current_stream=lambda *_: current[0], stream=use_stream)
    torch = SimpleNamespace(
        cuda=cuda,
        int64="int64",
        stack=lambda *_, **__: _Tensor(),
        distributed=SimpleNamespace(broadcast=lambda *_, **__: None),
    )
    namespace = {
        "torch": torch,
        "np": np,
        "current_platform": SimpleNamespace(is_xpu=lambda: False),
        "compute_need_sampled_mask": lambda _: np.array([True]),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom("__future__", [ast.alias("annotations")], 0),
            handler,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "pp_utils.py", "exec"), namespace)
    obj = object.__new__(namespace["PPHandler"])
    obj.device = "cuda"
    # The constructor's stream differs from the caller's execution stream.
    obj.main_stream = _Stream()
    obj.broadcast_stream = _Stream()
    obj.last_rank = 3
    obj.broadcast_group = object()
    return obj, current


def test_received_buffers_wait_on_and_belong_to_actual_consumer(source_file):
    pipeline.patch_pipeline_sampled_tokens()
    handler, current = _load_handler(source_file.read_text())
    slot = SimpleNamespace(
        event=object(),
        sampled_tokens=_Tensor(),
        num_sampled=_Tensor(),
        num_rejected=_Tensor(),
        idx_mapping=_Tensor(),
        idx_mapping_np=np.array([0]),
        need_sampled_mask=np.array([True]),
        gen_at_receive_np=np.array([2]),
    )
    handler.req_idx_gen_np = np.array([2])
    handler.queue = deque([slot, None, None, None])
    outputs = handler.get_prev_sampled_outputs()
    assert current[0].waited_events == [slot.event]
    assert handler.main_stream.waited_events == []
    assert all(tensor.recorded == [current[0]] for tensor in outputs.values())
    assert list(handler.queue) == [None] * 4


def test_broadcast_waits_for_actual_producer(source_file):
    pipeline.patch_pipeline_sampled_tokens()
    handler, current = _load_handler(source_file.read_text())
    handler.is_last_rank = True
    tensors = [_Tensor(), _Tensor(), _Tensor()]
    handler.broadcast(*tensors, SimpleNamespace())
    assert handler.broadcast_stream.waited_streams == [current[0]]
    assert all(tensor.recorded == [handler.broadcast_stream] for tensor in tensors)


def test_freed_request_feedback_is_still_discarded(source_file):
    pipeline.patch_pipeline_sampled_tokens()
    handler, current = _load_handler(source_file.read_text())
    handler.req_idx_gen_np = np.array([3])
    handler.queue = deque(
        [
            SimpleNamespace(
                idx_mapping_np=np.array([0]),
                gen_at_receive_np=np.array([2]),
                need_sampled_mask=np.array([True]),
                idx_mapping=_Tensor(),
            )
        ]
    )
    assert handler.get_prev_sampled_outputs() is None
    assert current[0].waited_events == []


def test_installation_is_idempotent_and_rejects_source_drift(source_file):
    pipeline.patch_pipeline_sampled_tokens()
    once = source_file.read_bytes()
    pipeline.patch_pipeline_sampled_tokens()
    assert source_file.read_bytes() == once
    source_file.write_text("class PPHandler: pass\n")
    with pytest.raises(RuntimeError, match="anchor mismatch"):
        pipeline.patch_pipeline_sampled_tokens()
    assert source_file.read_text() == "class PPHandler: pass\n"


def test_installation_rejects_already_imported_unpatched_module(
    source_file, monkeypatch
):
    before = source_file.read_bytes()
    monkeypatch.setitem(sys.modules, pipeline._MODULE, SimpleNamespace())
    with pytest.raises(RuntimeError, match="already imported"):
        pipeline.patch_pipeline_sampled_tokens()
    assert source_file.read_bytes() == before
