# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute patched vLLM control flow with CPU tensors and simulated PP I/O."""

import ast
import copy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import NoneType, SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.vllm import patches

_WORKER = "v1/worker/gpu_worker.py"
_CONFIG = "config/vllm.py"


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    install = getattr(patches, "patch_pipeline_routed_experts", None)
    edits_by_file = {_WORKER: (), _CONFIG: ()}
    marker = ""
    if install is not None:
        from nemo_rl.models.generation.vllm import pipeline_routed_experts as pipeline

        edits_by_file = pipeline._SOURCE_EDITS
        marker = pipeline._MARKER
    for relative, edits in edits_by_file.items():
        content = Path(patches._get_vllm_file(relative)).read_text()
        for old, new in edits:
            content = content.replace(new, old)
        if marker:
            content = content.replace(marker, "")
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda p: str(tmp_path / p))
    # Other tests may have imported the real package. Only the temporary copies
    # are patched here; the import-safety contract gets its own negative test.
    for name in ("vllm.config.vllm", "vllm.v1.worker.gpu_worker"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return tmp_path


def _load_worker(source):
    tree = ast.parse(source)
    worker = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Worker"
    )
    execute = copy.deepcopy(
        next(
            n
            for n in worker.body
            if isinstance(n, ast.FunctionDef) and n.name == "execute_model"
        )
    )
    execute.decorator_list = []
    async_intermediate = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AsyncIntermediateTensors"
    )

    class IntermediateTensors:
        def __init__(self, tensors):
            self.tensors = tensors

    class ModelRunnerOutput:
        def __init__(self, routes):
            self.routes = routes

    namespace = dict(
        torch=torch,
        NoneType=NoneType,
        IntermediateTensors=IntermediateTensors,
        ModelRunnerOutput=ModelRunnerOutput,
        AsyncModelRunnerOutput=type("AsyncModelRunnerOutput", (), {}),
        get_tp_group=lambda: None,
    )
    # Postponed annotations keep CUDA-only upstream types out of CPU tests.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            async_intermediate,
            execute,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "patched_gpu_worker.py", "exec"),
        namespace,
    )
    return namespace


def _config(*, pp=3, enabled=True, **overrides):
    parallel = SimpleNamespace(
        pipeline_parallel_size=pp,
        distributed_executor_backend="ray",
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_dbo=False,
    )
    model = SimpleNamespace(
        enable_return_routed_experts=enabled, runner_type="generate"
    )
    config = SimpleNamespace(
        model_config=model,
        parallel_config=parallel,
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
        speculative_config=None,
        kv_transfer_config=None,
    )
    for name, value in overrides.items():
        target = (
            parallel
            if hasattr(parallel, name)
            else model
            if hasattr(model, name)
            else config
        )
        setattr(target, name, value)
    return config


def _config_check(source):
    tree = ast.parse(source)
    guard = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and any(
            isinstance(a, ast.Attribute) and a.attr == "enable_return_routed_experts"
            for a in ast.walk(n.test)
        )
    )
    function = ast.parse("def check(self):\n    pass\n").body[0]
    function.body = [guard]
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "patched_config.py",
            "exec",
        ),
        namespace,
    )
    return namespace["check"]


class _Handle:
    def __init__(self, callback):
        self.callback = callback

    def wait(self):
        self.callback()


@pytest.mark.parametrize("use_v2", [False, True])
def test_every_stage_exports_its_real_routes_before_buffer_reuse(source_tree, use_v2):
    """Missing copy/send, reordered postprocessing or missing wait corrupts IDs."""
    install = getattr(patches, "patch_pipeline_routed_experts", None)
    if install is not None:
        install()
    ns = _load_worker((source_tree / _WORKER).read_text())
    groups = []
    workers = []
    messages = {}
    send_waits = []
    for rank in range(3):
        buffer = torch.zeros(8, 7, 2, dtype=torch.int32)
        capturer = SimpleNamespace(get_device_buffer=lambda b=buffer: b)

        def execute(scheduler, intermediate, *, rank=rank, buffer=buffer):
            if intermediate is not None:
                # Real AsyncIntermediateTensors executes receive postprocessors
                # here, at the same boundary used by both model runners.
                assert set(intermediate.tensors) == {"hidden_states"}
            count = scheduler.total_num_scheduled_tokens
            for local, layer in enumerate((2 * rank + 1, 2 * rank + 2)):
                value = 10 * (rank + 1) + local + 1
                buffer[:count, layer] = (
                    torch.tensor([value, value + 100])
                    + torch.arange(count)[:, None] * 200
                    + scheduler.route_offset
                )
            if rank == 2:
                return ns["ModelRunnerOutput"](buffer[:count].clone())
            return ns["IntermediateTensors"]({"hidden_states": torch.zeros(count, 4)})

        def send(tensors, *, rank=rank, **kwargs):
            # Keep references (as the real asynchronous send does), and verify
            # the next invocation waits before model code can overwrite them.
            messages[rank + 1] = dict(tensors)
            frozen = {k: v.clone() for k, v in tensors.items()}

            def wait():
                for key, value in tensors.items():
                    torch.testing.assert_close(value, frozen[key])
                send_waits.append(rank)

            return [_Handle(lambda: None), _Handle(wait)]

        def receive(*, rank=rank, **kwargs):
            incoming = messages[rank]
            received = {k: torch.empty_like(v) for k, v in incoming.items()}
            complete = []

            def gather():
                assert complete == [True]
                for key, value in incoming.items():
                    received[key].copy_(value)

            return received, [_Handle(lambda: complete.append(True))], [gather]

        groups.append(
            SimpleNamespace(
                is_first_rank=rank == 0,
                is_last_rank=rank == 2,
                isend_tensor_dict=send,
                irecv_tensor_dict=receive,
            )
        )
        workers.append(
            SimpleNamespace(
                _pp_send_work=[],
                vllm_config=_config(),
                model_config=_config().model_config,
                use_v2_model_runner=use_v2,
                annotate_profile=lambda _: nullcontext(),
                model_runner=SimpleNamespace(
                    execute_model=execute,
                    routed_experts_capturer=capturer,
                    is_pooling_model=False,
                ),
            )
        )

    pointers = [
        w.model_runner.routed_experts_capturer.get_device_buffer().data_ptr()
        for w in workers
    ]
    expected = torch.tensor(
        [[0, 0], [11, 111], [12, 112], [21, 121], [22, 122], [31, 131], [32, 132]],
        dtype=torch.int32,
    )
    for step, count in enumerate((5, 1, 3)):
        for rank, worker in enumerate(workers):
            ns["get_pp_group"] = lambda rank=rank: groups[rank]
            result = ns["execute_model"](
                worker,
                SimpleNamespace(
                    total_num_scheduled_tokens=count, route_offset=step * 1000
                ),
            )
        wanted = expected.expand(count, -1, -1).clone()
        wanted[:, 1:] += torch.arange(count)[:, None, None] * 200 + step * 1000
        torch.testing.assert_close(result.routes, wanted)
    assert send_waits == [0, 1, 0, 1]
    assert pointers == [
        w.model_runner.routed_experts_capturer.get_device_buffer().data_ptr()
        for w in workers
    ]


@pytest.mark.parametrize(
    "changes,allowed",
    [
        ({}, True),
        ({"pp": 1, "distributed_executor_backend": "external_launcher"}, True),
        ({"enabled": False, "runner_type": "pooling"}, True),
        ({"distributed_executor_backend": "external_launcher"}, False),
        ({"runner_type": "pooling"}, False),
        ({"speculative_config": object()}, False),
        ({"enable_dbo": True}, False),
        ({"decode_context_parallel_size": 2}, False),
        ({"prefill_context_parallel_size": 2}, False),
        ({"kv_transfer_config": SimpleNamespace(is_kv_transfer_instance=True)}, False),
    ],
)
def test_unsupported_configs_remain_rejected(source_tree, changes, allowed):
    patches.patch_pipeline_routed_experts()
    check = _config_check((source_tree / _CONFIG).read_text())
    if allowed:
        check(_config(**changes))
    else:
        with pytest.raises(ValueError):
            check(_config(**changes))


def test_source_installation_is_idempotent(source_tree):
    patches.patch_pipeline_routed_experts()
    once = {p: p.read_bytes() for p in source_tree.rglob("*.py")}
    patches.patch_pipeline_routed_experts()
    assert once == {p: p.read_bytes() for p in source_tree.rglob("*.py")}


@pytest.mark.parametrize("relative", [_CONFIG, _WORKER])
def test_source_drift_does_not_partially_install(source_tree, relative):
    (source_tree / relative).write_text("# Unknown upstream implementation\n")
    original = {p: p.read_bytes() for p in source_tree.rglob("*.py")}
    with pytest.raises(RuntimeError, match="anchor"):
        patches.patch_pipeline_routed_experts()
    assert original == {p: p.read_bytes() for p in source_tree.rglob("*.py")}


@pytest.mark.parametrize("module", ["vllm.config.vllm", "vllm.v1.worker.gpu_worker"])
def test_imported_unpatched_modules_are_rejected(source_tree, monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    with pytest.raises(RuntimeError, match="already imported"):
        patches.patch_pipeline_routed_experts()


def test_dependency_upgrade_requires_compatibility_review(source_tree, monkeypatch):
    from nemo_rl.models.generation.vllm import pipeline_routed_experts as pipeline

    monkeypatch.setattr(pipeline, "version", lambda _: "0.30.0")
    original = {p: p.read_bytes() for p in source_tree.rglob("*.py")}
    with pytest.raises(RuntimeError, match="requires vLLM 0.29.0"):
        patches.patch_pipeline_routed_experts()
    assert original == {p: p.read_bytes() for p in source_tree.rglob("*.py")}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        torch.zeros(2, 7, 2, dtype=torch.int32),
        torch.zeros(3, 1, 2, dtype=torch.int32),
        torch.zeros(3, 7, 2),
    ],
)
def test_missing_or_malformed_routes_fail_before_forward(source_tree, payload):
    patches.patch_pipeline_routed_experts()
    ns = _load_worker((source_tree / _WORKER).read_text())
    incoming = {"hidden_states": torch.ones(3, 4)}
    if payload is not None:
        incoming["routed_experts"] = payload
    ns["get_pp_group"] = lambda: SimpleNamespace(
        is_first_rank=False,
        is_last_rank=True,
        irecv_tensor_dict=lambda **_: (incoming, [], []),
    )
    buffer = torch.zeros(8, 7, 2, dtype=torch.int32)

    def execute(scheduler, intermediate):
        intermediate.wait_for_comm()
        pytest.fail("Invalid routes reached the model forward")

    worker = SimpleNamespace(
        _pp_send_work=[],
        vllm_config=_config(),
        use_v2_model_runner=False,
        model_config=_config().model_config,
        annotate_profile=lambda _: nullcontext(),
        model_runner=SimpleNamespace(
            execute_model=execute,
            routed_experts_capturer=SimpleNamespace(get_device_buffer=lambda: buffer),
        ),
    )
    with pytest.raises(RuntimeError, match="[Rr]outed.experts|[Rr]oute"):
        ns["execute_model"](worker, SimpleNamespace(total_num_scheduled_tokens=3))
    assert torch.count_nonzero(buffer) == 0


@pytest.mark.parametrize(
    "pp,enabled,count", [(1, True, 3), (3, False, 3), (3, True, 0)]
)
def test_disabled_or_empty_steps_do_not_access_capture_storage(
    source_tree, pp, enabled, count
):
    patches.patch_pipeline_routed_experts()
    ns = _load_worker((source_tree / _WORKER).read_text())
    ns["get_pp_group"] = lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)
    worker = SimpleNamespace(
        _pp_send_work=[],
        vllm_config=_config(pp=pp, enabled=enabled),
        model_config=_config(enabled=enabled).model_config,
        use_v2_model_runner=False,
        annotate_profile=lambda _: nullcontext(),
        model_runner=SimpleNamespace(execute_model=lambda *_: None),
    )
    assert (
        ns["execute_model"](worker, SimpleNamespace(total_num_scheduled_tokens=count))
        is None
    )
