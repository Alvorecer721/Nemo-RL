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
"""Make vLLM's XIELU activation torch.compile-safe when the CUDA kernel is present.

vLLM's ``XIELU.forward_native`` branches on ``torch._dynamo.is_compiling()`` and
calls ``logger.warning_once`` in the compiled branch. With the kernel importable
that is fatal on every fresh trace: a graph break under fullgraph/AOT capture,
and the raw torchbind handle cannot be serialized into the compile cache
(``Tried to serialize object __torch__.torch.classes.xielu.XIELU``). Worse when
it does not crash — a cache written by a kernel-less run silently inlines the
Python fallback, and the cache key does not record kernel presence.

The fix wraps the kernel in a torch custom op (an opaque, cacheable graph node)
and replaces the activation's forward with a branch-free version calling it, so
the kernel runs *inside* compiled graphs.

This is installed as a ``vllm.general_plugins`` entry point rather than called
from :mod:`nemo_rl.models.generation.vllm.patches`: vLLM builds the model in a
separate EngineCore process, and ``load_general_plugins()`` is invoked in each
of its processes (``v1/engine/core.py``, ``v1/worker/worker_base.py``), whereas
an in-memory patch applied in the Ray actor would never reach them.

Upstream fix pending: if vLLM registers the kernel as a custom op itself, this
module can be deleted outright.
"""

import logging

logger = logging.getLogger(__name__)

_OP_NAMESPACE = "xielu_shim"
_OP_NAME = "fused_forward"

_op_registered = False
_kernel_handle = None


def _kernel():
    """The torchbind handle, constructed once so no work happens per call."""
    global _kernel_handle
    import torch

    if _kernel_handle is None:
        _kernel_handle = torch.classes.xielu.XIELU()
    return _kernel_handle


def _register_custom_op() -> None:
    """Register the kernel as an opaque, cacheable graph node (once per process).

    Dynamo cannot trace into a torchbind method and cannot serialize the handle
    into vLLM's compile cache; as a custom op it becomes a single node referenced
    by name. The fake implementation is what lets shape propagation run without
    executing CUDA, which fullgraph capture requires.
    """
    global _op_registered
    if _op_registered:
        return

    import torch

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::{_OP_NAME}", mutates_args=(), device_types="cuda"
    )
    def fused_forward(
        x: torch.Tensor,
        alpha_p: torch.Tensor,
        alpha_n: torch.Tensor,
        beta: float,
        eps: float,
        with_vector_loads: bool,
    ) -> torch.Tensor:
        original_shape = x.shape
        while x.dim() < 3:
            x = x.unsqueeze(0)
        if x.dim() > 3:
            x = x.reshape(-1, 1, x.size(-1))
        out = _kernel().forward(
            x.contiguous(), alpha_p, alpha_n, beta, eps, with_vector_loads
        )
        return out.reshape(original_shape)

    @fused_forward.register_fake
    def _(x, alpha_p, alpha_n, beta, eps, with_vector_loads):
        return torch.empty_like(x)

    _op_registered = True


def apply() -> None:
    """Entry point for ``vllm.general_plugins``; safe to call in any process.

    No-op unless both vLLM's XIELU layer and the CUDA kernel are importable, so
    non-Apertus models and kernel-less environments are unaffected.
    """
    try:
        import xielu.ops  # noqa: F401  (registers torch.classes.xielu)
    except ImportError:
        return

    try:
        from vllm.model_executor.layers import activation
    except ImportError:
        return

    cls = getattr(activation, "XIELU", None)
    if cls is None or getattr(cls, "_nemo_rl_compile_safe", False):
        return

    import torch

    _register_custom_op()

    def forward_native(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_cuda:
            return torch.ops.xielu_shim.fused_forward(
                input,
                self.alpha_p,
                self.alpha_n,
                self._beta_scalar,
                self._eps_scalar,
                self.with_vector_loads,
            )
        return self._xielu_python(input)

    def forward(self, *args, **kwargs):
        return forward_native(self, *args, **kwargs)

    cls.forward_native = forward_native
    cls.forward_cuda = forward_native
    # CustomOp.__init__ snapshots self._forward_method, and for the first
    # instance that happens before this plugin can run; overriding forward()
    # makes the lookup happen at call time for every instance.
    cls.forward = forward
    cls._nemo_rl_compile_safe = True
    # warning level so the marker survives default logging config in every
    # engine process — probes grep for it as proof the plugin applied.
    logger.warning("Patched vLLM XIELU for torch.compile-safe CUDA kernel dispatch.")
