# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Carry pipeline routed-expert export on the pinned vLLM 0.29 runtime.

The runtime changes use only vLLM APIs and can be applied upstream unchanged.
Installation follows the other Python source patches in ``patches.py``. All
anchors are checked before writing, and the config guard is changed last.
"""

import sys
from contextlib import ExitStack
from importlib.metadata import version

_MODULE_NAMES = ("vllm.v1.worker.gpu_worker", "vllm.config.vllm")
_MARKER = "\n_NRL_PIPELINE_ROUTED_EXPERTS_PATCH = 1\n"

_SOURCE_EDITS = {
    "v1/worker/gpu_worker.py": (
        (
            "        parallel_config = self.vllm_config.parallel_config\n\n"
            "        if (\n"
            "            parallel_config.pipeline_parallel_size > 1\n",
            "        parallel_config = self.vllm_config.parallel_config\n"
            "        return_pipeline_routes = (\n"
            "            parallel_config.pipeline_parallel_size > 1\n"
            "            and self.model_config.enable_return_routed_experts\n"
            "        )\n\n"
            "        if (\n"
            "            parallel_config.pipeline_parallel_size > 1\n",
        ),
        (
            "            assert tensor_dict is not None\n"
            "            intermediate_tensors = AsyncIntermediateTensors(\n",
            "            assert tensor_dict is not None\n"
            "            if return_pipeline_routes:\n"
            "                capturer = self.model_runner.routed_experts_capturer\n"
            "                assert capturer is not None\n\n"
            "                def restore_routed_experts() -> None:\n"
            "                    # Run after the receive and TP reconstruction,\n"
            "                    # before the model overwrites this stage's layers.\n"
            '                    routes = tensor_dict.pop("routed_experts", None)\n'
            "                    destination = capturer.get_device_buffer()[\n"
            "                        :num_scheduled_tokens\n"
            "                    ]\n"
            "                    if (\n"
            "                        routes is None\n"
            "                        or routes.shape != destination.shape\n"
            "                        or routes.dtype != destination.dtype\n"
            "                        or routes.device != destination.device\n"
            "                    ):\n"
            "                        raise RuntimeError(\n"
            '                            "Missing or incompatible pipeline routed-experts "\n'
            '                            "tensor. All stages must enable route capture."\n'
            "                        )\n"
            "                    # Preserve the allocation referenced by CUDA graphs.\n"
            "                    destination.copy_(routes)\n\n"
            "                comm_postprocess.append(restore_routed_experts)\n"
            "            intermediate_tensors = AsyncIntermediateTensors(\n",
        ),
        (
            "        # Non-blocking send of the intermediate tensors. The metadata handle\n",
            "        if return_pipeline_routes:\n"
            "            capturer = self.model_runner.routed_experts_capturer\n"
            "            assert capturer is not None\n"
            "            # Keep the full layer axis: each stage fills its own layers.\n"
            "            # _pp_send_work protects this view until the next forward.\n"
            '            output.tensors["routed_experts"] = (\n'
            "                capturer.get_device_buffer()[:num_scheduled_tokens]\n"
            "            )\n\n"
            "        # Non-blocking send of the intermediate tensors. The metadata handle\n",
        ),
    ),
    "config/vllm.py": (
        (
            "            if self.parallel_config.pipeline_parallel_size > 1:\n"
            "                raise ValueError(\n"
            '                    "--enable-return-routed-experts is incompatible with "\n'
            '                    "pipeline parallelism (PP > 1)."\n'
            "                )\n",
            "            if self.parallel_config.pipeline_parallel_size > 1:\n"
            "                if (\n"
            "                    self.parallel_config.distributed_executor_backend\n"
            '                    == "external_launcher"\n'
            "                ):\n"
            "                    raise ValueError(\n"
            '                        "Pipeline routed-experts capture is not supported "\n'
            '                        "with the external_launcher executor."\n'
            "                    )\n"
            '                if self.model_config.runner_type != "generate":\n'
            "                    raise ValueError(\n"
            '                        "Pipeline routed-experts capture requires the "\n'
            '                        "generate runner."\n'
            "                    )\n"
            "                if self.speculative_config is not None:\n"
            "                    raise ValueError(\n"
            '                        "Pipeline routed-experts capture does not support "\n'
            '                        "speculative decoding."\n'
            "                    )\n"
            "                if self.parallel_config.enable_dbo:\n"
            "                    raise ValueError(\n"
            '                        "Pipeline routed-experts capture does not support "\n'
            '                        "dual batch overlap."\n'
            "                    )\n",
        ),
    ),
}


def patch_pipeline_routed_experts() -> None:
    """Install both halves before importing config or GPUWorker; fail on drift."""
    # Break the cycle: patches.py exposes this installer with the other patches.
    from nemo_rl.models.generation.vllm.patches import (
        _get_vllm_file,
        _locked_file_patch,
    )

    installed_version = version("vllm")
    if installed_version.split("+", 1)[0] != "0.29.0":
        raise RuntimeError(
            "Pipeline routed-experts compatibility patch requires vLLM 0.29.0; "
            f"found {installed_version}. Check upstream support before updating it."
        )
    for name in _MODULE_NAMES:
        module = sys.modules.get(name)
        if (
            module is not None
            and getattr(module, "_NRL_PIPELINE_ROUTED_EXPERTS_PATCH", None) != 1
        ):
            raise RuntimeError(
                f"{name} was already imported without pipeline routed-experts "
                "support. Install the compatibility patch before importing vLLM."
            )

    with ExitStack() as stack:
        updates = []
        for relative, edits in _SOURCE_EDITS.items():
            source, write_back = stack.enter_context(
                _locked_file_patch(_get_vllm_file(relative))
            )
            updated = source
            for old, new in edits:
                if updated.count(new) == 1:
                    continue
                if updated.count(old) != 1:
                    raise RuntimeError(
                        "Pipeline routed-experts patch anchor mismatch in "
                        f"{relative}; no source files have been changed."
                    )
                updated = updated.replace(old, new, 1)
            if _MARKER not in updated:
                updated += _MARKER
            compile(updated, relative, "exec")
            updates.append((write_back, source, updated))
        # Validate every source first. The config guard is lifted only after the
        # worker transport is installed, including on a shared installation.
        for write_back, source, updated in updates:
            if updated != source:
                write_back(updated)
