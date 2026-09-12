# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run under a baked vLLM worker Python, before expensive engine startup."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("vllm"), "requires a vLLM worker environment"
)
class WorkerExtensionAPITests(unittest.TestCase):
    def test_native_worker_can_accept_each_extension(self):
        from nemo_rl.models.generation.vllm.vllm_backend import (
            VllmInternalWorkerExtension,
            VllmInternalWorkerExtensionWithCheckpointEngine,
            VllmWorker,
        )

        # vLLM WorkerWrapperBase rejects every overlapping non-dunder attribute
        # before dynamically adding the extension as a base class.
        for extension in (
            VllmInternalWorkerExtension,
            VllmInternalWorkerExtensionWithCheckpointEngine,
        ):
            with self.subTest(extension=extension.__name__):
                overlaps = [
                    name
                    for name in dir(extension)
                    if not name.startswith("__") and hasattr(VllmWorker, name)
                ]
                self.assertEqual(overlaps, [])


if __name__ == "__main__":
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError("Run this qualification under the baked vLLM worker Python")
    unittest.main()
