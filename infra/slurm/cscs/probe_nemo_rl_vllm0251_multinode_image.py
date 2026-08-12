#!/usr/bin/env python3
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

"""Start one TP8 vLLM engine across two four-GPU GH200 nodes."""

import gc
import os

import ray
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration


def main() -> None:
    model_name = os.environ["APERTUS_CKPT"]
    tokenizer_name = os.environ["APERTUS_TOKENIZER"]
    init_ray()

    alive_nodes = [node for node in ray.nodes() if node["Alive"]]
    assert len(alive_nodes) == 2, alive_nodes
    assert int(ray.cluster_resources()["GPU"]) == 8, ray.cluster_resources()
    print(f"ray_nodes={len(alive_nodes)}")
    print(f"ray_gpus={int(ray.cluster_resources()['GPU'])}")

    tokenizer = get_tokenizer({"name": tokenizer_name})
    config: VllmConfig = {
        "backend": "vllm",
        "model_name": model_name,
        "tokenizer": {"name": tokenizer_name},
        "dtype": "bfloat16",
        "max_new_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "precision": "bfloat16",
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.35,
            "max_model_len": 512,
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": "auto",
            "enforce_eager": True,
            "kv_cache_dtype": "auto",
            "env_vars": {"PYTHONPATH": ""},
        },
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "vllm_kwargs": {"distributed_executor_backend": "ray"},
    }
    config = configure_generation_config(config, tokenizer)
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[4, 4],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=4,
        name="vllm0251-two-node-image-probe",
    )
    policy = None
    try:
        policy = VllmGeneration(cluster, config)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Name the capital of Switzerland."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            [prompt],
            padding=True,
            return_tensors="pt",
            padding_side="right",
        )
        batch = BatchedDataDict(
            {
                "input_ids": encoded["input_ids"],
                "input_lengths": encoded["attention_mask"].sum(dim=1).to(torch.int32),
            }
        )
        output = policy.generate(batch)
        length = int(output["generation_lengths"][0])
        assert length > 0, output
        print(f"generated_tokens={length}")
        print("multinode_vllm_startup=OK")
    finally:
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()


if __name__ == "__main__":
    main()
