#!/usr/bin/env python3
"""Bounded exact-image GLM-5.1 vLLM async-engine scale probe on 32 GH200s."""

import asyncio
import gc
import os
import re
import time

import ray
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration


MODEL_PATH = os.environ["GLM_CKPT"]
NUM_NODES = int(os.environ.get("NUM_NODES", "8"))
GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", "4"))
WORLD_SIZE = NUM_NODES * GPUS_PER_NODE


def make_batch(
    tokenizer, prompt_text: str, *, enable_thinking: bool = True
) -> BatchedDataDict:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    encoded = tokenizer(
        [prompt], padding=True, return_tensors="pt", padding_side="right"
    )
    return BatchedDataDict(
        {
            "input_ids": encoded["input_ids"],
            "input_lengths": encoded["attention_mask"].sum(dim=1).to(torch.int32),
        }
    )


async def generate_one(policy: VllmGeneration, batch: BatchedDataDict, *, greedy: bool):
    async for _, output in policy.generate_async(batch, greedy=greedy):
        return output
    raise RuntimeError("vLLM returned no generation result")


async def run_generation(
    policy: VllmGeneration,
    batches: list[BatchedDataDict],
    *,
    greedy: bool,
):
    return await asyncio.gather(
        *(generate_one(policy, batch, greedy=greedy) for batch in batches)
    )


def main() -> None:
    expected_source = os.environ["EXPECTED_SOURCE_HEAD"]
    actual_source = os.environ.get("NEMO_RL_COMMIT")
    assert actual_source == expected_source, (actual_source, expected_source)
    init_ray()
    alive_nodes = [node for node in ray.nodes() if node["Alive"]]
    ray_gpus = int(ray.cluster_resources()["GPU"])
    assert len(alive_nodes) == NUM_NODES, alive_nodes
    assert ray_gpus == WORLD_SIZE, ray.cluster_resources()
    print(f"ray_nodes={len(alive_nodes)}", flush=True)
    print(f"ray_gpus={ray_gpus}", flush=True)
    print(f"model_path={MODEL_PATH}", flush=True)
    print(f"image_source_head={actual_source}", flush=True)

    tokenizer = get_tokenizer({"name": MODEL_PATH})
    config: VllmConfig = {
        "backend": "vllm",
        "model_name": MODEL_PATH,
        "tokenizer": {"name": MODEL_PATH},
        "dtype": "bfloat16",
        "max_new_tokens": 64,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 0.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "precision": "bfloat16",
            "tensor_parallel_size": WORLD_SIZE,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": WORLD_SIZE,
            "gpu_memory_utilization": 0.75,
            "max_model_len": 512,
            "async_engine": True,
            "skip_tokenizer_init": False,
            "load_format": "auto",
            "enforce_eager": True,
            "kv_cache_dtype": "auto",
            "sleep_level": 2,
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 0.5,
            "env_vars": {"PYTHONPATH": ""},
        },
        "colocated": {
            "enabled": False,
            "resources": {
                "gpus_per_node": GPUS_PER_NODE,
                "num_nodes": NUM_NODES,
            },
        },
        "vllm_kwargs": {},
    }
    config = configure_generation_config(config, tokenizer, is_eval=True)
    assert config["vllm_cfg"]["load_format"] == "auto"
    print(f"resolved_load_format={config['vllm_cfg']['load_format']}", flush=True)
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[GPUS_PER_NODE] * NUM_NODES,
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=GPUS_PER_NODE,
        name="glm5p1-vllm-tp32-async-preflight",
    )
    policy = None
    try:
        print("model_load_begin=1", flush=True)
        load_started = time.perf_counter()
        policy = VllmGeneration(cluster, config)
        load_seconds = time.perf_counter() - load_started
        print(f"model_load_seconds={load_seconds:.3f}", flush=True)

        prompts = [
            f"Solve carefully and give only the final integer: What is {137 + index} + {286 + 2 * index}?"
            for index in range(32)
        ]
        correctness_batch = make_batch(
            tokenizer,
            "What is 137 + 286? Respond with only the final integer.",
            enable_thinking=False,
        )
        warmup_started = time.perf_counter()
        warmup_outputs = asyncio.run(
            run_generation(policy, [correctness_batch], greedy=True)
        )
        warmup_seconds = time.perf_counter() - warmup_started
        warmup_tokens = int(warmup_outputs[0]["generation_lengths"].sum())
        warmup_token_ids = warmup_outputs[0]["output_ids"][0][-warmup_tokens:].tolist()
        warmup_text = tokenizer.decode(warmup_token_ids, skip_special_tokens=True)
        print(f"warmup_tokens={warmup_tokens}", flush=True)
        print(f"warmup_seconds={warmup_seconds:.3f}", flush=True)
        print(f"correctness_generation={warmup_text[:300]!r}", flush=True)
        assert re.search(r"(^|\D)423(\D|$)", warmup_text), warmup_text
        print("semantic_arithmetic_check=OK", flush=True)

        policy.clear_logger_metrics()
        batches = [make_batch(tokenizer, prompt) for prompt in prompts]
        input_tokens = sum(int(batch["input_lengths"].sum()) for batch in batches)
        generation_started = time.perf_counter()
        outputs = asyncio.run(run_generation(policy, batches, greedy=False))
        generation_seconds = time.perf_counter() - generation_started
        output_tokens = sum(
            int(output["generation_lengths"].sum()) for output in outputs
        )
        assert output_tokens > 0, outputs

        first_length = int(outputs[0]["generation_lengths"][0])
        first_tokens = outputs[0]["output_ids"][0][-first_length:].tolist()
        first_text = tokenizer.decode(first_tokens, skip_special_tokens=True)
        print(f"measured_requests={len(outputs)}", flush=True)
        print(f"input_tokens={input_tokens}", flush=True)
        print(f"output_tokens={output_tokens}", flush=True)
        print(f"generation_seconds={generation_seconds:.3f}", flush=True)
        print(
            f"input_tokens_per_second={input_tokens / generation_seconds:.3f}",
            flush=True,
        )
        print(
            f"output_tokens_per_second={output_tokens / generation_seconds:.3f}",
            flush=True,
        )
        print(f"first_generation={first_text[:300]!r}", flush=True)
        print(f"vllm_logger_metrics={policy.get_logger_metrics()}", flush=True)
        print("glm5p1_vllm_tp32_async=OK", flush=True)
    finally:
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()


if __name__ == "__main__":
    main()
