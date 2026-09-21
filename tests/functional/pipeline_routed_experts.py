# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare exported PP routes with actual per-stage MoE captures.

Run inside a Ray cluster using the certified worker environments. This is a
correctness test: observation synchronizes CUDA and is not a throughput test.
"""

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.venvs import image_venv_python, image_venvs_enabled


def verify_routes(
    observations: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> dict[str, int]:
    """Match exported IDs to actual local captures with unambiguous token keys.

    Prompts deliberately use distinct token IDs and separated generation
    positions. Repeated prefixes can reuse observations from earlier requests.
    Use the newest observation when a repeated request recomputes a token.
    Validate each round before adding the next; TP ranks must agree exactly.
    """
    expected: dict[tuple[int, int, int], list[int]] = {}
    all_layers = set()
    counts = []
    for worker in observations:
        all_layers.update(worker["layers"])
        worker_routes = {}
        for step in worker["records"]:
            counts.append(len(step["token_ids"]))
            for token, position, routes in zip(
                step["token_ids"], step["positions"], step["routes"], strict=True
            ):
                for layer, ids in zip(worker["layers"], routes, strict=True):
                    key = (token, position, layer)
                    worker_routes[key] = ids
        for key, ids in worker_routes.items():
            previous = expected.setdefault(key, ids)
            assert previous == ids, f"TP ranks disagree at {key}: {previous} != {ids}"
    assert counts and max(counts) > 1 and min(counts) < max(counts)
    compared = 0
    for output in outputs:
        tokens = output["tokens"]
        routes = output["routes"]
        assert len(routes) == len(tokens)
        for position, token in enumerate(tokens[:-1]):
            for layer in sorted(all_layers):
                key = (token, position, layer)
                assert key in expected, f"No local capture for {key}"
                assert routes[position][layer] == expected[key], (
                    f"Exported route differs at {key}: "
                    f"{routes[position][layer]} != {expected[key]}"
                )
                compared += len(expected[key])
    return {
        "compared_expert_ids": compared,
        "observed_layers": len(all_layers),
        "observed_workers": len(observations),
        "min_scheduled_tokens": min(counts),
        "max_scheduled_tokens": max(counts),
    }


async def generate_batch(
    generation: VllmGeneration, prompts: list[list[int]]
) -> list[dict[str, Any]]:
    async def generate_one(prompt: list[int]) -> dict[str, Any]:
        # NeMo's async API accepts one request. Concurrent calls let vLLM
        # schedule the mixed batch through its ordinary continuous batcher.
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor([prompt]),
                "input_lengths": torch.tensor([len(prompt)]),
            }
        )
        outputs = []
        async for _, batch in generation.generate_async(data, greedy=True):
            length = int(batch["unpadded_sequence_lengths"][0])
            outputs.append(
                {
                    "tokens": batch["output_ids"][0, :length].tolist(),
                    "routes": batch["routed_experts"][0, :length].tolist(),
                    "logprobs": batch["logprobs"][0, :length].tolist(),
                }
            )
        assert len(outputs) == 1
        return outputs[0]

    return await asyncio.gather(*(generate_one(prompt) for prompt in prompts))


def verify_prefix_cache(
    observations: list[dict[str, Any]], prefix: list[int]
) -> dict[str, int]:
    """Prove the repeated full blocks were reused, not recomputed successfully."""
    for worker in observations:
        counts: Counter[tuple[int, int]] = Counter()
        for step in worker["records"]:
            counts.update(zip(step["token_ids"], step["positions"], strict=True))
        for position, token in enumerate(prefix):
            assert counts[(token, position)] == 1, (
                f"Prefix token {position} was executed {counts[(token, position)]} "
                f"times on PP{worker['pp_rank']}/TP{worker['tp_rank']}"
            )
    return {"cached_tokens_per_request": len(prefix), "reuse_requests": 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--runner-version", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "config": vars(args),
        "outputs": [],
        "checks": [],
        "passed": False,
    }
    register_omegaconf_resolvers()
    config = load_config("examples/configs/grpo_math_1B_megatron.yaml")
    gen = config.policy.generation
    gen.backend = "vllm"
    gen.model_name = args.model
    gen.colocated.enabled = False
    gen.max_new_tokens = 8
    gen.vllm_cfg.async_engine = True
    gen.vllm_cfg.tensor_parallel_size = args.tp
    gen.vllm_cfg.pipeline_parallel_size = args.pp
    gen.vllm_cfg.expert_parallel_size = args.tp
    gen.vllm_cfg.enforce_eager = not args.graphs
    gen.vllm_cfg.gpu_memory_utilization = 0.6
    gen.vllm_cfg.max_model_len = 160
    gen.vllm_cfg.enable_prefix_caching = True
    gen.vllm_cfg.env_vars = {"VLLM_USE_V2_MODEL_RUNNER": str(args.runner_version - 1)}
    gen.vllm_kwargs = {
        "enable_return_routed_experts": True,
        "enable_chunked_prefill": True,
        "block_size": 16,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 32,
        "moe_backend": "triton",
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_config = AutoConfig.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_config = configure_generation_config(
        OmegaConf.to_container(gen, resolve=True), tokenizer
    )
    gen_config["vllm_cfg"]["load_format"] = "auto"
    base_worker = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
    actor_env = get_actor_python_env(base_worker)
    if image_venvs_enabled():
        actor_env = image_venv_python(actor_env, base_worker)
    fqn = "tests.functional.pipeline_routed_experts_workers.PipelineRoutesWorker"
    ACTOR_ENVIRONMENT_REGISTRY[fqn] = actor_env
    gen_config["worker_extension_cls_fqn"] = fqn
    init_ray(
        log_dir=None
        if os.environ.get("RAY_ADDRESS")
        else f"/tmp/pp-routes-{os.getpid()}"
    )
    cluster = RayVirtualCluster(
        [args.tp * args.pp], num_gpus_per_node=4, name="pipeline_routes"
    )
    generation = VllmGeneration(cluster, gen_config)
    try:
        owners = generation._refit_leader_workers()
        report["workers"] = [
            item
            for engine in ray.get([w.install_route_observer.remote() for w in owners])
            for item in engine
        ]
        assert len(report["workers"]) == args.tp * args.pp
        assert all(
            w["runner_v2"] == (args.runner_version == 2) for w in report["workers"]
        )
        assert {w["pp_rank"] for w in report["workers"]} == set(range(args.pp))
        owned_layers = {
            layer for worker in report["workers"] for layer in worker["layers"]
        }
        assert owned_layers == set(
            range(
                getattr(hf_config, "first_k_dense_replace", 0),
                hf_config.num_hidden_layers,
            )
        )
        if args.tp * args.pp > 4:
            assert len({w["hostname"] for w in report["workers"]}) > 1
        generation.prepare_for_generation()
        prefix = list(range(1000, 1097))
        rounds = [
            [prefix],
            [prefix, list(range(2000, 2033)), list(range(3000, 3067))],
            [prefix[:96] + list(range(4000, 4025))],
        ]
        for prompts in rounds:
            outputs = asyncio.run(generate_batch(generation, prompts))
            report["outputs"].extend(outputs)
            nested = ray.get([w.read_route_observations.remote() for w in owners])
            observations = [worker for engine in nested for worker in engine]
            out.with_suffix(".observations.json").write_text(
                json.dumps(observations) + "\n"
            )
            out.write_text(json.dumps(report, indent=2) + "\n")
            report["checks"].append(verify_routes(observations, outputs))
            out.write_text(json.dumps(report, indent=2) + "\n")
        report["prefix_cache"] = verify_prefix_cache(observations, prefix[:96])
        generation.finish_generation()
        report["passed"] = True
        out.write_text(json.dumps(report, indent=2) + "\n")
        print("PIPELINE_ROUTES_RESULT " + json.dumps(report["checks"]), flush=True)
    finally:
        generation.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
