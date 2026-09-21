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

"""BF16 Qwen3, Apertus and GLM refits on Megatron and vLLM."""

import argparse
import asyncio
import json
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.loss.loss_functions import NLLLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.megatron.router_replay import configure_vllm_for_router_replay
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.venvs import image_venv_python, image_venvs_enabled
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.nccl_reshard_utils import check_nccl_reshard_refit_support
from tests.functional.nccl_reshard_pp_faults import (
    lose_stage_between_refits,
    lose_stage_during_refit,
)
from tests.functional.nccl_reshard_pp_reference import verify_snapshots


def verify_distributed_snapshots(
    snapshots: list[dict[str, Any]],
    model_path: str,
    updates: int,
    *,
    reference_workers: int,
    reference_nodes: int,
) -> dict[str, Any]:
    """Keep full-model checkpoint reads bounded to one checker per node."""
    if reference_nodes == 1:
        return verify_snapshots(
            snapshots, model_path, updates, reference_workers=reference_workers
        )
    nodes = sorted(
        node["NodeID"]
        for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("GPU", 0) > 0
    )
    if reference_nodes > len(nodes):
        raise ValueError(
            f"Requested {reference_nodes} reference nodes; found {len(nodes)}"
        )
    shared_snapshots = ray.put(snapshots)
    checker = ray.remote(verify_snapshots)
    parts = ray.get(
        [
            checker.options(
                num_cpus=reference_workers,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False),
            ).remote(
                shared_snapshots,
                model_path,
                updates,
                reference_workers=reference_workers,
                shard_index=index,
                shard_count=reference_nodes,
            )
            for index, node in enumerate(nodes[:reference_nodes])
        ]
    )
    return {key: sum(part[key] for part in parts) for key in parts[0]}


async def generate_one(
    generation: VllmGeneration,
    inputs: BatchedDataDict,
    *,
    dp_shard_idx: int | None = None,
) -> BatchedDataDict:
    stream = (
        generation.generate_async(inputs, greedy=True)
        if dp_shard_idx is None
        else generation._generate_on_shard(
            data=inputs,
            method_name="generate_async",
            greedy=True,
            dp_shard_idx=dp_shard_idx,
            leader_worker_idx=generation.worker_group.get_dp_leader_worker_idx(
                dp_shard_idx
            ),
        )
    )
    results = [result async for _, result in stream]
    assert len(results) == 1
    return results[0]


def training_batch(
    tokenizer: PreTrainedTokenizerBase, batch_size: int
) -> BatchedDataDict:
    texts = [
        "The capital of France is Paris. Paris is a city in Europe.",
        "Two plus three equals five. Five minus two equals three.",
        "A python function can return the sum of two input numbers.",
        "Water freezes at zero degrees Celsius under normal pressure.",
    ]
    encoded = tokenizer(
        [texts[i % len(texts)] for i in range(batch_size)],
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True,
    )
    return BatchedDataDict(
        {
            "input_ids": encoded["input_ids"],
            "input_lengths": encoded["attention_mask"].sum(dim=1).to(torch.int32),
            "attention_mask": encoded["attention_mask"],
            "token_mask": encoded["attention_mask"].float(),
            "sample_mask": torch.ones(batch_size),
        }
    )


def get_unreplayed_logprobs(
    policy: Policy, data: BatchedDataDict[Any]
) -> BatchedDataDict[Any]:
    """Use the normal policy sharding with the test-only no-replay worker RPC."""
    sharded, unsorted_indices = policy._shard_for_logprob(data)
    futures = policy.worker_group.run_all_workers_sharded_data(
        "get_unreplayed_logprobs",
        data=sharded,
        in_sharded_axes=["data_parallel"],
        replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
        output_is_replicated=[
            "context_parallel",
            "tensor_parallel",
            "pipeline_parallel",
        ],
    )
    result = BatchedDataDict.from_batches(
        policy.worker_group.get_all_worker_results(futures)
    )
    if unsorted_indices is not None:
        result.reorder_data(unsorted_indices)
    return result


def check_complete_routes(
    result: BatchedDataDict[Any], hf_config: dict[str, Any]
) -> dict[str, int]:
    """Require every executed token's MoE layers, excluding the final sampled token."""
    routes = result["routed_experts"]
    assert routes.shape[:2] == result["output_ids"].shape
    assert routes.shape[2:] == (
        hf_config["num_hidden_layers"],
        hf_config["num_experts_per_tok"],
    )
    first_layer = hf_config.get("first_k_dense_replace", 0)
    num_experts = hf_config.get("n_routed_experts", hf_config.get("num_experts"))
    assert num_experts is not None
    compared = 0
    for row, length in enumerate(result["unpadded_sequence_lengths"]):
        executed = routes[row, : int(length) - 1, first_layer:]
        assert executed.numel() > 0
        # The exclusive bound may not fit the compact wire dtype: comparing
        # int8 routes directly with 128 wraps the scalar to -128 in PyTorch.
        min_id, max_id = int(executed.min()), int(executed.max())
        assert 0 <= min_id and max_id < num_experts, (
            "Missing or invalid route for an executed token's MoE layer: "
            f"min={min_id}, max={max_id}, num_experts={num_experts}"
        )
        sorted_ids = executed.sort(dim=-1).values
        assert torch.all(sorted_ids[..., 1:] != sorted_ids[..., :-1]), (
            "Duplicate expert IDs in a top-k route"
        )
        compared += executed.numel()
    return {"complete_expert_ids": compared, "missing_expert_ids": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "megatron"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config", default="examples/configs/grpo_math_1B_megatron.yaml"
    )
    parser.add_argument("--train-tp", type=int, default=2)
    parser.add_argument("--train-pp", type=int, default=1)
    parser.add_argument("--train-dp", type=int, default=1)
    parser.add_argument("--train-ep", type=int, default=1)
    parser.add_argument("--train-etp", type=int)
    parser.add_argument("--gen-tp", type=int, default=1)
    parser.add_argument("--gen-pp", type=int, default=2)
    parser.add_argument("--gen-dp", type=int, default=1)
    parser.add_argument("--gen-ep", type=int, default=1)
    parser.add_argument("--gen-etp", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--warm-refits", type=int, default=5)
    parser.add_argument(
        "--router-replay",
        action="store_true",
        help="Compare Megatron with and without replay of the same PP rollout routes",
    )
    parser.add_argument(
        "--update-mode",
        choices=["scale", "optimizer"],
        default="scale",
        help="Change weights by deterministic BF16 scaling or real Adam steps",
    )
    parser.add_argument(
        "--reference-workers", type=int, default=1, help="Concurrent HF storage checks"
    )
    parser.add_argument(
        "--reference-nodes",
        type=int,
        default=1,
        help="Distribute the exact checkpoint oracle across this many Ray GPU nodes",
    )
    parser.add_argument(
        "--reference-report", help="Matching backend/TP PP1 output oracle"
    )
    parser.add_argument("--rebuild-after-stage-loss", action="store_true")
    parser.add_argument("--abort-during-refit", action="store_true")
    parser.add_argument("--refit-timeout", type=float, default=120)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    use_optimizer = args.update_mode == "optimizer"
    args.train_etp = (
        (1 if args.backend == "vllm" else args.train_tp)
        if args.train_etp is None
        else args.train_etp
    )
    args.gen_etp = (
        (1 if args.gen_ep > 1 else args.gen_tp)
        if args.gen_etp is None
        else args.gen_etp
    )
    if args.backend == "vllm" and args.gen_etp != (
        1 if args.gen_ep > 1 else args.gen_tp
    ):
        parser.error("vLLM ETP must be 1 with expert parallelism, otherwise TP")
    hf_config = json.loads((Path(args.model) / "config.json").read_text())
    if hf_config["model_type"] not in ("qwen3", "qwen3_moe", "apertus", "glm_moe_dsa"):
        parser.error("The independent storage oracle supports Qwen3, Apertus and GLM-5")
    if args.router_replay and (
        args.backend != "vllm"
        or hf_config["model_type"] not in ("qwen3_moe", "glm_moe_dsa")
        or use_optimizer
    ):
        parser.error("The paired replay diagnostic requires vLLM MoE and scale updates")
    if (args.train_tp * args.train_dp) % (args.train_ep * args.train_etp):
        parser.error("Training TP*DP must divide evenly into EP*ETP groups")
    if hf_config["num_key_value_heads"] % args.gen_tp:
        parser.error("The storage oracle requires evenly sharded KV heads")
    if args.warm_refits < 1:
        parser.error("--warm-refits must be positive")
    if args.reference_workers < 1:
        parser.error("--reference-workers must be positive")
    if args.reference_nodes < 1:
        parser.error("--reference-nodes must be positive")
    if args.rebuild_after_stage_loss and (
        args.backend != "vllm" or args.gen_dp != 2 or args.gen_pp < 2
    ):
        parser.error("Stage-loss recovery requires vLLM, DP2 and PP>1")
    if args.abort_during_refit and (args.backend != "vllm" or args.gen_pp < 2):
        parser.error("In-flight stage loss requires vLLM and PP>1")
    reference_report = None
    if args.reference_report:
        reference_report = json.loads(Path(args.reference_report).read_text())
        assert reference_report["passed"]
        assert reference_report["config"]["backend"] == args.backend
        assert reference_report["config"]["gen_tp"] == args.gen_tp
        assert reference_report["config"]["gen_pp"] == 1
        assert reference_report["config"]["model"] == args.model
        assert (
            reference_report["config"].get("update_mode", "scale") == args.update_mode
        )
    report = {
        "config": vars(args),
        "refit_seconds": [],
        "checks": [],
        "snapshot_files": [],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    register_omegaconf_resolvers()
    config = load_config(args.config)
    config.policy.model_name = args.model
    config.policy.max_total_sequence_length = 128
    config.policy.train_micro_batch_size = 1
    batch_size = (2 if use_optimizer else 1) * args.train_dp
    config.policy.train_global_batch_size = batch_size
    config.policy.generation_batch_size = 1
    config.policy.sequence_packing.enabled = False
    config.policy.megatron_cfg.tensor_model_parallel_size = args.train_tp
    config.policy.megatron_cfg.pipeline_model_parallel_size = args.train_pp
    config.policy.megatron_cfg.expert_model_parallel_size = args.train_ep
    config.policy.megatron_cfg.expert_tensor_parallel_size = args.train_etp
    config.policy.megatron_cfg.train_iters = 10
    if hf_config["model_type"] in ("qwen3_moe", "glm_moe_dsa") and args.train_tp > 1:
        # Megatron's MoE training forward requires SP when attention uses TP.
        config.policy.megatron_cfg.sequence_parallel = True
    if hf_config["model_type"] == "apertus":
        config.policy.megatron_cfg.bias_activation_fusion = False
    config.policy.megatron_cfg.optimizer.use_distributed_optimizer = use_optimizer
    if use_optimizer:
        config.policy.megatron_cfg.optimizer.lr = 1e-4
        config.policy.megatron_cfg.optimizer.min_lr = 1e-4
        config.policy.megatron_cfg.scheduler.lr_warmup_iters = 0
    config.policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather = (
        False
    )
    gen = config.policy.generation
    gen.backend = args.backend
    gen.refit_transport = "nccl_reshard"
    gen.colocated.enabled = False
    gen.max_new_tokens = 8
    gen.vllm_cfg.async_engine = True
    gen.vllm_cfg.tensor_parallel_size = args.gen_tp
    gen.vllm_cfg.pipeline_parallel_size = args.gen_pp
    gen.vllm_cfg.expert_parallel_size = args.gen_ep
    gen.vllm_cfg.enforce_eager = True
    gen.vllm_cfg.logprobs_mode = "raw_logprobs"
    gen.vllm_cfg.gpu_memory_utilization = args.gpu_memory_utilization
    gen.vllm_cfg.max_model_len = 128
    gen.vllm_kwargs.max_num_seqs = 4
    gen.vllm_kwargs.max_num_batched_tokens = 128
    if hf_config["model_type"] in ("qwen3_moe", "glm_moe_dsa"):
        # The independent CPU oracle compares canonical BF16 expert storage.
        gen.vllm_kwargs.moe_backend = "triton"
    gen.mcore_generation_config.tensor_model_parallel_size = args.gen_tp
    gen.mcore_generation_config.pipeline_model_parallel_size = args.gen_pp
    gen.mcore_generation_config.sequence_parallel = args.gen_tp > 1
    gen.mcore_generation_config.expert_model_parallel_size = args.gen_ep
    gen.mcore_generation_config.expert_tensor_parallel_size = args.gen_etp
    gen.mcore_generation_config.cuda_graph_impl = "none"
    gen.mcore_generation_config.pop("inference_cuda_graph_scope", None)
    gen.mcore_generation_config.logprobs_mode = "raw_logprobs"
    gen.mcore_generation_config.max_tokens = 128
    gen.mcore_generation_config.max_requests = 4
    gen.mcore_generation_config.buffer_size_gb = 1
    check_nccl_reshard_refit_support(config)
    policy_cfg = OmegaConf.to_container(config.policy, resolve=True)
    if args.router_replay:
        policy_cfg["router_replay"] = {"enabled": True}
        configure_vllm_for_router_replay(policy_cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    policy_cfg["generation"]["model_name"] = args.model
    policy_cfg["generation"] = configure_generation_config(
        policy_cfg["generation"], tokenizer
    )
    fqn = "tests.functional.nccl_reshard_pp_megatron.PipelineRefitMegatronWorker"
    if use_optimizer:
        fqn = "tests.functional.nccl_reshard_pp_train_worker.PipelineTrainingRefitMegatronWorker"
    base_worker = (
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )
    actor_env = get_actor_python_env(base_worker)
    if image_venvs_enabled():
        actor_env = image_venv_python(actor_env, base_worker)
    ACTOR_ENVIRONMENT_REGISTRY[fqn] = actor_env
    policy_cfg["worker_extension_cls_fqn"] = fqn
    if args.backend == "vllm":
        fqn = "tests.functional.nccl_reshard_pp_vllm.PipelineRefitVllmWorker"
        if args.rebuild_after_stage_loss or args.abort_during_refit:
            fqn = "tests.functional.nccl_reshard_pp_fault_vllm.PipelineFaultVllmWorker"
        base_worker = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
        actor_env = get_actor_python_env(base_worker)
        if image_venvs_enabled():
            actor_env = image_venv_python(actor_env, base_worker)
        ACTOR_ENVIRONMENT_REGISTRY[fqn] = actor_env
        policy_cfg["generation"]["worker_extension_cls_fqn"] = fqn
    init_ray(
        log_dir=None if os.environ.get("RAY_ADDRESS") else f"/tmp/pp-gate-{os.getpid()}"
    )
    train_cluster = RayVirtualCluster(
        [args.train_tp * args.train_pp * args.train_dp],
        num_gpus_per_node=4,
        name="pp_gate_train",
    )
    gen_cluster = RayVirtualCluster(
        [args.gen_tp * args.gen_pp * args.gen_dp],
        num_gpus_per_node=4,
        name="pp_gate_gen",
    )
    policy = Policy(
        train_cluster,
        policy_cfg,
        tokenizer,
        init_optimizer=use_optimizer,
        init_reference_model=False,
    )
    if args.backend == "vllm":
        generation = VllmGeneration(gen_cluster, policy_cfg["generation"])
    else:
        generation = MegatronGeneration(
            policy_cfg, tokenizer, cluster=gen_cluster, skip_weight_load=True
        )
    sync = create_weight_synchronizer(
        policy,
        generation,
        args.backend,
        False,
        train_cluster,
        gen_cluster,
        refit_timeout_s=args.refit_timeout,
    )
    generation.weight_synchronizer = sync
    start = time.perf_counter()
    sync.init_communicator()
    report["communicator_setup_seconds"] = time.perf_counter() - start
    batch = training_batch(tokenizer, batch_size) if use_optimizer else None
    previous_hashes = None
    if use_optimizer:
        report["optimizer_updates"] = []
    last_iteration = args.warm_refits + int(args.rebuild_after_stage_loss)
    for iteration in range(last_iteration + 1):
        if iteration > args.warm_refits:
            report["stage_loss_recovery"] = lose_stage_between_refits(
                generation, sync, out.with_suffix(".stage-loss"), args.gen_pp - 1
            )
        reference_dir = None
        if iteration and use_optimizer:
            policy.prepare_for_training()
            training = policy.train(batch, NLLLossFn(), gbs=batch_size, mbs=1)
            policy.finish_training()
            policy.sync_params_before_refit()
            assert torch.isfinite(torch.as_tensor(training["loss"])).all(), training
            assert 0 < training["grad_norm"] < float("inf"), training
            report["optimizer_updates"].append(
                {
                    "loss": float(training["loss"]),
                    "grad_norm": float(training["grad_norm"]),
                }
            )
            print(f"OPTIMIZER_STEP iteration={iteration} {training}", flush=True)
            policy.prepare_for_lp_inference()
            reference_dir = tempfile.mkdtemp(
                prefix=f"hf-reference-{iteration}-", dir=out.parent
            )
            exports = ray.get(
                [
                    w.export_refit_reference.remote(directory=reference_dir)
                    for w in policy.worker_group.workers
                ]
            )
            assert len({item["tensors"] for item in exports}) == 1, exports
            assert exports[0]["tensors"] > 0, exports
            print(f"FULL_HF_EXPORT iteration={iteration} {exports[0]}", flush=True)
        elif iteration:
            ray.get(
                [
                    worker.mutate_refit_weights.remote()
                    for worker in policy.worker_group.workers
                ]
            )
        start = time.perf_counter()
        metrics = refit_policy_generation(policy, generation, colocated_inference=False)
        elapsed = time.perf_counter() - start
        report["refit_seconds"].append(elapsed)
        # Preserve partial timings if a later, untimed correctness check fails.
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"REFIT iteration={iteration} seconds={elapsed:.6f} metrics={metrics}",
            flush=True,
        )
        if args.backend == "vllm":
            nested = ray.get(
                [
                    w.inspect_refit_weights.remote()
                    for w in generation._refit_leader_workers()
                ]
            )
            snapshots = [s for engine in nested for s in engine]
        else:
            snapshots = ray.get(
                [
                    w.inspect_refit_weights.remote()
                    for w in generation._policy.worker_group.workers
                ]
            )
        # Persist hashes and layout metadata so oracle failures can be diagnosed
        # offline without loading another full model. No weight tensors are saved.
        snapshot_file = out.with_name(f"{out.stem}.snapshots-{iteration}.json")
        snapshot_file.write_text(json.dumps(snapshots, separators=(",", ":")) + "\n")
        report["snapshot_files"].append(str(snapshot_file))
        out.write_text(json.dumps(report, indent=2) + "\n")
        if use_optimizer:
            hashes = {
                (snap["rank"], name): value["sha256"]
                for snap in snapshots
                for name, value in snap["params"].items()
            }
            if previous_hashes is not None:
                changed = sum(
                    value != previous_hashes[key] for key, value in hashes.items()
                )
                assert changed > 0, "Optimizer did not change any transferred weight"
                report["optimizer_updates"][-1]["changed_bulk_shards"] = changed
            previous_hashes = hashes
        checked = verify_distributed_snapshots(
            snapshots,
            reference_dir or args.model,
            0 if use_optimizer else iteration,
            reference_workers=args.reference_workers,
            reference_nodes=args.reference_nodes,
        )
        if reference_dir is not None:
            shutil.rmtree(reference_dir)
        report["checks"].append(checked)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"EXACT_CHECK iteration={iteration} {checked}", flush=True)
        if iteration in (0, 1, args.warm_refits, last_iteration):
            generation.prepare_for_generation()
            tokens = tokenizer("The capital of France is", return_tensors="pt")[
                "input_ids"
            ]
            prompt_length = tokens.shape[1]
            inputs = BatchedDataDict(
                {"input_ids": tokens, "input_lengths": torch.tensor([prompt_length])}
            )
            if args.backend == "vllm":
                result = asyncio.run(
                    generate_one(
                        generation,
                        inputs,
                        dp_shard_idx=1 if iteration > args.warm_refits else None,
                    )
                )
            else:
                result = generation.generate(inputs, greedy=True)
            generation.finish_generation()
            policy.prepare_for_lp_inference()
            policy_data = BatchedDataDict(
                {
                    # Supply one identical reference sequence per training
                    # DP rank, as the policy requires evenly sized shards.
                    "input_ids": result["output_ids"].repeat(args.train_dp, 1),
                    "input_lengths": result["unpadded_sequence_lengths"].repeat(
                        args.train_dp
                    ),
                }
            )
            length = int(result["unpadded_sequence_lengths"][0])
            a = result["logprobs"][0, prompt_length:length]
            if args.router_replay:
                route_file = out.with_name(f"{out.stem}.routes-{iteration}.json")
                route_file.write_text(
                    json.dumps(result["routed_experts"].tolist()) + "\n"
                )
                checked["route_file"] = str(route_file)
                checked["tokens"] = result["output_ids"][0].tolist()
                checked["prompt_length"] = prompt_length
                checked["generation_logprobs"] = a.tolist()
                out.write_text(json.dumps(report, indent=2) + "\n")
                checked["route_coverage"] = check_complete_routes(result, hf_config)
                control = get_unreplayed_logprobs(policy, policy_data)
                control_lp = control["logprobs"][0, prompt_length:length]
                checked["source_logprobs_without_replay"] = control_lp.tolist()
                checked["unreplayed_logprob_max_abs_error"] = (
                    (a - control_lp).abs().max().item()
                )
                checked["unreplayed_logprob_mean_abs_error"] = (
                    (a - control_lp).abs().mean().item()
                )
                policy_data["routed_experts"] = result["routed_experts"].repeat(
                    args.train_dp, 1, 1, 1
                )
                out.write_text(json.dumps(report, indent=2) + "\n")
            computed = policy.get_logprobs(policy_data)
            b = computed["logprobs"][0, prompt_length:length]
            diff = (a - b).abs()
            assert len(diff) > 0 and torch.isfinite(diff).all()
            # Different BF16 inference kernels need not match Megatron's training
            # forward numerically. All vLLM parameters are checked bitwise above;
            # PP output parity uses a same-backend, same-TP PP1 reference below.
            if args.backend == "megatron":
                assert diff.max().item() < 0.2, (a, b, diff)
            checked["logprob_max_abs_error"] = diff.max().item()
            checked["logprob_mean_abs_error"] = diff.mean().item()
            checked["tokens"] = result["output_ids"][0].tolist()
            checked["prompt_length"] = prompt_length
            checked["generation_logprobs"] = a.tolist()
            checked["source_logprobs"] = b.tolist()
            if reference_report is not None:
                reference_check = reference_report["checks"][iteration]
                assert checked["tokens"] == reference_check["tokens"]
                expected_logprobs = torch.tensor(reference_check["generation_logprobs"])
                torch.testing.assert_close(a, expected_logprobs, atol=1e-5, rtol=0)
                checked["pp1_logprob_max_abs_error"] = (
                    (a - expected_logprobs).abs().max().item()
                )
            print(f"FORWARD_CHECK {checked}", flush=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    warm = report["refit_seconds"][1 : args.warm_refits + 1]
    report["warm_median_seconds"] = statistics.median(warm)
    report["warm_min_seconds"] = min(warm)
    report["warm_max_seconds"] = max(warm)
    if args.abort_during_refit:
        ray.get(
            [
                worker.mutate_refit_weights.remote()
                for worker in policy.worker_group.workers
            ]
        )
        report["inflight_failure"] = lose_stage_during_refit(
            generation,
            lambda: refit_policy_generation(
                policy, generation, colocated_inference=False
            ),
            directory=out.with_suffix(".inflight-loss"),
            stage=args.gen_pp - 1,
            engine_world_size=args.gen_pp * args.gen_tp,
            deadline_seconds=3 * args.refit_timeout + 60,
        )
    report["passed"] = True
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("GATE_PASSED " + json.dumps(report), flush=True)
    if args.rebuild_after_stage_loss or args.abort_during_refit:
        # Only this test's actors. A fatal NCCL abort must not reuse their CUDA contexts.
        for worker in generation.worker_group.workers + policy.worker_group.workers:
            ray.kill(worker, no_restart=True)
    else:
        generation.shutdown()
        policy.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()
