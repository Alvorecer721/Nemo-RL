#!/usr/bin/env python3
"""Resolve the Apertus-1.5 70B async config without allocating model tensors."""

import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from apertus70b_local_dapo import (
    DAPO_LOGICAL_ROWS,
    DAPO_SMOKE_INDICES_SHA256,
    DAPO_SMOKE_ROWS,
    DAPO_SMOKE_SEED,
    LocalFormattedDAPOSmokeDataset,
    indices_sha256,
)
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.weight_sync.nccl_reshard_utils import check_nccl_reshard_refit_support


def main() -> None:
    recipe = Path(os.environ["APERTUS70B_RECIPE"])
    checkpoint = Path(os.environ["APERTUS70B_CKPT"])
    dataset_path = Path(os.environ["APERTUS70B_DAPO_ARROW"])
    cache = Path(os.environ["APERTUS70B_MEGATRON_CACHE"])
    register_omegaconf_resolvers()
    config = MasterConfig(**OmegaConf.to_container(load_config(recipe), resolve=True))

    total_nodes = config.cluster["num_nodes"]
    gpus_per_node = config.cluster["gpus_per_node"]
    generation = config.policy["generation"]
    generation_nodes = generation["colocated"]["resources"]["num_nodes"]
    train_nodes = total_nodes - generation_nodes
    train_world = train_nodes * gpus_per_node
    generation_world = generation_nodes * gpus_per_node
    megatron = config.policy["megatron_cfg"]
    train_tp = megatron["tensor_model_parallel_size"]
    train_pp = megatron["pipeline_model_parallel_size"]
    dense_dp = train_world // (train_tp * train_pp)
    vllm = generation["vllm_cfg"]
    rollout_dp = generation_world // (
        vllm["tensor_parallel_size"] * vllm["pipeline_parallel_size"]
    )

    assert (total_nodes, train_nodes, generation_nodes) == (5, 4, 1)
    assert (gpus_per_node, train_world, generation_world) == (4, 16, 4)
    assert (train_tp, train_pp, dense_dp) == (4, 4, 1)
    assert megatron["sequence_parallel"] is True
    assert megatron["activation_checkpointing"] is True
    assert megatron["defer_fp32_logits"] is True
    assert megatron["use_fused_linear_logprobs"] is True
    assert config.policy["logprob_chunk_size"] == 256
    assert (vllm["tensor_parallel_size"], rollout_dp) == (4, 1)
    assert vllm["async_engine"] is True
    assert vllm["env_vars"] == {"PYTHONPATH": ""}
    assert generation["refit_transport"] == "nccl_reshard"
    assert config.grpo.num_prompts_per_step == 16
    assert config.grpo.num_generations_per_prompt == 8
    assert config.policy["train_global_batch_size"] == 128
    assert config.policy["train_global_batch_size"] == (
        config.grpo.num_prompts_per_step * config.grpo.num_generations_per_prompt
    )
    assert config.policy["max_total_sequence_length"] == 1536
    assert generation["max_new_tokens"] == 1024
    assert config.grpo.max_num_steps == 3
    assert config.grpo.val_period == 0
    assert config.data["validation"] is None
    assert config.grpo.skip_reference_policy_logprobs_calculation is True
    assert config.loss_fn.reference_policy_kl_penalty == 0.0
    assert config.loss_fn.use_importance_sampling_correction is True
    assert config.grpo.async_grpo.enabled is True
    assert config.grpo.async_grpo.max_trajectory_age_steps == 1
    assert config.grpo.async_grpo.in_flight_weight_updates is True
    check_nccl_reshard_refit_support(config)

    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    checkpoint_bytes = int(index["metadata"]["total_size"])
    assert checkpoint_bytes > 100_000_000_000
    converted = (
        cache
        / "model__capstor_store_cscs_swissai_infra01_apertus_1p5_hf_checkpoints_ap1p5-70b-sft-262k-3000"
        / "iter_0000000"
    )
    assert (converted / ".metadata").is_file()
    assert (converted / "metadata.json").is_file()
    assert len(list(converted.glob("*.distcp"))) == 16

    assert dataset_path.stat().st_size == 1_008_251_856
    dataset = LocalFormattedDAPOSmokeDataset(str(dataset_path), seed=DAPO_SMOKE_SEED)
    assert len(dataset.dataset) == DAPO_SMOKE_ROWS
    assert len(set(dataset.logical_indices)) == DAPO_SMOKE_ROWS
    assert max(dataset.logical_indices) < DAPO_LOGICAL_ROWS
    assert indices_sha256(dataset.logical_indices) == DAPO_SMOKE_INDICES_SHA256
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    task_spec = TaskDataSpec(task_name="DAPOMath17K")
    prompt_lengths = []
    unique_rows = set()
    for index, sample in enumerate(dataset.dataset):
        unique_rows.add(
            hashlib.sha256(
                json.dumps(sample, sort_keys=True).encode("utf-8")
            ).hexdigest()
        )
        processed = math_hf_data_processor(
            sample,
            task_data_spec=task_spec,
            tokenizer=tokenizer,
            max_seq_length=config.data["max_input_seq_length"],
            idx=index,
        )
        prompt_lengths.append(processed["length"])
    assert len(unique_rows) == DAPO_SMOKE_ROWS
    assert max(prompt_lengths) < config.data["max_input_seq_length"]

    print(f"recipe={recipe}")
    print(f"checkpoint_bytes={checkpoint_bytes}")
    print(f"megatron_cache={converted}")
    print(f"train_nodes_world={train_nodes}/{train_world}")
    print(f"train_tp_pp_dp={train_tp}/{train_pp}/{dense_dp}")
    print(f"generation_nodes_world={generation_nodes}/{generation_world}")
    print(f"generation_tp_dp={vllm['tensor_parallel_size']}/{rollout_dp}")
    print(f"prompt_tokens_min_max={min(prompt_lengths)}/{max(prompt_lengths)}")
    print("nccl_reshard_config=OK")
    print("apertus70b_async_config_preflight=OK")


if __name__ == "__main__":
    main()
