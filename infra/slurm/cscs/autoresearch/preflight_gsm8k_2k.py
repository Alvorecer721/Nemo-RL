# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the requested recipe and cached inputs without loading model weights."""

import hashlib
import json
import os
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf

from infra.slurm.cscs.autoresearch.launch_gsm8k_baked import configure_baked_workers
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.environments.bracket_math_environment import BracketMathEnvConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()
resolved = OmegaConf.to_container(load_config(os.environ["AP_RECIPE"]), resolve=True)
config = MasterConfig.model_validate(resolved)
validate_single_controller_config(config)
assert config.grpo.max_num_steps == 92
assert config.grpo.num_prompts_per_step * config.grpo.num_generations_per_prompt == 768
assert config.grpo.async_grpo is None and config.data_plane["enabled"]
assert config.async_rl.sampler.name == "windowed"
assert config.async_rl.sampler.max_staleness_versions == 1
assert config.async_rl.sampler.sample_freshest_first is False
assert (
    config.loss_fn.force_on_policy_ratio
    and config.grpo.seq_logprob_error_threshold is None
)
assert config.loss_fn.use_importance_sampling_correction
assert config.loss_fn.truncated_importance_sampling_type == "tis"
assert config.loss_fn.truncated_importance_sampling_ratio == 2
assert config.loss_fn.truncated_importance_sampling_ratio_min == 0
assert not config.grpo.reward_shaping.enabled
assert config.policy["max_total_sequence_length"] == 2048
assert config.policy["generation"]["max_new_tokens"] == 2048
assert config.policy["generation"]["vllm_cfg"]["max_model_len"] == 2048
assert config.policy["generation"]["vllm_cfg"]["load_format"] == "dummy"
assert config.policy["generation"]["temperature"] == 1
assert config.policy["generation"]["top_p"] == 1
assert config.cluster["num_nodes"] == 16
BracketMathEnvConfig.model_validate(config.env["bracket_math"])

workers = configure_baked_workers()

data = load_dataset("openai/gsm8k", "main", split="train")
assert len(data) == 7473
tokenizer = get_tokenizer(config.policy["tokenizer"])
system_path = Path(config.data["default"]["system_prompt_file"])
system = system_path.read_text()
assert r"\boxed{42}" in system and "[[[" not in system
lengths = []
digest = hashlib.sha256()
for row in data:
    assert "####" in row["answer"]
    digest.update(json.dumps(row, sort_keys=True).encode())
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": row["question"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert "Deliberation: disabled" in prompt
    lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
eligible = sum(n <= config.data["max_input_seq_length"] for n in lengths)
assert eligible >= 48 * 92 + 96
out = Path(os.environ["AP_PREFLIGHT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
result = {
    "config": "PASS",
    "policy_logprobs_required": False,
    "train_rows": len(data),
    "rows_within_input_limit": eligible,
    "max_prompt_tokens": max(lengths),
    "dataset_content_sha256": digest.hexdigest(),
    "workers": workers,
    "format": "boxed",
    "thinking": False,
    "total_sequence_tokens": 2048,
    "steps": 92,
    "scope": "configuration, inputs and installed worker environments; no 70B gradient parity claim",
}
(out / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
