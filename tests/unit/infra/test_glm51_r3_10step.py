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

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from infra.slurm.cscs.autoresearch.validate_glm51_r3_10step import (
    summarize_logprob_tails,
    validate_logprob_tails,
    validate_metrics,
)
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).parents[3]
RECIPE = (
    REPO_ROOT
    / "examples/configs/recipes/llm/autoresearch"
    / "grpo-glm5.1-80n4g-megatron-tp2pp18ep16-async-vllm-tp32-r3-10step.yaml"
)


def test_glm51_r3_recipe_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLM_CKPT", "/models/glm-5.1")
    monkeypatch.setenv("GLM_RUN_DIR", "/run")
    register_omegaconf_resolvers()
    config = MasterConfig(**OmegaConf.to_container(load_config(RECIPE), resolve=True))
    megatron = config.policy["megatron_cfg"]
    generation = config.policy["generation"]

    assert config.grpo.max_num_steps == 10
    assert config.grpo.async_grpo.enabled is True
    assert config.grpo.async_grpo.max_trajectory_age_steps == 1
    assert config.policy["router_replay"]["enabled"] is True
    assert config.policy["max_total_sequence_length"] == 2048
    assert generation["max_new_tokens"] == 1536
    assert generation["vllm_cfg"]["max_model_len"] == 2048
    assert not (config.data_plane or {}).get("enabled", False)
    assert config.checkpointing["enabled"] is False
    assert config.checkpointing["save_optimizer"] is False
    assert (config.cluster["num_nodes"], config.cluster["gpus_per_node"]) == (80, 4)
    assert (
        megatron["tensor_model_parallel_size"],
        megatron["pipeline_model_parallel_size"],
        megatron["expert_tensor_parallel_size"],
        megatron["expert_model_parallel_size"],
    ) == (2, 18, 1, 16)
    assert generation["refit_transport"] == "nccl_reshard"
    assert (
        generation["vllm_cfg"]["tensor_parallel_size"],
        generation["vllm_cfg"]["expert_parallel_size"],
    ) == (32, 32)


def _metrics(
    *, kl: float = 0.0005, signal_steps: int = 10, token_mult: float = 1.01
) -> dict:
    steps = range(1, 11)
    return {
        "train/gen_kl_error": {str(step): kl for step in steps},
        "train/token_mult_prob_error": {str(step): token_mult for step in steps},
        "train/js_divergence_error": {str(step): 0.0003 for step in steps},
        "train/loss": {
            str(step): 0.01 if step <= signal_steps else 0.0 for step in steps
        },
        "train/reward": {
            str(step): 0.1 if step <= signal_steps else 0.0 for step in steps
        },
        "train/advantages/min": {
            str(step): -1.0 if step <= signal_steps else 0.0 for step in steps
        },
        "train/advantages/max": {
            str(step): 1.0 if step <= signal_steps else 0.0 for step in steps
        },
        "train/truncation_rate": {str(step): 0.5 for step in steps},
    }


def test_glm51_r3_validator_reports_improvement_and_learning_signal() -> None:
    summary = validate_metrics(_metrics(signal_steps=8))

    assert summary["steps"] == 10
    assert summary["learning_signal_steps"] == 8
    assert summary["nonzero_loss_steps"] == 8
    assert summary["kl"]["below_0_002_steps"] == 10
    assert summary["kl"]["mean_ratio_to_historical_r3_off"] < 1.0


@pytest.mark.parametrize(
    "metrics",
    [
        _metrics(kl=0.001),
        _metrics(signal_steps=7),
        _metrics(token_mult=1.02),
    ],
)
def test_glm51_r3_validator_rejects_weak_evidence(metrics: dict) -> None:
    with pytest.raises(ValueError):
        validate_metrics(metrics)


def test_glm51_r3_validator_summarizes_direct_logprob_tails(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_001"
    exp_dir.mkdir()
    for step in range(1, 11):
        record = {
            "generation_logprobs": [[-1.0, -2.0, -3.0, -4.0]],
            "prev_logprobs": [[-1.0, -2.1, -3.6, -6.0]],
            "token_loss_mask": [[0, 1, 1, 1]],
            "sample_loss_mask": [1],
        }
        (exp_dir / f"train_data_step{step}.jsonl").write_text(json.dumps(record) + "\n")

    summary = summarize_logprob_tails(tmp_path)

    assert summary["total_tokens"] == 30
    assert summary["count_gt_0_5"] == 20
    assert summary["count_gt_1_0"] == 10
    assert summary["per_step"]["1"]["max_abs"] == 2.0


@pytest.mark.parametrize(
    "summary",
    [
        {"total_tokens": 1_000_000, "count_gt_0_5": 100, "count_gt_1_0": 0},
        {"total_tokens": 1_000_000, "count_gt_0_5": 1, "count_gt_1_0": 1},
    ],
)
def test_glm51_r3_validator_rejects_logprob_tails(summary: dict) -> None:
    with pytest.raises(ValueError):
        validate_logprob_tails(summary)


def test_glm51_r3_validator_accepts_observed_r3_tail() -> None:
    summary = validate_logprob_tails(
        {"total_tokens": 1_291_712, "count_gt_0_5": 4, "count_gt_1_0": 0}
    )

    assert summary["fraction_gt_0_5"] < 1.0e-4


def test_glm51_r3_validator_rejects_old_truncation_regime() -> None:
    metrics = _metrics()
    metrics["train/truncation_rate"] = {str(step): 0.95 for step in range(1, 11)}

    with pytest.raises(ValueError, match="truncates too many"):
        validate_metrics(metrics)


def test_glm51_r3_launcher_uses_cluster_and_route_safety_controls() -> None:
    submitter = (
        REPO_ROOT / "infra/slurm/cscs/autoresearch/submit_glm51_r3_10step.sh"
    ).read_text()
    runner = (
        REPO_ROOT / "infra/slurm/cscs/autoresearch/run_glm51_r3_10step.sh"
    ).read_text()

    assert "docker/nemo_rl_vllm0251.toml" in submitter
    assert "GLM_RESERVATION=${GLM_RESERVATION-SD-69241-apertus-1-5-0}" in submitter
    assert "RAY_SINGLE_SRUN=1" in submitter
    assert "--nodes=80" in submitter
    assert "--mem=850000M" in submitter
    assert "NRL_ROUTER_REPLAY_VALIDATE=1" in runner
    assert "NRL_R3_TRACE_VERIFY_FORWARD=1" in runner
    assert "--transport-contract legacy-async" in runner
    assert "--require-forward-verify" in runner
    assert "--require-cp-identity" in runner
    assert "R3 router replay fallback:" in runner
    assert "--train-data-dir" in runner
    assert "checkpointing_enabled" in runner
    assert "trap write_failure_terminal EXIT" in runner
    assert '"failure_phase"' in runner
    assert "RUN_DIR=$RUN_ROOT/${NRL_SLURM_JOB_ID:?}" in runner
    assert "Refusing to reuse GLM attempt directory" in runner
