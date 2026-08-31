# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infra.slurm.cscs.autoresearch.validate_glm51_sc_scale import (
    GLM51ScaleProfile,
    load_scale_config,
    validate_metrics,
    validate_scale_config,
)
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig

REPO_ROOT = Path(__file__).parents[3]
BASE_RECIPE = (
    REPO_ROOT
    / "examples/configs/recipes/llm/autoresearch"
    / "grpo-glm5.1-136n4g-megatron-tp2pp18ep16-ready-first.yaml"
)
MTP_RECIPE = (
    REPO_ROOT
    / "examples/configs/recipes/llm/autoresearch"
    / "grpo-glm5.1-136n4g-megatron-tp2pp18ep16-ready-first-mtp3.yaml"
)


def _load_recipe(
    monkeypatch: pytest.MonkeyPatch, recipe: Path, model_dir: Path
) -> MasterConfig:
    monkeypatch.setenv("GLM_CKPT", str(model_dir))
    monkeypatch.setenv("GLM_RUN_DIR", "/run")
    return load_scale_config(recipe)


def _validate_config(
    config: MasterConfig,
    *,
    speculative_tokens: int,
    speculative_method: str,
    fused_linear_logprobs: bool,
) -> GLM51ScaleProfile:
    return validate_scale_config(
        config,
        expected_total_nodes=136,
        expected_generation_nodes=64,
        expected_steps=10,
        expected_sampler="ready_first",
        expected_speculative_tokens=speculative_tokens,
        expected_speculative_method=speculative_method,
        expected_fused_linear_logprobs=fused_linear_logprobs,
    )


def _write_mtp_model(model_dir: Path) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "num_hidden_layers": 78,
                "num_nextn_predict_layers": 1,
            }
        )
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.78.self_attn.q_proj.weight": "model-00001.safetensors"
                }
            }
        )
    )


def test_glm51_mtp_recipe_preserves_topology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "glm-5.1"
    _write_mtp_model(model_dir)
    config = _load_recipe(monkeypatch, MTP_RECIPE, model_dir)
    profile = _validate_config(
        config,
        speculative_tokens=3,
        speculative_method="deepseek_mtp",
        fused_linear_logprobs=True,
    )

    generation = config.policy["generation"]
    vllm = generation["vllm_cfg"]
    speculative = generation["vllm_kwargs"]["speculative_config"]
    assert config.cluster["num_nodes"] == 136
    assert generation["colocated"]["resources"]["num_nodes"] == 64
    assert (vllm["tensor_parallel_size"], vllm["pipeline_parallel_size"]) == (32, 1)
    assert vllm["expert_parallel_size"] == 32
    assert speculative == {
        "method": "deepseek_mtp",
        "num_speculative_tokens": 3,
    }
    assert config.policy["megatron_cfg"]["use_fused_linear_logprobs"] is True
    assert config.policy["megatron_cfg"]["fused_linear_logprobs_chunk_size"] == 256
    assert (
        config.policy["megatron_cfg"]["distributed_data_parallel_config"][
            "overlap_param_gather"
        ]
        is False
    )
    assert profile.describe().endswith(
        "spec_method=deepseek_mtp spec_tokens=3 fused_logprobs=true "
        "logprob_chunk=256 overlap_param_gather=false"
    )


def test_validate_scale_config_accepts_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _load_recipe(monkeypatch, BASE_RECIPE, tmp_path / "glm-5.1")

    profile = _validate_config(
        config,
        speculative_tokens=0,
        speculative_method="none",
        fused_linear_logprobs=False,
    )

    assert profile.describe() == (
        "glm51_sc_scale_config=OK tp=2 pp=18 etp=1 ep=16 "
        "dense_dp=8 expert_dp=1 total_seq=4096 max_new=3584 "
        "vllm_tp=32 vllm_dp=8 transport=transfer-queue "
        "sampler=ready_first steps=10 spec_method=none spec_tokens=0 "
        "fused_logprobs=false logprob_chunk=256 overlap_param_gather=true"
    )


def test_validate_scale_config_rejects_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _load_recipe(monkeypatch, BASE_RECIPE, tmp_path / "glm-5.1")
    config.policy["generation"]["refit_transport"] = "ipc"

    with pytest.raises(
        ValueError,
        match=(
            "requires policy.generation.refit_transport='nccl_reshard'; resolved 'ipc'"
        ),
    ):
        _validate_config(
            config,
            speculative_tokens=0,
            speculative_method="none",
            fused_linear_logprobs=False,
        )


def test_validate_scale_config_applies_single_controller_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _load_recipe(monkeypatch, BASE_RECIPE, tmp_path / "glm-5.1")
    config.grpo.num_prompts_per_step = 15

    with pytest.raises(
        ValueError,
        match="num_prompts_per_step \\(15\\) must be >= .* \\(16\\)",
    ):
        _validate_config(config, speculative_tokens=0, speculative_method="none")


def test_validate_scale_config_rejects_training_mtp_with_speculation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "glm-5.1"
    _write_mtp_model(model_dir)
    config = _load_recipe(monkeypatch, MTP_RECIPE, model_dir)
    config.policy["megatron_cfg"]["mtp_num_layers"] = 1

    with pytest.raises(
        ValueError,
        match="requires policy.megatron_cfg.mtp_num_layers=0; resolved 1",
    ):
        _validate_config(
            config, speculative_tokens=3, speculative_method="deepseek_mtp"
        )


def test_validate_scale_config_fails_closed_under_python_optimization(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "GLM_CKPT": str(tmp_path / "glm-5.1"),
            "GLM_RUN_DIR": "/run",
            "GLM_TEST_RECIPE": str(BASE_RECIPE),
            "PYTHONOPTIMIZE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    program = """
import os
from pathlib import Path

from infra.slurm.cscs.autoresearch.validate_glm51_sc_scale import (
    load_scale_config,
    validate_scale_config,
)

config = load_scale_config(Path(os.environ["GLM_TEST_RECIPE"]))
config.policy["generation"]["refit_transport"] = "ipc"
validate_scale_config(
    config,
    expected_total_nodes=136,
    expected_generation_nodes=64,
    expected_steps=10,
    expected_sampler="ready_first",
    expected_speculative_tokens=0,
    expected_speculative_method="none",
    expected_fused_linear_logprobs=False,
)
"""

    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "requires policy.generation.refit_transport='nccl_reshard'; resolved 'ipc'"
        in result.stderr
    )


def test_scale_runner_uses_fail_closed_validator() -> None:
    launcher = (
        REPO_ROOT / "infra/slurm/cscs/autoresearch/run_glm51_sc_scale.sh"
    ).read_text()
    preflight = launcher.split("GLM_PHASE=config_preflight", maxsplit=1)[1].split(
        "RUN_LOG=", maxsplit=1
    )[0]

    assert "validate_scale_config(" in preflight
    assert "\nassert " not in preflight


def _metrics(steps: int = 10) -> dict[str, dict[str, float]]:
    indices = range(1, steps + 1)

    def series(value: float) -> dict[str, float]:
        return {str(step): value for step in indices}

    return {
        "train/gen_kl_error": series(0.0004),
        "train/token_mult_prob_error": series(1.01),
        "train/js_divergence_error": series(0.0001),
        "train/loss": series(-0.01),
        "train/reward": series(0.5),
        "train/advantages/min": series(-1.0),
        "train/advantages/max": series(1.0),
        "train/grad_norm": series(0.2),
        "timing/train/total_step_time": series(100.0),
        "timing/train/exposed_generation": series(60.0),
        "timing/train/policy_training": series(30.0),
        "timing/train/weight_sync": series(5.0),
        "timing/train/valid_tokens_per_sec_per_gpu": series(10.0),
        "train/vllm/spec_num_drafts": series(100.0),
        "train/vllm/spec_num_draft_tokens": series(300.0),
        "train/vllm/spec_num_accepted_tokens": series(240.0),
        "train/vllm/spec_acceptance_length": series(3.4),
        "train/vllm/spec_acceptance_rate": series(0.8),
        "train/logprob_tails/valid_tokens": series(100_000.0),
        "train/logprob_tails/mean_abs": series(0.01),
        "train/logprob_tails/p95_abs": series(0.02),
        "train/logprob_tails/p99_abs": series(0.03),
        "train/logprob_tails/max_abs": series(0.4),
        "train/logprob_tails/count_gt_0_5": series(1.0),
        "train/logprob_tails/count_gt_1_0": series(0.0),
    }


def test_validate_metrics_accepts_ten_green_steps() -> None:
    summary = validate_metrics(_metrics(), expected_steps=10)

    assert summary["steps"] == 10
    assert summary["learning_signal_steps"] == 10
    assert summary["timing"]["total_step_time"]["steady_state_mean"] == 100.0
    assert summary["per_token_logprob_tails"]["total_tokens"] == 1_000_000
    assert summary["per_token_logprob_tails"]["fraction_gt_0_5"] == pytest.approx(
        1.0e-5
    )


def test_validate_metrics_accepts_eight_of_ten_learning_signal_steps() -> None:
    metrics = _metrics()
    for metric in (
        "train/loss",
        "train/advantages/min",
        "train/advantages/max",
        "train/grad_norm",
    ):
        metrics[metric]["9"] = 0.0
        metrics[metric]["10"] = 0.0

    summary = validate_metrics(metrics, expected_steps=10)

    assert summary["learning_signal_steps"] == 8
    assert summary["nonzero_loss_steps"] == 8
    assert summary["nonzero_grad_steps"] == 8


def test_validate_metrics_reports_speculative_acceptance() -> None:
    summary = validate_metrics(_metrics(), expected_steps=10, speculative_tokens=3)

    spec = summary["speculative_decoding"]
    assert spec["total_drafts"] == 1000
    assert spec["aggregate_acceptance_rate"] == pytest.approx(0.8)
    assert spec["aggregate_acceptance_length"] == pytest.approx(3.4)


def test_validate_metrics_rejects_inconsistent_speculative_counters() -> None:
    metrics = _metrics()
    metrics["train/vllm/spec_acceptance_rate"]["4"] = 0.9

    with pytest.raises(ValueError, match="Inconsistent speculative acceptance rate"):
        validate_metrics(metrics, expected_steps=10, speculative_tokens=3)


def test_validate_metrics_rejects_bad_logprob_tail_counters() -> None:
    metrics = _metrics()
    metrics["train/logprob_tails/count_gt_1_0"]["4"] = 1.0

    with pytest.raises(ValueError, match=r"abs\(delta log p\) > 1.0"):
        validate_metrics(metrics, expected_steps=10)


def test_validate_metrics_rejects_seven_of_ten_learning_signal_steps() -> None:
    metrics = _metrics()
    for metric in (
        "train/loss",
        "train/advantages/min",
        "train/advantages/max",
        "train/grad_norm",
    ):
        for step in (8, 9, 10):
            metrics[metric][str(step)] = 0.0

    with pytest.raises(ValueError, match="required=8"):
        validate_metrics(metrics, expected_steps=10)


@pytest.mark.parametrize(
    ("metric", "value", "match"),
    (
        ("train/gen_kl_error", 0.001, "KL escaped"),
        ("train/loss", 0.0, "learning-signal"),
        ("train/grad_norm", 0.0, "learning-signal"),
    ),
)
def test_validate_metrics_rejects_failed_gates(
    metric: str, value: float, match: str
) -> None:
    metrics = _metrics()
    metrics[metric] = {str(step): value for step in range(1, 11)}

    with pytest.raises(ValueError, match=match):
        validate_metrics(metrics, expected_steps=10)
