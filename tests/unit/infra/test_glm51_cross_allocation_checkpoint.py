from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


REPO_ROOT = Path(__file__).parents[3]
RECIPE_ROOT = REPO_ROOT / "examples/configs/recipes/llm/autoresearch"


@pytest.mark.parametrize("phase", ["save", "resume"])
def test_glm51_checkpoint_recipe_contract(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GLM_CKPT", "/models/glm-5.1")
    monkeypatch.setenv("GLM_RESUME_CHECKPOINT_DIR", "/checkpoints")
    monkeypatch.setenv("GLM_RUN_DIR", "/run")
    register_omegaconf_resolvers()
    recipe = (
        RECIPE_ROOT
        / f"grpo-glm5.1-80n4g-megatron-async-vllm-tp32-checkpoint-{phase}.yaml"
    )
    config = MasterConfig(**OmegaConf.to_container(load_config(recipe), resolve=True))
    megatron = config.policy["megatron_cfg"]
    generation = config.policy["generation"]
    vllm = generation["vllm_cfg"]

    assert (config.cluster["num_nodes"], config.cluster["gpus_per_node"]) == (
        80,
        4,
    )
    assert generation["colocated"]["resources"]["num_nodes"] == 8
    assert (
        megatron["tensor_model_parallel_size"],
        megatron["pipeline_model_parallel_size"],
        megatron["expert_model_parallel_size"],
    ) == (1, 18, 16)
    assert (vllm["tensor_parallel_size"], vllm["expert_parallel_size"]) == (
        32,
        32,
    )
    assert megatron["checkpoint"]["async_save"] is True
    assert megatron["checkpoint"]["fully_parallel_save"] is False
    assert generation["refit_transport"] == "nccl_reshard"
    assert config.checkpointing["save_optimizer"] is True
    assert config.checkpointing["enabled"] is (phase == "save")


def test_glm51_checkpoint_launcher_uses_cluster_safety_controls() -> None:
    launcher = (
        REPO_ROOT
        / "infra/slurm/cscs/autoresearch/submit_glm51_cross_allocation_checkpoint.sh"
    ).read_text()

    assert "CONTAINER_ENV" in launcher
    assert "docker/nemo_rl_vllm0251.toml" in launcher
    assert "RAY_SINGLE_SRUN=1" in launcher
    assert "RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-68719476736}" in launcher
    assert "--mem=850000M" in launcher
    assert "--reservation=SD-69241-apertus-1-5-0" in launcher


def test_glm51_checkpoint_runner_captures_rank_writer_failures() -> None:
    runner = (
        REPO_ROOT
        / "infra/slurm/cscs/autoresearch/run_glm51_cross_allocation_checkpoint.sh"
    ).read_text()
    diagnostics = (
        REPO_ROOT / "infra/slurm/cscs/autoresearch/collect_ray_node_diagnostics.py"
    ).read_text()

    assert "checkpoint_start_nodes.json" in runner
    assert "checkpoint_stall_nodes.json" in runner
    assert "memory.events" in diagnostics
    assert "memory.peak" in diagnostics
    assert "NodeAffinitySchedulingStrategy" in diagnostics
    assert "glm51-checkpoint-$EXPECTED_HEAD" in runner


def test_glm51_checkpoint_runner_uses_preserved_slurm_job_id() -> None:
    ray_launcher = (REPO_ROOT / "ray.sub").read_text()
    runner = (
        REPO_ROOT
        / "infra/slurm/cscs/autoresearch/run_glm51_cross_allocation_checkpoint.sh"
    ).read_text()

    preserve = "export NRL_SLURM_JOB_ID=${NRL_SLURM_JOB_ID:-${SLURM_JOB_ID:?}}"
    clear_slurm = "for v in \\$(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/"
    assert preserve in ray_launcher
    assert ray_launcher.index(preserve) < ray_launcher.index(clear_slurm)
    assert 'os.environ["NRL_SLURM_JOB_ID"]' in runner
    assert 'os.environ["SLURM_JOB_ID"]' not in runner
