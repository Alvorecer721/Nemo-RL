# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os
import pprint
import time

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    grpo_train,
    setup,
    shutdown_environments,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import should_use_async_rollouts
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing
from nemo_rl.utils.timer import Timer


def _select_trainer(master_config: MasterConfig):
    """Pick the synchronous trainer based on ``data_plane.enabled``.

    Factored out so test_architecture_invariants can verify dispatch
    without the full setup() path.
    """
    dp_cfg = master_config.data_plane or {}
    if dp_cfg.get("enabled", False):
        from nemo_rl.algorithms.grpo_sync import grpo_train_sync

        print("🚀 Running synchronous GRPO training (TransferQueue)")
        return grpo_train_sync
    print("🚀 Running synchronous GRPO training (legacy)")
    return grpo_train


def _validate_entrypoint_contract(master_config: MasterConfig) -> None:
    """Reject configs that select a training path this entrypoint cannot run.

    This must run before Ray, tokenizer, dataset, or worker setup.  The legacy
    async loop and the SingleController loop have different replay buffers and
    transport contracts; constructing one path's policy and then calling the
    other path's trainer is not a supported hybrid.
    """
    async_config = master_config.grpo.async_grpo
    if async_config is None:
        raise ValueError(
            "examples/run_grpo.py requires grpo.async_grpo to be present. "
            "A null block selects the SingleController config schema; launch it "
            "with examples/run_grpo_single_controller.py instead."
        )

    # ``MasterConfig`` intentionally allows extension fields, so a SingleController
    # block otherwise parses here and is silently ignored by the legacy trainer.
    if getattr(master_config, "async_rl", None) is not None:
        raise ValueError(
            "async_rl.* is consumed only by examples/run_grpo_single_controller.py; "
            "examples/run_grpo.py would ignore it. Remove async_rl or use the "
            "SingleController entrypoint with grpo.async_grpo: null."
        )

    if not async_config.enabled:
        return

    if (master_config.data_plane or {}).get("enabled", False):
        raise ValueError(
            "Legacy async GRPO does not support data_plane.enabled=true. It uses "
            "the in-memory ReplayBuffer, while TransferQueue async training is "
            "owned by SingleController. Set data_plane.enabled=false, or use "
            "examples/run_grpo_single_controller.py with grpo.async_grpo: null."
        )

    generation_config = master_config.policy.get("generation")
    backend = generation_config.get("backend", "") if generation_config else ""
    if backend not in ("vllm", "megatron", "trtllm", "dynamo"):
        raise ValueError(
            "Legacy async GRPO supports vLLM, Megatron, TRT-LLM, and Dynamo "
            f"generation; got policy.generation.backend={backend!r}."
        )
    if not should_use_async_rollouts(generation_config):
        raise ValueError(
            "Legacy async GRPO requires an async generation engine. Enable the "
            "backend's async engine before launching."
        )

    unsupported: list[str] = []
    if master_config.grpo.use_dynamic_sampling:
        unsupported.append("grpo.use_dynamic_sampling")
    if master_config.grpo.reward_scaling.enabled:
        unsupported.append("grpo.reward_scaling.enabled")
    if master_config.grpo.reward_shaping.enabled:
        unsupported.append("grpo.reward_shaping.enabled")
    if master_config.data["use_multiple_dataloader"]:
        unsupported.append("data.use_multiple_dataloader")
    if unsupported:
        raise NotImplementedError(
            "Legacy async GRPO does not consume these enabled settings: "
            + ", ".join(unsupported)
            + ". Disable them or use synchronous GRPO."
        )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def main() -> None:
    """Main entry point."""
    main_start = time.perf_counter()
    log_container_init_timing()

    rl_init_timer = Timer(context={"worker": "driver"})

    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_1B.yaml"
        )

    with rl_init_timer.time("config"):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")

        if overrides:
            print(f"Overrides: {overrides}")
            config = parse_hydra_overrides(config, overrides)

        config = OmegaConf.to_container(config, resolve=True)
        config = MasterConfig(**config)
        _validate_entrypoint_contract(config)
        print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    with rl_init_timer.time("ray_connect"):
        init_ray()

    # setup tokenizer
    with rl_init_timer.time("tokenizer"):
        tokenizer = get_tokenizer(config.policy["tokenizer"])
        assert config.policy["generation"] is not None, (
            "A generation config is required for GRPO"
        )
        has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
        megatron_cfg = config.policy.get("megatron_cfg") or {}
        trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
        config.policy["generation"] = configure_generation_config(
            config.policy["generation"],
            tokenizer,
            has_refit_draft_weights=has_refit_draft_weights,
            trains_mtp=trains_mtp,
        )

    # setup data
    with rl_init_timer.time("data"):
        dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
            tokenizer, config.data, config.env
        )

    # Pick the policy factory at the launcher level so the legacy trainer
    # stays data-plane-agnostic (architectural invariant — see
    # tests/data_plane/unit/test_architecture_invariants.py).
    _dp_cfg = config.data_plane or {}
    if _dp_cfg.get("enabled", False):
        from nemo_rl.models.policy.tq_policy import TQPolicy

        def _make_policy(**kwargs):
            return TQPolicy(**kwargs, dp_cfg=_dp_cfg)

        _policy_factory = _make_policy
    else:
        _policy_factory = None  # setup() defaults to plain Policy

    with rl_init_timer.time("setup"):
        (
            policy,
            policy_generation,
            _nemo_gym,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
            teacher_worker_groups,
            alias_to_group_alias,
        ) = setup(
            config,
            tokenizer,
            dataset,
            val_dataset,
            policy_factory=_policy_factory,
        )

    rl_init_timer.record("total", time.perf_counter() - main_start)

    rl_init_metrics = rl_init_timer.get_timing_metrics(reduction_op="sum")
    print("\n" + "=" * 60)
    print(" " * 14 + "RL INIT TIMING BREAKDOWN")
    for label, value in sorted(rl_init_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {label}: {value:.1f}s")
    print("=" * 60 + "\n", flush=True)

    try:
        # Check if async mode is enabled. Unsupported combinations were rejected
        # before Ray startup by _validate_entrypoint_contract.
        if config.grpo.async_grpo.enabled:
            from nemo_rl.algorithms.grpo import async_grpo_train

            print("🚀 Running async GRPO training")

            # Run async GRPO training
            async_grpo_train(
                policy=policy,
                policy_generation=policy_generation,
                dataloader=dataloader,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                task_to_env=task_to_env,
                val_task_to_env=val_task_to_env,
                logger=logger,
                checkpointer=checkpointer,
                grpo_save_state=grpo_state,
                master_config=master_config,
                max_trajectory_age_steps=config.grpo.async_grpo.max_trajectory_age_steps,
                teacher_worker_groups=teacher_worker_groups,
                alias_to_group_alias=alias_to_group_alias,
            )
        else:
            # Two parallel synchronous trainers (verl-style — main_ppo.py vs
            # main_ppo_sync.py). data_plane.enabled selects which one runs.
            trainer = _select_trainer(master_config)
            # grpo_train_sync defers checkpoint finalization to the checkpointer's
            # background threads; the context manager guarantees they are flushed on
            # exit. (grpo_train also flushes internally; shutdown() is idempotent.)
            with checkpointer:
                trainer(
                    policy,
                    policy_generation,
                    dataloader,
                    val_dataloader,
                    tokenizer,
                    loss_fn,
                    task_to_env,
                    val_task_to_env,
                    logger,
                    checkpointer,
                    grpo_state,
                    master_config,
                )
    finally:
        shutdown_environments(task_to_env, val_task_to_env)
        try:
            policy_generation.shutdown()
        except Exception as error:
            print(f"Error shutting down generation: {error}", flush=True)


if __name__ == "__main__":
    main()
