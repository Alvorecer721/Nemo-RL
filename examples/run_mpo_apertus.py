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

from omegaconf import OmegaConf

from nemo_rl.algorithms.dpo import (
    DPO_TIMEOUT_EXIT_CODE,
    DPOTrainStatus,
    MasterConfig,
    dpo_train,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_preference_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def main():
    """Main entry point."""
    args, overrides = parse_args()

    # Fail loud if nemo_rl is the stock /opt copy, not this checkout (see nemo_rl_apertus/runtime_guard.py).
    from nemo_rl_apertus.runtime_guard import assert_apertus_runtime

    assert_apertus_runtime()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "dpo.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config.policy["tokenizer"])

    # setup data
    dataset, val_dataset = setup_preference_data(tokenizer, config.data)

    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        dpo_save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # MPO: swap the stock DPO loss for MPOLossFn (DPO preference + BCO quality
    # + SFT generation; InternVL MPO, arXiv:2411.10442). Everything else in
    # this file is byte-identical to examples/run_dpo.py plus the runtime guard.
    #
    # delta (the BCO running reward anchor) resume threading. The stock
    # dpo_save_state cannot carry it without stock edits: it is driver-side
    # while the loss state is worker-side, the per-step train-metrics path
    # only reaches dpo_save_state through the single checkpointing.metric_name
    # slot, and validate() filters val metrics to DPOValMetrics keys. So the
    # state flows TRL-style (BCOTrainer's running.json):
    #   backward (worker -> disk): every train-microbatch update all-reduces
    #     (sum, count) over the data-parallel group and atomically rewrites
    #     <checkpoint_dir>/mpo_delta.json; `delta`/`delta_count` also appear
    #     in the train metrics (the driver SUMS metrics across microbatches x
    #     DP ranks before logging, so logged values are monitoring aggregates
    #     only — the sidecar is the source of truth).
    #   forward (disk -> worker): on (re)start each worker seeds its
    #     module-resident state from the sidecar if present, else from
    #     dpo.quality_delta_init / dpo.quality_delta_resume_count (the manual
    #     override surface, e.g. when resuming in a fresh checkpoint dir).
    #   Fresh runs (no checkpoint to resume) delete any stale sidecar so a
    #   previous experiment's delta cannot leak in. Caveat: resuming from a
    #   non-latest checkpoint leaves the sidecar delta slightly ahead of that
    #   checkpoint (later batches already folded); delta is a slow-moving
    #   anchor, so the skew is bounded — set the config keys to override.
    from nemo_rl_apertus.mpo_loss import MPOConfig, MPOLossFn, delta_sidecar_path

    mpo_cfg = MPOConfig.model_validate(master_config.dpo.model_dump())
    if master_config.checkpointing["enabled"]:
        sidecar = delta_sidecar_path(master_config.checkpointing["checkpoint_dir"])
        if checkpointer.get_latest_checkpoint_path() is None and os.path.exists(
            sidecar
        ):
            os.remove(sidecar)
        mpo_cfg.quality_delta_state_path = sidecar
    loss_fn = MPOLossFn(
        mpo_cfg,
        use_linear_ce_fusion=master_config.policy["megatron_cfg"]["enabled"]
        and master_config.policy["megatron_cfg"]["use_linear_ce_fusion_loss"],
    )

    # The checkpointer owns background async-checkpoint finalization threads;
    # the context manager guarantees they are flushed (rename + delete) on exit.
    with checkpointer:
        train_status = dpo_train(
            policy,
            train_dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            master_config,
            logger,
            checkpointer,
            dpo_save_state,
        )
    if train_status is DPOTrainStatus.TIMED_OUT:
        return DPO_TIMEOUT_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
