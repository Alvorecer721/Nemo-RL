#!/bin/bash
# Apertus recipe (CSCS Clariden, GH200 4-GPU nodes). Listed in disabled.txt:
# model/tokenizer paths and the container EDF are CSCS-specific, so nightly
# automation cannot run it. Operational Slurm launchers live in
# infra/slurm/cscs/, but this driver only preserves recipe accounting and is
# not evidence that the named recipe or topology has been certified.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=1
NUM_MINUTES=85
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.40 \
    logger.log_dir=$LOG_DIR \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG
