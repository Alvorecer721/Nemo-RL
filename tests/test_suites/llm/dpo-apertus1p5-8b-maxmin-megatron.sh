#!/bin/bash
# Apertus recipe (CSCS Clariden, GH200 4-GPU nodes). Listed in disabled.txt:
# model/tokenizer paths and the container EDF are CSCS-specific, so nightly
# automation cannot run it. The certified launch path is the Slurm wrapper in
# infra/slurm/cscs/ (see docs/apertus-quickstart.md); this driver mirrors that
# invocation for accounting and dry-run coverage.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=8
GPUS_PER_NODE=4
STEPS_PER_RUN=1921
MAX_STEPS=1921
NUM_RUNS=1
NUM_MINUTES=720
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT
uv run examples/run_dpo.py \
    --config $CONFIG_PATH \
    dpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG
