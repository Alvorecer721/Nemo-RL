#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Return the latest train/loss step, or zero when metrics do not exist yet.
max_recorded_train_loss_step() {
    jq -er '
        if has("train/loss") then
            (."train/loss" | keys | map(tonumber) | max // 0)
        else
            0
        end
    ' "$JSON_METRICS" 2>/dev/null || echo 0
}

_validate_final_test_suite_run() {
    local exit_code=$?
    local run_index=${NRL_RUN_INDEX:-1}
    local num_runs=${NRL_NUM_RUNS:-1}
    local max_recorded_step

    trap - EXIT

    # Preserve the original failure. The completion check exists only to stop a
    # clean-but-incomplete final run from being reported as successful.
    if [[ $exit_code -ne 0 ]]; then
        exit "$exit_code"
    fi

    if ! [[ $run_index =~ ^[1-9][0-9]*$ ]] || ! [[ $num_runs =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] NRL_RUN_INDEX and NRL_NUM_RUNS must be positive integers" >&2
        exit 1
    fi
    if [[ $run_index -gt $num_runs ]]; then
        echo "[ERROR] NRL_RUN_INDEX=$run_index exceeds NRL_NUM_RUNS=$num_runs" >&2
        exit 1
    fi

    # Slurm chains can finish below MAX_STEPS before their last allocation. The
    # final allocation, and direct invocations (1/1), must prove completion.
    if [[ $run_index -eq $num_runs ]]; then
        max_recorded_step=$(max_recorded_train_loss_step)
        if [[ $max_recorded_step -lt $MAX_STEPS ]]; then
            echo "[ERROR] Final run $run_index/$num_runs exited successfully, but train/loss only reached step $max_recorded_step/$MAX_STEPS" >&2
            exit 1
        fi
    fi
}

exit_if_max_steps_reached() {
    local steps_so_far

    # Install the completion contract before the early-exit path below. A
    # pre-completed run therefore still proves its metrics before returning 0.
    trap _validate_final_test_suite_run EXIT

    steps_so_far=$(max_recorded_train_loss_step)
    if [[ $steps_so_far -ge $MAX_STEPS ]]; then
        echo "[INFO] Target step $MAX_STEPS reached, skipping run"
        exit 0
    fi
    echo "[INFO] Steps so far: $steps_so_far, running till $MAX_STEPS steps"
}
