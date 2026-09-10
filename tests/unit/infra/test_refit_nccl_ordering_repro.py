# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the standalone NCCL refit ordering probe."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


PROBE_PATH = (
    Path(__file__).resolve().parents[2] / "functional" / "refit_nccl_ordering_repro.py"
)
RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "infra"
    / "slurm"
    / "cscs"
    / "autoresearch"
    / "run_refit_nccl_ordering_repro.sh"
)


def _load_probe():
    assert PROBE_PATH.is_file(), f"missing refit probe: {PROBE_PATH}"
    spec = importlib.util.spec_from_file_location(
        "refit_nccl_ordering_repro", PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stages", [2, 4])
@pytest.mark.parametrize("streams", [1, 2])
def test_parse_args_accepts_supported_refit_topologies(stages, streams):
    probe = _load_probe()

    config = probe.parse_args(
        [
            "--stages",
            str(stages),
            "--streams",
            str(streams),
            "--iterations",
            "3",
            "--transfers-per-stage",
            "5",
            "--tensor-mib",
            "1",
        ]
    )

    assert config.stages == stages
    assert config.streams == streams
    assert config.world_size == stages + 1
    assert config.iterations == 3
    assert config.transfers_per_stage == 5


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--stages", "3"),
        ("--streams", "3"),
        ("--iterations", "0"),
        ("--transfers-per-stage", "0"),
        ("--tensor-mib", "0"),
        ("--iteration-timeout-s", "0"),
        ("--coordination-timeout-s", "0"),
        ("--jitter-ms", "-1"),
    ],
)
def test_parse_args_rejects_unsupported_or_unbounded_values(option, value):
    probe = _load_probe()
    argv = ["--stages", "2", "--streams", "1", option, value]

    with pytest.raises((SystemExit, ValueError)):
        probe.parse_args(argv)


def test_runtime_requires_explicit_python_transport_and_matching_stream_count():
    probe = _load_probe()
    config = probe.parse_args(["--stages", "2", "--streams", "2"])
    valid = {
        "NCCL_LAUNCH_ORDER_IMPLICIT": "0",
        "NRL_REFIT_NUM_STREAMS": "2",
        "NRL_XFERDTENSOR_PYTHON": "1",
    }

    probe.validate_runtime_environment(config, valid)
    for name in valid:
        invalid = dict(valid)
        invalid.pop(name)
        with pytest.raises(RuntimeError, match=name):
            probe.validate_runtime_environment(config, invalid)


def test_local_device_selection_uses_slurm_local_id_with_whole_node_visibility():
    probe = _load_probe()
    config = probe.parse_args(["--stages", "2", "--streams", "1"])

    assert (
        probe.select_local_device(
            rank=1,
            config=config,
            slurm_node_id=0,
            slurm_local_id=1,
            requested_local_rank=1,
            visible_device_count=4,
        )
        == 1
    )
    assert (
        probe.select_local_device(
            rank=2,
            config=config,
            slurm_node_id=1,
            slurm_local_id=0,
            requested_local_rank=0,
            visible_device_count=4,
        )
        == 0
    )


def test_local_device_selection_rejects_wrong_device_assignment():
    probe = _load_probe()
    config = probe.parse_args(["--stages", "2", "--streams", "1"])

    with pytest.raises(RuntimeError, match="LOCAL_RANK=0.*SLURM_LOCALID=1"):
        probe.select_local_device(
            rank=1,
            config=config,
            slurm_node_id=0,
            slurm_local_id=1,
            requested_local_rank=0,
            visible_device_count=4,
        )


def test_launcher_uses_whole_node_visibility_and_slurm_local_id():
    runner = RUNNER_PATH.read_text()

    assert "--gpu-bind=none" in runner
    assert "--gpus-per-task" not in runner
    assert "export LOCAL_RANK=$SLURM_LOCALID" in runner


def test_payload_pattern_covers_elements_and_varies_by_iteration():
    probe = _load_probe()
    payload = torch.empty(6, dtype=torch.bfloat16)

    probe.fill_payload(payload, stage=1, iteration=2)
    assert payload.tolist() == [71.0, 72.0, 73.0, 74.0, 75.0, 76.0]

    next_payload = torch.empty_like(payload)
    probe.fill_payload(next_payload, stage=1, iteration=3)
    assert not torch.equal(payload, next_payload)


def test_payload_validation_rejects_corruption_away_from_element_zero():
    probe = _load_probe()
    payload = torch.empty(8, dtype=torch.bfloat16)
    probe.fill_payload(payload, stage=0, iteration=4)
    probe.validate_payload(payload, stage=0, iteration=4)

    payload[6] += 1
    with pytest.raises(RuntimeError, match=r"element 6.*expected.*observed"):
        probe.validate_payload(payload, stage=0, iteration=4)


def _rank_results(*, iterations=3, transfers=5):
    return [
        {
            "rank": 0,
            "role": "source",
            "stage": 0,
            "completed_iterations": iterations,
            "transfer_calls": iterations * transfers,
            "validated_payloads": 0,
            "hostname": "node-a",
            "slurm_node_id": 0,
            "slurm_local_id": 0,
            "cuda_device_index": 0,
            "cuda_device_uuid": "gpu-a0",
            "cuda_device_name": "GH200",
            "max_iteration_s": 0.2,
            "mean_iteration_s": 0.1,
        },
        {
            "rank": 1,
            "role": "source",
            "stage": 1,
            "completed_iterations": iterations,
            "transfer_calls": iterations * transfers,
            "validated_payloads": 0,
            "hostname": "node-a",
            "slurm_node_id": 0,
            "slurm_local_id": 1,
            "cuda_device_index": 1,
            "cuda_device_uuid": "gpu-a1",
            "cuda_device_name": "GH200",
            "max_iteration_s": 0.2,
            "mean_iteration_s": 0.1,
        },
        {
            "rank": 2,
            "role": "receiver",
            "stage": None,
            "completed_iterations": iterations,
            "transfer_calls": iterations * transfers * 2,
            "validated_payloads": iterations * 2,
            "hostname": "node-b",
            "slurm_node_id": 1,
            "slurm_local_id": 0,
            "cuda_device_index": 0,
            "cuda_device_uuid": "gpu-b0",
            "cuda_device_name": "GH200",
            "max_iteration_s": 0.3,
            "mean_iteration_s": 0.2,
        },
    ]


def test_result_validation_requires_every_repeated_transfer_and_payload_check():
    probe = _load_probe()
    config = probe.parse_args(
        [
            "--stages",
            "2",
            "--streams",
            "2",
            "--iterations",
            "3",
            "--transfers-per-stage",
            "5",
        ]
    )

    result = probe.validate_rank_results(_rank_results(), config)

    assert result == {
        "completed_iterations": 3,
        "iterations": 3,
        "stages": 2,
        "status": "pass",
        "streams": 2,
        "transfer_pairs": 30,
        "transfers_per_stage": 5,
        "validated_payloads": 6,
        "world_size": 3,
    }


@pytest.mark.parametrize(
    ("rank", "field", "value", "match"),
    [
        (0, "completed_iterations", 2, "completed_iterations"),
        (1, "transfer_calls", 14, "transfer_calls"),
        (2, "validated_payloads", 5, "validated_payloads"),
        (2, "role", "source", "role"),
        (2, "hostname", "node-a", "duplicate placement"),
        (1, "slurm_node_id", 1, "slurm_node_id"),
        (1, "cuda_device_index", 0, "cuda_device_index"),
        (1, "cuda_device_uuid", "gpu-a0", "duplicate CUDA device"),
    ],
)
def test_result_validation_rejects_incomplete_or_ambiguous_results(
    rank, field, value, match
):
    probe = _load_probe()
    config = probe.parse_args(
        [
            "--stages",
            "2",
            "--streams",
            "1",
            "--iterations",
            "3",
            "--transfers-per-stage",
            "5",
        ]
    )
    results = _rank_results()
    results[rank][field] = value

    with pytest.raises(RuntimeError, match=match):
        probe.validate_rank_results(results, config)


def test_log_validation_rejects_exit_zero_without_a_complete_pass_record(tmp_path):
    probe = _load_probe()
    config = probe.parse_args(
        [
            "--stages",
            "2",
            "--streams",
            "1",
            "--iterations",
            "3",
            "--transfers-per-stage",
            "5",
        ]
    )
    log = tmp_path / "probe.log"
    log.write_text("process exited with status 0\n")

    with pytest.raises(RuntimeError, match="PASS record"):
        probe.validate_result_log(log, config)

    result = probe.validate_rank_results(_rank_results(), config)
    result["teardown"] = "complete"
    log.write_text("noise\nREFIT_ORDER_REPRO=PASS " + json.dumps(result) + "\n")
    assert probe.validate_result_log(log, config) == result


@pytest.mark.parametrize(
    ("exit_code", "validation_exit_code", "log_text", "expected"),
    [
        (0, 0, "REFIT_ORDER_REPRO_LOG=VALID\n", "pass"),
        (124, -1, "terminated\n", "timeout"),
        (143, -1, "peer terminated\n", "error"),
        (143, -1, "refit: deadline exceeded\n", "timeout"),
        (0, -1, "validation failed\n", "error"),
    ],
)
def test_launcher_classifies_only_real_deadlines_as_timeouts(
    tmp_path, exit_code, validation_exit_code, log_text, expected
):
    log = tmp_path / "arm.log"
    log.write_text(log_text)

    completed = subprocess.run(
        [
            str(RUNNER_PATH),
            "--classify-status",
            str(exit_code),
            str(validation_exit_code),
            str(log),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected
