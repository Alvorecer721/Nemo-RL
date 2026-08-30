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

from __future__ import annotations

import json
from pathlib import Path

from tools.check_r3_trace import check_trace


def _tensor_hash(value: str) -> dict[str, str]:
    return {"valid_sha256": value}


def _route_records() -> list[dict]:
    records: list[dict] = []
    for stage in ("prev-logprob", "train"):
        records.extend(
            [
                {"event": "router_replay_assignment", "stage": stage},
                {
                    "event": "router_replay_action",
                    "stage": stage,
                    "action": "replay_forward",
                },
                {
                    "event": "router_replay_forward_verify",
                    "stage": stage,
                    "action": "replay_forward",
                    "matches_expected": True,
                },
            ]
        )
    records.append(
        {
            "event": "router_replay_action",
            "stage": "train",
            "action": "replay_backward",
        }
    )
    records.append(
        {
            "event": "cp_routed_experts",
            "stage": "train",
            "cp_token_identity_verified_count": 8,
        }
    )
    return records


def _tq_records() -> list[dict]:
    records = [
        {
            "event": "rollout_payload_sample",
            "key": "sample-1",
            "valid_length": 4,
            "input_ids": _tensor_hash("input"),
            "routed_experts": _tensor_hash("routes"),
        }
    ]
    for stage in ("prev_lp", "train"):
        records.append(
            {
                "event": "tq_fetch_sample",
                "stage": stage,
                "key": "sample-1",
                "valid_length": 4,
                "input_ids": _tensor_hash("input"),
                "routed_experts": _tensor_hash("routes"),
            }
        )
    return records


def _write_trace(trace_dir: Path, records: list[dict]) -> None:
    trace_dir.mkdir()
    (trace_dir / "r3_trace_0.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def test_transfer_queue_contract_requires_and_matches_payload_trace(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    _write_trace(trace_dir, _tq_records() + _route_records())

    assert (
        check_trace(
            trace_dir,
            transport_contract="transfer-queue",
            require_forward_verify=True,
            require_cp_identity=True,
        )
        == 0
    )


def test_legacy_async_contract_accepts_route_trace_without_tq_events(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    _write_trace(trace_dir, _route_records())

    assert (
        check_trace(
            trace_dir,
            transport_contract="legacy-async",
            require_forward_verify=True,
            require_cp_identity=True,
        )
        == 0
    )


def test_legacy_async_contract_rejects_mixed_transfer_queue_events(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    _write_trace(trace_dir, _tq_records() + _route_records())

    assert check_trace(trace_dir, transport_contract="legacy-async") == 1


def test_forward_verification_is_required_for_each_replay_stage(
    tmp_path: Path,
) -> None:
    records = _route_records()
    records = [
        record
        for record in records
        if not (
            record["event"] == "router_replay_forward_verify"
            and record["stage"] == "train"
        )
    ]
    trace_dir = tmp_path / "trace"
    _write_trace(trace_dir, records)

    assert (
        check_trace(
            trace_dir,
            transport_contract="legacy-async",
            require_forward_verify=True,
        )
        == 1
    )


def test_cp_identity_requirement_covers_every_cp_record(tmp_path: Path) -> None:
    records = _route_records()
    records.append(
        {
            "event": "cp_routed_experts",
            "stage": "prev-logprob",
            "cp_token_identity_verified_count": None,
        }
    )
    trace_dir = tmp_path / "trace"
    _write_trace(trace_dir, records)

    assert (
        check_trace(
            trace_dir,
            transport_contract="legacy-async",
            require_cp_identity=True,
        )
        == 1
    )


def test_tq_contract_accepts_fetch_outside_producer_sample(tmp_path: Path) -> None:
    records = _tq_records() + _route_records()
    for stage in ("prev_lp", "train"):
        records.append(
            {
                "event": "tq_fetch_sample",
                "stage": stage,
                "key": "consumer-only-sample",
                "valid_length": 4,
                "input_ids": _tensor_hash("other-input"),
                "routed_experts": _tensor_hash("other-routes"),
            }
        )
    trace_dir = tmp_path / "trace"
    _write_trace(trace_dir, records)

    assert check_trace(trace_dir, transport_contract="transfer-queue") == 0


def test_tq_contract_rejects_missing_fetch_for_sampled_producer(
    tmp_path: Path,
) -> None:
    records = _tq_records() + _route_records()
    records[1]["key"] = "orphan"
    trace_dir = tmp_path / "trace"
    _write_trace(trace_dir, records)

    assert check_trace(trace_dir, transport_contract="transfer-queue") == 1
