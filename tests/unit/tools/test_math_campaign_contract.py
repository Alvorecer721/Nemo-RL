"""Coverage, seed and prompt-isolation checks for the overnight evaluator."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3] / "tools/model_diagnostics/math_campaign.py"
)
spec = importlib.util.spec_from_file_location("math_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_request_shards_cover_each_repeat_once() -> None:
    rows = [
        dict(
            id=f"p{i}",
            dataset="aime24",
            question="q",
            answer="1",
            repeats=32,
            temperature=0.6,
            top_p=0.95,
        )
        for i in range(30)
    ]
    shards = [campaign.requests_for_shard(rows, n, 8, 42) for n in range(8)]
    keys = [r["request_id"] for shard in shards for r in shard]
    assert len(keys) == len(set(keys)) == 960
    assert len({r["sampling_seed"] for shard in shards for r in shard}) == 960
    assert len({r["shard"] for shard in shards for r in shard}) == 8


def test_prompt_length_cannot_silently_reduce_generated_budget() -> None:
    campaign.validate_token_budget(4096, 12288, 16384)
    with pytest.raises(ValueError, match="budget"):
        campaign.validate_token_budget(4097, 12288, 16384)


def test_duplicate_questions_fail_manifest_validation() -> None:
    row = dict(
        id="p1",
        dataset="math500",
        question="q",
        answer="1",
        repeats=1,
        temperature=0,
        top_p=1,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        campaign.requests_for_shard([row, row], 0, 8, 42)
