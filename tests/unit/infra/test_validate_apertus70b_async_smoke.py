import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[3]
    / "infra/slurm/cscs/autoresearch/validate_apertus70b_async_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_apertus70b_async_smoke", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _valid_scalars() -> dict[str, list[tuple[float, int, float]]]:
    values = {
        "valid_tokens": [100.0, 110.0, 120.0],
        "grad_norm": [0.5, 0.4, 0.3],
        "loss": [-0.1, -0.2, -0.3],
        "reward_min": [0.0, 0.0, 0.0],
        "reward_max": [1.0, 1.0, 1.0],
        "advantage_min": [-1.0, -0.5, -0.25],
        "advantage_max": [1.0, 0.5, 0.25],
        "trajectory_age": [0.0, 1.0, 1.0],
        "step_time_s": [10.0, 9.0, 8.0],
    }
    return {
        MODULE.REQUIRED_TAGS[name]: [
            (float(step), step, value) for step, value in enumerate(series, start=1)
        ]
        for name, series in values.items()
    }


def _set_value(
    scalars: dict[str, list[tuple[float, int, float]]],
    metric: str,
    step: int,
    value: float,
) -> None:
    tag = MODULE.REQUIRED_TAGS[metric]
    scalars[tag] = [
        (wall_time, event_step, value if event_step == step else event_value)
        for wall_time, event_step, event_value in scalars[tag]
    ]


def test_smoke_metrics_accepts_emitted_reward_min_max_tags() -> None:
    metrics = MODULE.validate_smoke_metrics(_valid_scalars(), steps=3)

    assert metrics["reward_min"] == {1: 0.0, 2: 0.0, 3: 0.0}
    assert metrics["reward_max"] == {1: 1.0, 2: 1.0, 3: 1.0}


def test_smoke_metrics_rejects_zero_reward_range_at_any_step() -> None:
    scalars = _valid_scalars()
    _set_value(scalars, "reward_max", 2, 0.0)

    with pytest.raises(AssertionError, match="nonzero reward range"):
        MODULE.validate_smoke_metrics(scalars, steps=3)


def test_smoke_metrics_rejects_obsolete_reward_stddev_only() -> None:
    scalars = _valid_scalars()
    scalars.pop(MODULE.REQUIRED_TAGS["reward_min"])
    scalars.pop(MODULE.REQUIRED_TAGS["reward_max"])
    scalars["train/total_reward/stddev"] = [
        (float(step), step, 1.0) for step in range(1, 4)
    ]

    with pytest.raises(AssertionError, match="min_total_reward missing steps"):
        MODULE.validate_smoke_metrics(scalars, steps=3)
