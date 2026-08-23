import importlib.util
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[3]
    / "infra/slurm/cscs/autoresearch/runtime_overlay/apertus70b_local_dapo.py"
)
SPEC = importlib.util.spec_from_file_location("apertus70b_local_dapo", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
RAW_DATASET_MODULE = "nemo_rl.data.datasets.raw_dataset"
saved_raw_dataset_module = sys.modules.get(RAW_DATASET_MODULE)
stub_raw_dataset_module = types.ModuleType(RAW_DATASET_MODULE)
stub_raw_dataset_module.RawDataset = type("RawDataset", (), {})
sys.modules[RAW_DATASET_MODULE] = stub_raw_dataset_module
try:
    SPEC.loader.exec_module(MODULE)
finally:
    if saved_raw_dataset_module is None:
        del sys.modules[RAW_DATASET_MODULE]
    else:
        sys.modules[RAW_DATASET_MODULE] = saved_raw_dataset_module


class _FakeDataset:
    column_names = ["messages", "task_name"]

    def __init__(self, indices: list[int] | None = None) -> None:
        self.indices = indices

    def __len__(self) -> int:
        return MODULE.DAPO_PHYSICAL_ROWS if self.indices is None else len(self.indices)

    def select(
        self, indices: list[int], *, keep_in_memory: bool = False
    ) -> "_FakeDataset":
        assert self.indices is None
        assert keep_in_memory is True
        return _FakeDataset([int(index) for index in indices])


def test_smoke_indices_are_deterministic_unique_logical_rows() -> None:
    indices = MODULE.sample_logical_indices(
        logical_rows=MODULE.DAPO_LOGICAL_ROWS,
        sample_rows=MODULE.DAPO_SMOKE_ROWS,
        seed=MODULE.DAPO_SMOKE_SEED,
    )

    assert len(indices) == MODULE.DAPO_SMOKE_ROWS
    assert len(set(indices)) == MODULE.DAPO_SMOKE_ROWS
    assert min(indices) >= 0
    assert max(indices) < MODULE.DAPO_LOGICAL_ROWS
    assert MODULE.indices_sha256(indices) == MODULE.DAPO_SMOKE_INDICES_SHA256


@pytest.mark.parametrize("sample_rows", [0, 17_918])
def test_smoke_index_sampling_rejects_invalid_size(sample_rows: int) -> None:
    with pytest.raises(ValueError, match="sample_rows must be"):
        MODULE.sample_logical_indices(
            logical_rows=MODULE.DAPO_LOGICAL_ROWS,
            sample_rows=sample_rows,
            seed=MODULE.DAPO_SMOKE_SEED,
        )


def test_smoke_dataset_selects_only_witnessed_logical_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE.Dataset, "from_file", lambda _path: _FakeDataset(), raising=True
    )

    dataset = MODULE.LocalFormattedDAPOSmokeDataset(
        "/dataset.arrow", seed=MODULE.DAPO_SMOKE_SEED
    )

    assert dataset.dataset.indices == dataset.logical_indices
    assert len(dataset.dataset) == MODULE.DAPO_SMOKE_ROWS
    assert max(dataset.dataset.indices) < MODULE.DAPO_LOGICAL_ROWS


def test_smoke_dataset_rejects_seed_drift() -> None:
    with pytest.raises(ValueError, match="requires seed=42"):
        MODULE.LocalFormattedDAPOSmokeDataset("/dataset.arrow", seed=43)
