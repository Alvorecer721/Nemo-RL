"""Experiment-only adapter for the immutable formatted DAPO Arrow cache."""

import hashlib
import json

import numpy as np
from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

DAPO_PHYSICAL_ROWS = 1_791_700
DAPO_LOGICAL_ROWS = 17_917
DAPO_REPETITION_FACTOR = 100
DAPO_SMOKE_ROWS = 64
DAPO_SMOKE_SEED = 42
DAPO_SMOKE_INDICES_SHA256 = (
    "fc7df7406ff17109033f6d7572d0a255cf3dcb0f6fb6fabdb45aa7c0c9daa309"
)


def sample_logical_indices(
    *, logical_rows: int, sample_rows: int, seed: int
) -> list[int]:
    """Return a deterministic sample without touching repeated physical rows."""
    if not 0 < sample_rows <= logical_rows:
        raise ValueError(
            f"sample_rows must be in [1, {logical_rows}], got {sample_rows}"
        )
    return np.random.default_rng(seed).permutation(logical_rows)[:sample_rows].tolist()


def indices_sha256(indices: list[int]) -> str:
    encoded = json.dumps(indices, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LocalFormattedDAPOMathDataset(RawDataset):
    """Read a preformatted DAPO dataset directly, without Hub resolution."""

    def __init__(self, data_path: str, **kwargs) -> None:
        del kwargs
        self.task_name = "DAPOMath17K"
        self.dataset = Dataset.from_file(data_path)
        required_columns = {"messages", "task_name"}
        missing = required_columns - set(self.dataset.column_names)
        if missing:
            raise ValueError(
                f"Formatted DAPO Arrow dataset is missing columns: {sorted(missing)}"
            )


class LocalFormattedDAPOSmokeDataset(LocalFormattedDAPOMathDataset):
    """Use one deterministic, non-repeated real-DAPO slice for certification."""

    def __init__(
        self,
        data_path: str,
        seed: int = DAPO_SMOKE_SEED,
        split_validation_size: float = 0.0,
        **kwargs,
    ) -> None:
        if split_validation_size:
            raise ValueError("The fixed Apertus smoke dataset has no validation split")
        if seed != DAPO_SMOKE_SEED:
            raise ValueError(
                f"The fixed Apertus smoke dataset requires seed={DAPO_SMOKE_SEED}"
            )
        super().__init__(data_path, **kwargs)
        if len(self.dataset) != DAPO_PHYSICAL_ROWS:
            raise ValueError(
                "The Apertus smoke requires the exact repeated DAPO Arrow artifact: "
                f"expected {DAPO_PHYSICAL_ROWS} rows, got {len(self.dataset)}"
            )
        if DAPO_PHYSICAL_ROWS != DAPO_LOGICAL_ROWS * DAPO_REPETITION_FACTOR:
            raise AssertionError("Invalid DAPO repetition provenance")
        self.logical_indices = sample_logical_indices(
            logical_rows=DAPO_LOGICAL_ROWS,
            sample_rows=DAPO_SMOKE_ROWS,
            seed=seed,
        )
        if indices_sha256(self.logical_indices) != DAPO_SMOKE_INDICES_SHA256:
            raise AssertionError("Apertus smoke logical-index witness changed")
        self.dataset = self.dataset.select(self.logical_indices, keep_in_memory=True)
