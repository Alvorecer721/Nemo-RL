# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-only dataset for the Apertus format-reward GRPO recipe.

Loads a JSONL of `{"prompt": str, "system": Optional[str]}` rows and emits the
field layout `math_data_processor` expects: `problem` (the user prompt) and
`expected_answer` (empty sentinel — the format env ignores `ground_truth`).
"""

import json
from typing import Any

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class ApertusFormatDataset(RawDataset):
    def __init__(
        self,
        data_path: str,
        split_validation_size: float = 0.0,
        seed: int = 42,
        **_: Any,
    ) -> None:
        self.task_name = "apertus_format"
        rows: list[dict[str, Any]] = []
        with open(data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                prompt = rec.get("prompt") or rec.get("input")
                if not prompt:
                    continue
                rows.append(
                    {
                        "problem": prompt,
                        "expected_answer": "",
                        "task_name": self.task_name,
                    }
                )
        self.dataset = Dataset.from_list(rows)
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)
