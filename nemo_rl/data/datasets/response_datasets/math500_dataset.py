# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""MATH-500 as a GRPO in-loop validation response dataset.

Wraps the existing ``math_500_test`` eval loader so GRPO can roll it out and report
``val:accuracy`` on text math each ``val_period`` (graded by the standard ``math``
env). Emits ``problem`` + ``expected_answer`` + ``task_name='math'`` for
``math_data_processor``.
"""

from nemo_rl.data.datasets.eval_datasets.math import MathDataset
from nemo_rl.data.datasets.raw_dataset import RawDataset


class Math500Dataset(RawDataset):
    """MATH-500 (math_500_test split) for GRPO in-loop validation."""

    def __init__(self, **kwargs) -> None:
        self.task_name = "math"
        rekeyed = MathDataset(variant="math_500_test").rekeyed_ds
        self.dataset = rekeyed.add_column("task_name", [self.task_name] * len(rekeyed))
