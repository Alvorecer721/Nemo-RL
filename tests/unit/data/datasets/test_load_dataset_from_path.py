# Copyright (c) 2026, the Apertus project.
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
"""save_to_disk directories must load via load_from_disk, never load_dataset.

load_dataset globs the loose .arrow files inside a save_to_disk directory —
including stale cache-*.arrow left by .map(), whose schema need not match —
instead of raising, so the exception-driven load_from_disk fallback never
fires. The loader detects the format by its marker files (dataset_dict.json /
state.json) up front.
"""

from datasets import Dataset, DatasetDict

from nemo_rl.data.datasets.utils import load_dataset_from_path


def _toy_dataset_dict(tmp_path):
    dd = DatasetDict(
        {
            "train": Dataset.from_dict({"a": [1, 2, 3]}),
            "validation": Dataset.from_dict({"a": [4, 5]}),
        }
    )
    root = tmp_path / "dd"
    dd.save_to_disk(str(root))
    return root


def test_save_to_disk_datasetdict_split_selection(tmp_path):
    root = _toy_dataset_dict(tmp_path)
    ds = load_dataset_from_path(str(root), None, "validation")
    assert len(ds) == 2
    assert ds.column_names == ["a"]


def test_save_to_disk_survives_stale_map_cache(tmp_path):
    root = _toy_dataset_dict(tmp_path)
    # A stale .map() cache with an unrelated schema, as left behind by real
    # preprocessing runs; load_dataset would try to cast it and die.
    (root / "validation" / "cache-deadbeef.arrow").write_bytes(b"")
    dd = load_dataset_from_path(str(root), None, "validation")
    assert len(dd) == 2


def test_save_to_disk_single_dataset(tmp_path):
    single = tmp_path / "single"
    Dataset.from_dict({"a": [1, 2, 3, 4]}).save_to_disk(str(single))
    ds = load_dataset_from_path(str(single), None, None)
    assert len(ds) == 4
