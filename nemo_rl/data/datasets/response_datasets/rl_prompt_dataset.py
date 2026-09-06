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

"""Train-side loader for the omni ``rl_prompt`` store.

The ``vision_tokenization`` producer (``rl_prompt`` task) emits, per split, a
Megatron MMIDIDX ``{split}.bin``/``.idx`` of prompt-only, generation-ready
documents (text + inline Emu3.5 image tokens, ending at ``<|assistant_start|>``)
plus ``index_{split}.parquet`` carrying ``answer``/``answer_variants`` per doc, in
``.bin`` document order. The image is already discrete tokens spliced into the doc,
so the prompt is never re-tokenized: ``mmididx_grpo_data_processor`` reads each doc
by index and passes its token-ids straight through.
"""

import os

import pyarrow.parquet as pq
from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class RLPromptDataset(RawDataset):
    """GRPO prompt dataset backed by an omni ``rl_prompt`` MMIDIDX store.

    Args:
        data_path: Store root containing ``{split}.bin``/``.idx`` and
            ``index_{split}.parquet``.
        split: Split to load (default ``"train"``).
        task_name: Routes each datum to its environment/processor (default
            ``"rl_prompt"``).
        max_prompt_len: If set, drop docs whose prompt exceeds this many tokens
            (frees output budget; e.g. 2048 leaves ~6k for generation at an 8k cap).
        repeat: Number of times to repeat the dataset, default 1.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        task_name: str = "rl_prompt",
        max_prompt_len: int | None = None,
        repeat: int = 1,
        **kwargs,
    ) -> None:
        # Lazy import so this module stays importable without megatron; the
        # .bin reader is only needed when the dataset is actually constructed.
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        self.task_name = task_name
        self.val_dataset = None
        store_prefix = os.path.join(data_path, split)

        index = pq.read_table(
            os.path.join(data_path, f"index_{split}.parquet"),
            columns=["answer", "answer_variants"],
        )
        idx_ds = IndexedDataset(store_prefix)
        n_docs = len(idx_ds)
        if n_docs == 0:
            raise ValueError(f"{store_prefix}: empty store (0 documents)")
        if n_docs != index.num_rows:
            raise ValueError(
                f"{store_prefix}: {n_docs} .bin docs vs {index.num_rows} index rows "
                "-- inconsistent store (re-run the producer binidx)"
            )

        answers = index.column("answer").to_pylist()
        variants = index.column("answer_variants").to_pylist()

        keep: list[int] | range = range(n_docs)
        if max_prompt_len is not None:
            lengths = idx_ds.index.sequence_lengths
            keep = [i for i in range(n_docs) if lengths[i] <= max_prompt_len]
            if not keep:
                raise ValueError(
                    f"{store_prefix}: no documents with prompt_len <= {max_prompt_len}"
                )

        self.dataset = Dataset.from_dict(
            {
                "store_prefix": [store_prefix] * len(keep),
                "doc_index": list(keep),
                "ground_truth": [str(answers[i]) for i in keep],
                "answer_variants": [
                    [str(v) for v in (variants[i] or [])] for i in keep
                ],
                "task_name": [task_name] * len(keep),
            }
        )

        if repeat > 1:
            self.dataset = self.dataset.repeat(repeat)
