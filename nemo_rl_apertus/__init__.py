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
"""Additive NeMo-RL adapter for the Apertus views+media preference data format.

This package feeds multimodal preference data (text-live views + pretokenized
image-token blocks spliced at marker id 131079) into NeMo-RL's DPO/RM training
through the public ``dpo.setup()`` seam. It contains zero modifications to
upstream NeMo-RL files; everything plugs in via ``AllTaskProcessedDataset``
with a custom processor callable.

Spec: ``docs/design-docs/apertus-multimodal-preference-data.md``.
Producer (writes the on-disk format): ``vision_tokenization`` ``alignment`` mode
in the ``benchmark-image-tokenzier`` repo.
"""
