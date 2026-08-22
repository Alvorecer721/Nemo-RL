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
"""Runtime safeguards for the Apertus NeMo-RL integration.

The fork's datasets, processors, environments, recipes, and vLLM compatibility
changes live in their normal NeMo-RL locations. This package contains the
launch-time guard that verifies the active checkout and Megatron-Bridge expose
the Apertus-specific weight-conversion hooks required for train-to-generation
parity.
"""
