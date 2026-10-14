# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

"""
Custom Qwen2-VL implementation forked from transformers 4.48.1.

Modifications from official transformers:
1. modeling_qwen2_vl.py: Mixed-batch forward (handles list-style pixel_values with None entries)
2. modeling_qwen2_vl.py: Disabled Flash Attention padding_side='right' check
3. image_processing_qwen2_vl.py: smart_resize handles extreme images (square resize instead of error)
4. image_processing_qwen2_vl.py: Truncated image error handling (try-except skip)
"""

from .configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from .image_processing_qwen2_vl import Qwen2VLImageProcessor
from .processing_qwen2_vl import Qwen2VLProcessor
from .tokenization_qwen2_fast import Qwen2TokenizerFast
from .modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
)

__all__ = [
    "Qwen2VLConfig",
    "Qwen2VLVisionConfig",
    "Qwen2VLImageProcessor",
    "Qwen2VLProcessor",
    "Qwen2TokenizerFast",
    "Qwen2VLForConditionalGeneration",
    "Qwen2VLModel",
    "Qwen2VLPreTrainedModel",
]
