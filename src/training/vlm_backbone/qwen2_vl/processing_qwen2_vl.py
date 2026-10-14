# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
#
# This processor intentionally keeps the Qwen2-VL behavior used by B3
# (transformers 4.48-era code). Transformers 5.x changed ProcessorMixin's
# generic kwarg routing, so the training path below routes kwargs explicitly.

from typing import List, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
)


VideoInput = ImageInput


class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class Qwen2VLProcessor(ProcessorMixin):
    """B3-compatible Qwen2-VL processor."""

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    # Transformers 5.x validates components with isinstance() against classes
    # from the transformers package. B3's vendored implementations are sibling
    # classes rather than subclasses of the packaged Qwen2-VL classes, so use
    # their common HF base classes for validation. The concrete components are
    # still explicitly constructed in load_processor().
    image_processor_class = "BaseImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def check_argument_for_proper_class(self, attribute_name, arg):
        """Validate vendored Qwen2 components without optional tokenizer imports.

        Transformers 5.x resolves the string class names above through its
        global lazy module. In an MDL image that also contains an old optional
        ``mistral_common``, that lookup imports the unrelated Mistral tokenizer
        and fails before Qwen2 initialization. Both vendored components already
        inherit the stable HF base classes, so validate those directly.
        """
        expected_class = {
            "image_processor": BaseImageProcessor,
            "tokenizer": PreTrainedTokenizerBase,
        }.get(attribute_name)
        if expected_class is None:
            return super().check_argument_for_proper_class(attribute_name, arg)
        if not isinstance(arg, expected_class):
            raise TypeError(
                f"Qwen2VLProcessor expected {attribute_name} to inherit "
                f"{expected_class.__name__}, got {type(arg).__name__}"
            )

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        # Keyword arguments work on both the B3-era and current ProcessorMixin
        # and avoid depending on positional component ordering.
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )

    @staticmethod
    def _merge_b3_call_kwargs(kwargs):
        """Route the kwargs used by B3 without 5.x ProcessorMixin defaults."""
        kwargs = dict(kwargs)
        text_kwargs = dict(kwargs.pop("text_kwargs", {}) or {})
        images_kwargs = dict(kwargs.pop("images_kwargs", {}) or {})
        videos_kwargs = dict(kwargs.pop("videos_kwargs", {}) or {})
        common_kwargs = dict(kwargs.pop("common_kwargs", {}) or {})

        text_keys = {
            "add_special_tokens",
            "padding",
            "truncation",
            "max_length",
            "stride",
            "is_split_into_words",
            "pad_to_multiple_of",
            "return_attention_mask",
            "return_token_type_ids",
            "return_overflowing_tokens",
            "return_special_tokens_mask",
            "return_offsets_mapping",
            "return_length",
            "verbose",
        }
        vision_keys = {
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "data_format",
            "input_data_format",
        }

        for key, value in common_kwargs.items():
            text_kwargs[key] = value
            images_kwargs[key] = value
            videos_kwargs[key] = value

        for key, value in kwargs.items():
            if key == "return_tensors":
                text_kwargs[key] = value
                images_kwargs[key] = value
                videos_kwargs[key] = value
            elif key in text_keys:
                text_kwargs[key] = value
            elif key in vision_keys:
                images_kwargs[key] = value
                videos_kwargs[key] = value
            else:
                raise TypeError(
                    f"Unsupported Qwen2-VL processor argument in B3 "
                    f"compatibility mode: {key}"
                )

        text_kwargs.setdefault("padding", False)
        return {
            "text_kwargs": text_kwargs,
            "images_kwargs": images_kwargs,
            "videos_kwargs": videos_kwargs,
        }

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_b3_call_kwargs(kwargs)

        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                videos=None,
                **output_kwargs["images_kwargs"],
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(
                images=None,
                videos=videos,
                **output_kwargs["videos_kwargs"],
            )
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        # Keep B3's placeholder expansion exactly: one image placeholder is
        # expanded to the number of merged visual tokens produced by the image
        # processor.
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    token_count = int(image_grid_thw[index].prod() // merge_length)
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * token_count,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    token_count = int(video_grid_thw[index].prod() // merge_length)
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * token_count,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["Qwen2VLProcessor"]
