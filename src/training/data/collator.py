from itertools import repeat
from typing import Optional

from torch.jit import isinstance as torch_isinstance

import logging
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.training.arguments import DataArguments, ModelArguments, TrainingArguments
import torch
from PIL import Image
import io
from src.training.utils import print_rank, print_master
from src.training.model import QWEN2_VL, QWEN3_5_VL, VLM_IMAGE_TOKENS, VLM_VIDEO_TOKENS
from src.debug_fingerprint import log_fingerprint

logger = logging.getLogger(__name__)
_QWEN3_5_PROMPT_LOGGED = set()


# ==================== GradCache helper functions ====================
def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = []
    for k in keys:
        if isinstance(arg_val[k], torch.Tensor):
            chunked_tensor = arg_val[k].split(chunk_size, dim=0)
        else:
            chunked_tensor = [arg_val[k][i: i + chunk_size] for i in list(range(0, len(arg_val[k]), chunk_size))]
        chunked_tensors.append(chunked_tensor)
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
    chunked_inputs = [{arg_key: c} for c in chunked_arg_val]

    return chunked_inputs


def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps.
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    else:
        return x["qry_reps"]


# ==================== Process functions ====================
def _append_qwen2_embedding_token(input_id, processor):
    if not getattr(processor, "_qwen2_embedding_token_enabled", False):
        return input_id
    token_id = getattr(processor, "_qwen2_embedding_token_id", None)
    if not isinstance(token_id, int):
        raise RuntimeError("Qwen2 processor has no valid embedding token ID")
    if token_id in input_id:
        raise ValueError(
            "Raw Qwen2 input already contains the reserved embedding token; "
            "the collator must append exactly one copy"
        )
    return [*input_id, token_id]


def Qwen2_VL_process_fn(model_inputs: dict, processor, max_length=None):
    from transformers.image_utils import ChannelDimension
    import PIL

    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]
    processor_max_length = max_length
    if getattr(processor, "_qwen2_embedding_token_enabled", False) and max_length is not None:
        if max_length < 2:
            raise ValueError(
                "Qwen2 max_length must leave room for the embedding token"
            )
        processor_max_length = max_length - 1

    # 1. iterate each pair and process, since processors do not support processing for mixed batch (contains data w/ and w/o visual inputs)
    for text, visual_input in zip(texts, visual_inputs):
        if not visual_input or (type(visual_input) == list and any(i is None for i in visual_input)):
            # if text inputs only (all images must be valid)
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=processor_max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(_append_qwen2_embedding_token(input_id, processor))
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            # Ensure visual_input is a list for consistent handling
            if isinstance(visual_input, PIL.Image.Image):
                visual_input = [visual_input]

            # If text is empty but visual_input exists, add image token automatically
            if not text.strip() and visual_input:
                text = vlm_image_token

            try:
                if vlm_image_token in text:
                    for iid, image in enumerate(visual_input):
                        # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                        if image.size[0] < 28 or image.size[1] < 28:
                            image = image.resize((56, 56))
                            visual_input[iid] = image
                    inputs = processor(text=[text], images=visual_input, return_tensors="np", max_length=processor_max_length, truncation=(processor_max_length is not None), input_data_format=ChannelDimension.LAST)
                elif vlm_video_token in text:
                    # check text/video data validity
                    if not isinstance(visual_input, (list, tuple)) or len(visual_input) == 0:
                        print(f"Warning: invalid video data for text={text[:100]}..., treating as text-only")
                        inputs = processor(text=[text], images=None, return_tensors="np", max_length=processor_max_length, truncation=True)
                    else:
                        inputs = processor(text=[text], videos=[visual_input], return_tensors="np", max_length=processor_max_length, truncation=(processor_max_length is not None), input_data_format=ChannelDimension.LAST)
                else:
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")
            except Exception as e:
                for i in visual_input:
                    print(i.filename)
                raise e
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                input_id = [input_id]
            input_ids.append(_append_qwen2_embedding_token(input_id, processor))
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'texts': texts,
        'images': visual_inputs,
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw
    # Transformers 5.9's native Qwen2-VL model requires this modality map for
    # M-RoPE. The B3-vendored 4.48 processor predates the field.
    image_token_id = processor.tokenizer.convert_tokens_to_ids(vlm_image_token)
    video_token_id = processor.tokenizer.convert_tokens_to_ids(vlm_video_token)
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    mm_token_type_ids.masked_fill_(input_ids == image_token_id, 1)
    mm_token_type_ids.masked_fill_(input_ids == video_token_id, 2)
    inputs['mm_token_type_ids'] = mm_token_type_ids

    return inputs


def Qwen3_5_VL_process_fn(
    model_inputs: dict,
    processor,
    max_length=None,
    prompt_profile="legacy",
    prompt_dataset_names=None,
    prompt_side=None,
):
    """Build replay-safe per-example inputs for native Qwen3.5-VL.

    Qwen3.5 uses Qwen3VLProcessor, whose visual tensors are concatenated over
    patches rather than over samples.  Process each sample independently and
    keep its complete visual tensor in a list; GradCache may then slice the
    batch without losing the association between a sample, its grid and its
    placeholder tokens.  MMEBModel concatenates those lists only at the native
    model boundary.
    """
    import unicodedata
    import PIL
    from qwen_vl_utils import process_vision_info

    POS_MOD_PREFIXES = [
        "Represent the class label: ",
        "Represent the answer: ",
        "Represent the image caption: ",
    ]
    DEFAULT_INSTRUCTION = "Represent the user's input."
    VLM_IMAGE_TOKEN = "<|image_pad|>"
    VLM_VIDEO_TOKEN = "<|video_pad|>"

    input_ids = []
    mm_token_type_ids_list = []
    pixel_values = []
    image_grid_thw = []
    pixel_values_videos = []
    video_grid_thw = []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    if prompt_profile != "legacy":
        from src.training.qwen3_5_prompt import validate_profile

        validate_profile(prompt_profile)
        if prompt_side not in {"query", "target"}:
            raise ValueError(
                "Qwen3.5 non-legacy prompts require prompt_side=query/target"
            )
        if prompt_dataset_names is None or len(prompt_dataset_names) != len(texts):
            raise ValueError(
                "Qwen3.5 non-legacy prompts require one dataset name per input: "
                f"names={None if prompt_dataset_names is None else len(prompt_dataset_names)} "
                f"inputs={len(texts)}"
            )

    image_patch_size = getattr(processor.image_processor, "patch_size", None)
    image_merge_size = getattr(processor.image_processor, "merge_size", None)
    if image_patch_size != 16 or image_merge_size != 2:
        raise ValueError(
            "Qwen3.5 processor/model mismatch: expected image "
            f"patch_size=16 and merge_size=2, got patch_size={image_patch_size} "
            f"and merge_size={image_merge_size}"
        )
    min_pixels = getattr(processor, "_b3_resize_min_pixels", None)
    max_pixels = getattr(processor, "_b3_resize_max_pixels", None)
    if (
        not isinstance(min_pixels, int)
        or not isinstance(max_pixels, int)
        or min_pixels <= 0
        or max_pixels < min_pixels
    ):
        raise ValueError(
            "Qwen3.5 processor is missing valid B3 resize bounds: "
            f"min_pixels={min_pixels}, max_pixels={max_pixels}"
        )

    for input_index, (text, visual_input) in enumerate(zip(texts, visual_inputs)):
        # Detect and strip POS_MOD prefix from text
        instruction = DEFAULT_INSTRUCTION
        clean_text = (
            text.replace(VLM_IMAGE_TOKEN, '').replace(VLM_VIDEO_TOKEN, '').strip()
            if text
            else ''
        )
        if prompt_profile == "legacy":
            for prefix in POS_MOD_PREFIXES:
                if clean_text.startswith(prefix):
                    instruction = prefix
                    clean_text = clean_text[len(prefix):].strip()
                    break
        else:
            from src.training.qwen3_5_prompt import (
                clean_legacy_payload,
                instruction_for,
            )

            dataset_name = prompt_dataset_names[input_index]
            instruction = instruction_for(dataset_name, prompt_side)
            clean_text = clean_legacy_payload(
                clean_text, dataset_name, prompt_side
            )
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            log_key = (rank, prompt_side)
            if rank == 0 and log_key not in _QWEN3_5_PROMPT_LOGGED:
                import json
                from src.training.qwen3_5_prompt import task_for_dataset

                print(
                    "[QWEN3_5_PROMPT_V2_PACK] "
                    + json.dumps(
                        {
                            "rank": rank,
                            "dataset": dataset_name,
                            "task": task_for_dataset(dataset_name),
                            "side": prompt_side,
                            "system": instruction,
                            "raw_text": (text or "")[:300],
                            "clean_user_text": clean_text[:300],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                _QWEN3_5_PROMPT_LOGGED.add(log_key)

        # Ensure instruction ends with punctuation
        instruction = instruction.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
            instruction = instruction + '.'

        # B3 image data contains either one image or no image.  Do not silently
        # discard an extra image: doing so would keep training alive with the
        # wrong query/target semantics.
        if isinstance(visual_input, PIL.Image.Image):
            images_for_sample = [visual_input]
        elif isinstance(visual_input, (list, tuple)):
            images_for_sample = [image for image in visual_input if image is not None]
        elif visual_input is None:
            images_for_sample = []
        else:
            raise TypeError(
                "Unsupported Qwen3.5 visual input type: "
                f"{type(visual_input).__name__}"
            )

        is_video = VLM_VIDEO_TOKENS[QWEN3_5_VL] in (text or "")
        if not is_video and len(images_for_sample) > 1:
            raise ValueError(
                "B3 Qwen3.5 image training expects at most one image per "
                f"sample, got {len(images_for_sample)}"
            )

        content = []
        if is_video and images_for_sample:
            video_item = {
                "type": "video",
                "video": images_for_sample,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
            content.append(video_item)
        elif images_for_sample:
            content.append(
                {
                    "type": "image",
                    "image": images_for_sample[0],
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            )
        if clean_text:
            content.append({"type": "text", "text": clean_text})
        if not content:
            content.append({"type": "text", "text": "NULL"})

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]
        chat_text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversation,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception as error:
            raise RuntimeError(
                "Qwen3.5 vision preprocessing failed; refusing to replace a "
                "multimodal B3 sample with text-only input"
            ) from error

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        proc_kwargs = {
            'text': [chat_text],
            'images': images,
            'videos': videos,
            'video_metadata': video_metadata,
            'return_tensors': 'np',
            'padding': False,
            'truncation': max_length is not None,
            'return_mm_token_type_ids': True,
            # qwen-vl-utils has already resized with patch_size=16 and the
            # configured B3 pixel bounds. The official Qwen3-VL path disables
            # a second processor resize to avoid changing pixels twice.
            'do_resize': False,
            **(video_kwargs or {}),
        }
        if max_length is not None:
            proc_kwargs['max_length'] = max_length
        inputs = processor(**proc_kwargs)

        sample_ids = inputs['input_ids'].squeeze().tolist()
        if isinstance(sample_ids, int):
            sample_ids = [sample_ids]
        sample_mm_types = inputs.get('mm_token_type_ids')
        if sample_mm_types is None:
            raise RuntimeError(
                "Qwen3.5 processor did not return mm_token_type_ids"
            )
        sample_mm_types = sample_mm_types.squeeze().tolist()
        if isinstance(sample_mm_types, int):
            sample_mm_types = [sample_mm_types]
        if len(sample_ids) != len(sample_mm_types):
            raise ValueError(
                "Qwen3.5 input_ids/mm_token_type_ids length mismatch: "
                f"{len(sample_ids)} != {len(sample_mm_types)}"
            )

        sample_pixels = inputs.get('pixel_values')
        sample_image_grid = inputs.get('image_grid_thw')
        sample_video_pixels = inputs.get('pixel_values_videos')
        sample_video_grid = inputs.get('video_grid_thw')
        if (sample_pixels is None) != (sample_image_grid is None):
            raise ValueError("Qwen3.5 image pixels/grid must be present together")
        if (sample_video_pixels is None) != (sample_video_grid is None):
            raise ValueError("Qwen3.5 video pixels/grid must be present together")

        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            VLM_IMAGE_TOKENS[QWEN3_5_VL]
        )
        video_token_id = processor.tokenizer.convert_tokens_to_ids(
            VLM_VIDEO_TOKENS[QWEN3_5_VL]
        )
        image_token_count = sum(token == image_token_id for token in sample_ids)
        video_token_count = sum(token == video_token_id for token in sample_ids)
        typed_image_count = sum(token_type == 1 for token_type in sample_mm_types)
        typed_video_count = sum(token_type == 2 for token_type in sample_mm_types)
        if image_token_count != typed_image_count or video_token_count != typed_video_count:
            raise ValueError(
                "Qwen3.5 placeholder/mm_token_type_ids mismatch: "
                f"image={image_token_count}/{typed_image_count}, "
                f"video={video_token_count}/{typed_video_count}"
            )
        if sample_image_grid is not None:
            expected = int(
                (torch.as_tensor(sample_image_grid).prod(dim=-1) // image_merge_size**2)
                .sum()
                .item()
            )
            if image_token_count != expected:
                raise ValueError(
                    "Qwen3.5 image placeholder/feature mismatch: "
                    f"tokens={image_token_count}, expected_features={expected}"
                )
        if sample_video_grid is not None:
            expected = int(
                (torch.as_tensor(sample_video_grid).prod(dim=-1) // image_merge_size**2)
                .sum()
                .item()
            )
            if video_token_count != expected:
                raise ValueError(
                    "Qwen3.5 video placeholder/feature mismatch: "
                    f"tokens={video_token_count}, expected_features={expected}"
                )

        input_ids.append(sample_ids)
        mm_token_type_ids_list.append(sample_mm_types)
        pixel_values.append(sample_pixels)
        image_grid_thw.append(sample_image_grid)
        pixel_values_videos.append(sample_video_pixels)
        video_grid_thw.append(sample_video_grid)

    batch_encoding = processor.tokenizer.pad(
        {'input_ids': input_ids}, return_tensors='pt'
    )
    input_ids_tensor = batch_encoding['input_ids'].long()
    attention_mask = batch_encoding['attention_mask'].long()
    seq_len = input_ids_tensor.shape[1]
    padding_side = processor.tokenizer.padding_side
    padded_mm_ttids = []
    for sample_mm_types in mm_token_type_ids_list:
        pad_len = seq_len - len(sample_mm_types)
        if pad_len < 0:
            raise ValueError("Qwen3.5 modality ids exceed padded input length")
        padding = [0] * pad_len
        padded_mm_ttids.append(
            padding + sample_mm_types
            if padding_side == 'left'
            else sample_mm_types + padding
        )

    result = {
        'input_ids': input_ids_tensor,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': visual_inputs,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'pixel_values_videos': pixel_values_videos,
        'video_grid_thw': video_grid_thw,
        'mm_token_type_ids': torch.tensor(padded_mm_ttids, dtype=torch.long),
    }
    return result


process_vlm_inputs_fns = {
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN3_5_VL: Qwen3_5_VL_process_fn,
}


@dataclass
class EvalCollator:
    """B3 evaluation collator using the same processor path as training."""

    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin
    prompt_dataset_name: Optional[str] = None
    prompt_side: Optional[str] = None

    def __call__(self, examples):
        model_inputs = {
            "text": [example[0] for example in examples],
            "images": [example[1] for example in examples],
        }
        process_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        process_kwargs = {
            "processor": self.processor,
            "max_length": self.data_args.max_len,
        }
        if self.model_args.model_backbone == QWEN3_5_VL:
            process_kwargs.update(
                {
                    "prompt_profile": self.model_args.qwen3_5_prompt_profile,
                    "prompt_dataset_names": [self.prompt_dataset_name] * len(examples),
                    "prompt_side": self.prompt_side,
                }
            )
        inputs = process_fn(model_inputs, **process_kwargs)
        # Keep B3's auxiliary fields. MMEBModel filters them before invoking
        # the Transformers-native backbone, while they remain useful for
        # debugging and cached-evaluation bookkeeping.
        inputs["image_paths"] = [
            image.filename if image is not None else None
            for image in model_inputs["images"]
        ]
        inputs["text"] = model_inputs["text"]
        inputs["images"] = model_inputs["images"]
        return inputs


# ==================== Data Collator ====================
@dataclass
class MultimodalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    batch_size: Optional[int] = None  # used to verify if a batch has invalid data

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            # @ruimeng filter invalid data examples here may lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                text, visual_input = '  ', None
                texts.append(text)
                visual_inputs.append(visual_input)
                assert False, "This should not happen"
            else:
                text_list, raw_images_list = example[text_keyname], example[image_keyname]
                for text, raw_images in zip(text_list, raw_images_list):
                    if type(raw_images) == dict:
                        visual_input = []
                        assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                        num_images = len(raw_images['resolutions'])
                        for image_idx in range(num_images):
                            bytes = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                            path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                            image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                            if bytes is None and ((path is None) or (not path)):
                                image = None
                            elif bytes is not None:
                                # vidore, image inputs are already bytes
                                image = Image.open(io.BytesIO(bytes))
                            elif (path is not None) and (path):
                                # mmeb/video datasets, lazy image loading and processing
                                image = Image.open(path)
                            else:
                                print_rank(f"\n{'=' * 50}\nsomething went wrong with a data point from {example['global_dataset_name']}, neither bytes or path is given. \n\t\tquery_text: {example['query_text']}")
                            if not self.data_args.resize_use_processor and image is not None and image_resolution:
                                image = image.resize(image_resolution)
                            visual_input.append(image)
                    else:
                        visual_input = None
                    texts.append(text)
                    visual_inputs.append(visual_input)
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs

    def __call__(self, examples):
        """
        :param examples: 'query_text', 'query_image_path', 'pos_text', 'pos_image_path', 'neg_text', 'neg_image_path'
        """
        if (
            getattr(self.data_args, "negative_strategy", "b3") == "vlm2vec_inbatch"
            and not getattr(self, "_vlm2vec_inbatch_logged", False)
        ):
            import collections
            import hashlib
            import json

            query_counts = [len(example["query_text"]) for example in examples]
            target_counts = [len(example["pos_text"]) for example in examples]
            if any(count != 1 for count in query_counts) or any(
                count != 1 for count in target_counts
            ):
                raise RuntimeError(
                    "VLM2Vec ordinary in-batch collator requires one query and "
                    "one positive target per dataset row, but got "
                    f"query_counts={query_counts} target_counts={target_counts}"
                )
            example_indices = [int(example["idx"]) for example in examples]
            dataset_names = [example["global_dataset_name"] for example in examples]
            index_bytes = ",".join(map(str, example_indices)).encode("utf-8")
            pair_records = [
                {
                    "dataset": example["global_dataset_name"],
                    "idx": int(example["idx"]),
                    "query_text": example["query_text"],
                    "query_image_paths": example["query_image"][0].get("paths", []),
                    "positive_text": example["pos_text"],
                    "positive_image_paths": example["pos_image"][0].get("paths", []),
                }
                for example in examples
            ]
            pair_bytes = json.dumps(
                pair_records,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            query_image_rows = sum(
                any(bool(path) for path in example["query_image"][0].get("paths", []))
                for example in examples
            )
            target_image_rows = sum(
                any(bool(path) for path in example["pos_image"][0].get("paths", []))
                for example in examples
            )
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            print(
                "[VLM2VEC_INBATCH_COLLATOR_OK] "
                f"rank={rank} local_rows={len(examples)} "
                f"unique_rows={len(set(example_indices))}/{len(example_indices)} "
                f"target_per_query=1 "
                f"dataset_counts={dict(collections.Counter(dataset_names))} "
                f"indices_sha256={hashlib.sha256(index_bytes).hexdigest()[:20]} "
                f"pairs_sha256={hashlib.sha256(pair_bytes).hexdigest()[:20]} "
                f"query_image_rows={query_image_rows}/{len(examples)} "
                f"target_image_rows={target_image_rows}/{len(examples)} "
                f"first_indices={example_indices[:16]}",
                flush=True,
            )
            self._vlm2vec_inbatch_logged = True

        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        bs = len(qry_inputs['text'])
        assert bs > 0, 'An empty batch'
        # pad batch to batch_size to avoid hanging in distributed training
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expect batch size {self.batch_size}, but got batch size of {bs}")
        process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
        process_kwargs = {
            "processor": self.processor,
            "max_length": self.data_args.max_len,
        }
        qry_kwargs = dict(process_kwargs)
        pos_kwargs = dict(process_kwargs)
        if self.training_args.model_backbone == QWEN3_5_VL:
            qry_dataset_names = [
                example["global_dataset_name"]
                for example in examples
                for _ in example["query_text"]
            ]
            pos_dataset_names = [
                example["global_dataset_name"]
                for example in examples
                for _ in example["pos_text"]
            ]
            qry_kwargs.update(
                {
                    "prompt_profile": self.model_args.qwen3_5_prompt_profile,
                    "prompt_dataset_names": qry_dataset_names,
                    "prompt_side": "query",
                }
            )
            pos_kwargs.update(
                {
                    "prompt_profile": self.model_args.qwen3_5_prompt_profile,
                    "prompt_dataset_names": pos_dataset_names,
                    "prompt_side": "target",
                }
            )
        processed_qry_inputs = process_fn(qry_inputs, **qry_kwargs)
        processed_pos_inputs = process_fn(pos_inputs, **pos_kwargs)
        log_fingerprint(
            "collator_output",
            example_indices=[example["idx"] for example in examples],
            dataset_names=[example["global_dataset_name"] for example in examples],
            raw_query_text=qry_inputs["text"],
            raw_target_text=pos_inputs["text"],
            query=processed_qry_inputs,
            target=processed_pos_inputs,
        )
        if('input_ids' in processed_qry_inputs):
            print_rank(f"\t\tQry collator: processed_qry_inputs['input_ids'].shape={processed_qry_inputs['input_ids'].shape}\t\tPos collator: processed_pos_inputs['input_ids'].shape={processed_pos_inputs['input_ids'].shape}")
        processed_qry_inputs['text'] = [f for e in examples for f in e['query_text']]
        processed_pos_inputs['text'] = [f for e in examples for f in e['pos_text']]
        processed_qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        processed_pos_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples for f in e['pos_text']]

        print_rank(f"::::::::::::::::::::::{[example['idx'] for example in examples]}")
        return processed_qry_inputs, processed_pos_inputs
