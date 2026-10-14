from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os

from torch.jit import isinstance as torch_isinstance
from src.training.model import PHI3V, VLM_IMAGE_TOKENS
from src.training.utils import print_master, print_rank
from torch.utils.data import Dataset
import torch
import logging
from dataclasses import dataclass
import torch.distributed as dist
from typing import Iterator, List, Tuple, Union
from functools import wraps
from datasets import Features, Value, Sequence

logger = logging.getLogger(__name__)

# ==================== Constants ====================
MULTIMODAL_FEATURES = Features(**{
    "query_text": Value(dtype='string'),
    "query_image": {
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    },
    "pos_text": Value(dtype='string'),
    "pos_image": {
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    },
    "neg_text": Value(dtype='string'),
    "neg_image": {
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    },
    "global_dataset_name": Value(dtype='string'),
})

RESOLUTION_MAPPING = {
    "high": (1344, 1344),
    "mid": (672, 672),
    "low": (128, 128),
}


# ==================== AutoPairDataset ====================
class AutoPairDataset:
    registry = {}

    @classmethod
    def instantiate(cls, dataset_parser, *args, **kwargs):
        try:
            return cls.registry[dataset_parser](*args, **kwargs)
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoPairDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper


def add_metainfo_hook(f):
    """
    A post-processing wrapper function that add meta information (e.g. data_type, dataset_name, loss_type) into batches
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # go through data pipeline customized to each dataset
        batch_data = f(*args, **kwargs)
        # append common metadata
        batch_size = len(batch_data['query_text'])
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        batch_data['global_dataset_name'] = global_dataset_name
        return batch_data

    return wrapper


# ==================== Data preparation ====================
def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image


@add_metainfo_hook
def data_prepare(example, *args, **kwargs):
    image_dir = kwargs['image_dir']
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']

    qry_text = example['qry']
    qry_image_path = example['qry_image_path']
    pos_text = example['pos_text']
    pos_image_path = example['pos_image_path']
    neg_text_list = example.get('neg_text', [])
    neg_image_path_list = example.get('neg_image_path', [])

    neg_text_list = [] if ((not neg_text_list) or type(neg_text_list) == str) else neg_text_list
    neg_image_path_list = [] if ((not neg_image_path_list) or type(neg_image_path_list) == str) else neg_image_path_list

    if model_backbone != PHI3V:
        qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        neg_text_list = [neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]) if neg_text else '' for neg_text in neg_text_list]

    # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
    qry_image = {"bytes": [None], "paths": [os.path.join(image_dir, qry_image_path) if qry_image_path else ''], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
    pos_image = {"bytes": [None], "paths": [os.path.join(image_dir, pos_image_path) if pos_image_path else ''], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}

    try:
        neg_image_path_list = [{"bytes": [None], "paths": [os.path.join(image_dir, neg_image_path) if neg_image_path else ''], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for neg_image_path in neg_image_path_list]
    except:
        import ipdb; ipdb.set_trace()

    if not qry_text:
        print('something went wrong')

    if(kwargs["pos_only"]):
        pos_text_list = [pos_text]
        pos_image_list = [pos_image]
    else:
        pos_text_list = [pos_text] + neg_text_list
        pos_image_list = [pos_image] + neg_image_path_list

    return {"query_text": [qry_text], "query_image": [qry_image],
            "pos_text": pos_text_list, "pos_image": pos_image_list,
            "neg_text": [], "neg_image": []}


DATASET_PARSER_NAME = "mmeb"


@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_mmeb_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    dataset_split = kwargs.get("dataset_split", "original")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))

    # B3's MMEB config calls this split "original", while its loader reads the
    # physical test-*.parquet files unconditionally. Preserve that convention
    # explicitly instead of silently switching the training corpus to
    # train-*.parquet. Other explicitly named splits keep their normal meaning.
    parquet_split = "test" if dataset_split == "original" else dataset_split

    # Use glob to match parquet files (handles sharded datasets like OVEN:
    # test-00000-of-00002.parquet).
    import glob
    parquet_pattern = os.path.join(dataset_name, subset_name, f"{parquet_split}-*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))
    if not parquet_files:
        raise FileNotFoundError(
            "No B3 MMEB parquet files found: "
            f"logical_split={dataset_split!r}, physical_split={parquet_split!r}, "
            f"pattern={parquet_pattern!r}"
        )
    print(
        "[B3_MMEB_DATA]",
        f"subset={subset_name}",
        f"logical_split={dataset_split}",
        f"physical_split={parquet_split}",
        f"files={parquet_files}",
        flush=True,
    )
    dataset = load_dataset(
        f"parquet",
        data_files=parquet_files,
        split=f"{data_args.dataset_split}",
    )

    # VLM2Vec applies num_sample_per_subset before converting the source into
    # its iterable/interleaved training stream.  Keep this behavior isolated
    # to the ordinary-in-batch control so the verified B3 corpus is untouched.
    if (
        getattr(data_args, "negative_strategy", "b3") == "vlm2vec_inbatch"
        and num_sample_per_subset is not None
        and int(num_sample_per_subset) < dataset.num_rows
    ):
        original_rows = dataset.num_rows
        dataset = dataset.select(range(int(num_sample_per_subset)))
        print(
            "[VLM2VEC_INBATCH_TRUNCATE] "
            f"subset={subset_name} original_rows={original_rows} "
            f"selected_rows={dataset.num_rows}",
            flush=True,
        )

    # VLM2Vec's batched MMEB mapper drops rows whose query side or positive
    # side has neither text nor image. A map-style Dataset cannot drop rows
    # from this repository's per-example mapper, so reproduce the same filter
    # explicitly after the public recipe's per-source prefix selection.
    if getattr(data_args, "negative_strategy", "b3") == "vlm2vec_inbatch":
        rows_before_filter = dataset.num_rows
        dataset = dataset.filter(
            lambda example: bool(example.get("qry") or example.get("qry_image_path"))
            and bool(example.get("pos_text") or example.get("pos_image_path")),
            load_from_cache_file=False,
            cache_file_name=None,
            keep_in_memory=True,
        )
        print(
            "[VLM2VEC_INBATCH_VALID_PAIR_FILTER] "
            f"subset={subset_name} before={rows_before_filter} "
            f"after={dataset.num_rows} removed={rows_before_filter - dataset.num_rows}",
            flush=True,
        )

        if model_args.model_backbone == "qwen2_vl" and dataset.num_rows > 0:
            # A lightweight corpus sentinel for comparing the offline parquet
            # mirror with an official VLM2Vec run. Hash raw query/positive
            # fields before prompt replacement or image loading so processor
            # and filesystem differences cannot hide a wrong training split.
            import hashlib
            import json

            sentinel_indices = sorted(
                {
                    0,
                    dataset.num_rows // 2,
                    dataset.num_rows - 1,
                }
            )
            sentinel_fields = (
                "qry",
                "qry_image_path",
                "pos_text",
                "pos_image_path",
            )
            sentinel_rows = [
                {
                    field: dataset[index].get(field)
                    for field in sentinel_fields
                }
                for index in sentinel_indices
            ]
            sentinel_json = json.dumps(
                sentinel_rows,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
            print_master(
                "[VLM2VEC_INBATCH_QWEN2_SOURCE_FINGERPRINT] "
                f"subset={subset_name} rows={dataset.num_rows} "
                f"indices={sentinel_indices} "
                f"sha256={hashlib.sha256(sentinel_json.encode('utf-8')).hexdigest()}"
            )

    column_names = dataset.column_names
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    # Prompt routing must use the canonical YAML dataset key rather than the
    # physical B3 subset name, whose HNPS/Metis suffix is not part of MMEB's
    # task identity. Fall back to the subset only for legacy direct callers.
    prompt_dataset_name = kwargs.get('global_dataset_name') or subset_name
    kwargs['global_dataset_name'] = prompt_dataset_name
    kwargs['pos_only'] = data_args.pos_only
    remove_columns = ['qry', 'qry_image_path', 'pos_text', 'pos_image_path']
    if 'neg_image_path' in column_names:
        remove_columns.append('neg_text')
        remove_columns.append('neg_image_path')
    dataset = dataset.map(lambda x, idx: {**data_prepare(x, **kwargs), "idx": idx}, with_indices=True, batched=False, remove_columns=remove_columns, load_from_cache_file=False, cache_file_name=None, keep_in_memory=True,)
    print_master(
        f"Loaded {DATASET_PARSER_NAME}/{subset_name} dataset with {num_rows} "
        f"samples; prompt_dataset_name={prompt_dataset_name}"
    )

    return dataset


# ==================== EvalDataset ====================
class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field, mod_instruction=None):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone
        # Load dataset from local path directly (no network needed)
        local_dataset_path = os.path.join(self.data_args.dataset_name, subset)
        if os.path.exists(local_dataset_path):
            print_rank(f"Loading dataset from local path: {local_dataset_path}")
            self.eval_data = load_dataset(
                local_dataset_path,
                split=self.data_args.dataset_split,
            )
        else:
            print_rank(f"Local path not found: {local_dataset_path}, falling back to HuggingFace Hub")
            self.eval_data = load_dataset(
                self.data_args.dataset_name,
                subset,
                split=self.data_args.dataset_split,
            )
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        print_rank(f"DEBUG EvalDataset: paired_data len={len(self.paired_data)}, text_field={text_field}")

        if(("tgt" in text_field) and (mod_instruction is not None) and self.data_args.tgt_prefix_mod):
            print("Using TGT mod", mod_instruction, flush=True)
            self.paired_dataset = datasets.Dataset.from_dict({
                                "text": [mod_instruction + pair["text"] for pair in self.paired_data],
                                "img_path": [pair["img_path"] for pair in self.paired_data]
                                })
            print(">>>>>>>>>>>>>inside tgt_mod_txt", flush=True)
        else:
            print("Not using TGT mod", mod_instruction, flush=True)
            self.paired_dataset = datasets.Dataset.from_dict({
                                "text": [pair["text"] for pair in self.paired_data],
                                "img_path": [pair["img_path"] for pair in self.paired_data]
                                })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.backbone != PHI3V:
            text = text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.backbone])
        return text, self._get_image(img_path)

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        import builtins
        first_row = self.eval_data[0]
        print_rank(f"DEBUG get_paired_data START: text_field={text_field}, img_path_field={img_path_field}")
        print_rank(f"  eval_data size={len(self.eval_data)}")
        text_sample = first_row[text_field]
        img_sample = first_row[img_path_field]
        print_rank(f"  first_row[{text_field}] type={type(text_sample).__name__}, is_list={builtins.isinstance(text_sample, list)}")
        if builtins.isinstance(text_sample, list):
            print_rank(f"    list len={len(text_sample)}, first_item_type={type(text_sample[0]).__name__}, first_item={repr(text_sample[0])[:80]}")
        else:
            print_rank(f"    value={repr(text_sample)[:80]}")
        print_rank(f"  first_row[{img_path_field}] type={type(img_sample).__name__}, is_list={builtins.isinstance(img_sample, list)}")
        if builtins.isinstance(img_sample, list):
            print_rank(f"    list len={len(img_sample)}, first_item_type={type(img_sample[0]).__name__}, first_item={repr(img_sample[0])[:80]}")
        else:
            print_rank(f"    value={repr(img_sample)[:80]}")

        unique_pair = set()
        pair_count_before = 0
        for row_idx, row in enumerate(self.eval_data):
            text_val = row[text_field]
            img_val = row[img_path_field]
            text_is_list = builtins.isinstance(text_val, list)
            img_is_list = builtins.isinstance(img_val, list)

            if len(unique_pair) == 0:
                print_rank(f"DEBUG get_paired_data: text_field={text_field}, img_path_field={img_path_field}")
                print_rank(f"  text_val type={type(text_val).__name__}, text_is_list={text_is_list}, text_val_repr={repr(text_val)[:80] if builtins.isinstance(text_val, str) else (f'list(len={len(text_val)})' if text_is_list else repr(text_val)[:80])}")
                print_rank(f"  img_val type={type(img_val).__name__}, img_is_list={img_is_list}, img_val_repr={repr(img_val)[:80] if builtins.isinstance(img_val, str) else (f'list(len={len(img_val)})' if img_is_list else repr(img_val)[:80])}")

            if text_is_list and img_is_list:
                if len(text_val) != len(img_val):
                    print_rank(f"WARNING: len mismatch in {text_field}/{img_path_field}: "
                               f"{len(text_val)} vs {len(img_val)}, skipping row")
                    continue
                for text, img_path in zip(text_val, img_val):
                    if text or img_path:
                        unique_pair.add((text, img_path))
            elif text_is_list and not img_is_list:
                for text in text_val:
                    if text or img_val:
                        unique_pair.add((text, img_val))
            elif not text_is_list and img_is_list:
                if text_val:
                    for img_path in img_val:
                        if img_path:
                            unique_pair.add((text_val, img_path))
                else:
                    for img_path in img_val:
                        if img_path:
                            unique_pair.add((text_val, img_path))
            else:
                if text_val or img_val:
                    unique_pair.add((text_val, img_val))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        paired_data = sorted(paired_data, key=lambda k: (k['img_path'], k['text']))
        return paired_data


# ==================== CustomRandomSampler ====================
@dataclass
class CustomRandomSampler(torch.utils.data.sampler.RandomSampler):
    """
    Sampler used when training on multiple datasets to ensure each
    batch only contains samples from one dataset for the majority of cases.
    """
    total_batch_size: int = 8
    ds_lens: List[int] = None
    _num_samples: int = None
    data_source: Dataset = None
    replacement: bool = False
    ordered_datasets: bool = False
    same_datasets: bool = False
    random_datasets: bool = False
    num_repeats: int = 1
    chunk_size: int = None

    def __len__(self):
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        if not hasattr(self, "generator") or self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # We have multiple datasets each with a different number of samples
        # e.g. [100, 150, 50]
        # We would like to sample from them such that as much as possible each batch
        # only has samples from the same dataset.

        if(self.ordered_datasets):
            print(">>>>>>>>>>>>>>>>>>>>>>Asking for ordered datasets", flush=True)
            ds_indices = []
            for n in self.ds_lens:
                usable_n = n - (n % self.chunk_size)  # truncate to make divisible
                if usable_n < n:
                    logger.warning(f"Dataset length {n} is not divisible by chunk_size {self.chunk_size}, truncating to {usable_n}")
                chunks = torch.arange(usable_n).view(-1, self.chunk_size)  # shape (usable_n // chunk_size, chunk_size)
                perm = torch.randperm(chunks.size(0))  # randomize chunks
                shuffled_chunks = chunks[perm]
                ds_indices.append(shuffled_chunks.flatten().tolist())

            ds_indices = [[i + sum(self.ds_lens[:j]) for i in ds_indices[j]] for j in range(len(self.ds_lens))]
            ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]
        elif(self.same_datasets):
            ds_indices = [torch.randperm(n, generator=generator).tolist() for n in self.ds_lens]
            ds_indices = [[i + sum(self.ds_lens[:j]) for i in ds_indices[j]] for j in range(len(self.ds_lens))]
            ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]
        elif(self.random_datasets):
            total_len = sum(self.ds_lens)
            shuffled_indices = torch.randperm(total_len, generator=generator)
            ds_batches = [list(torch.split(shuffled_indices, self.total_batch_size))]
        else:
            assert False, "none of odibn,sdibn,rdibn found"
        incomplete_indices = []

        for b in ds_batches:
            if len(b[-1]) < self.total_batch_size:
                incomplete_indices.append(b.pop())

        if False:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=generator).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(torch.tensor(incomplete_indices), self.total_batch_size))
            if len(mixed_batches[-1]) < self.total_batch_size:
                mixed_batches.pop()
            # Merge all batches to look like [...tensor([259, 273, 284, 289]), tensor([262, 280, 295, 258]), ...]
            ds_batches = sum(ds_batches, []) + mixed_batches
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches) - len(mixed_batches)} single-dataset batches & {len(mixed_batches)} mixed dataset batches.")
        else:
            ds_batches = sum(ds_batches, [])
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches)} single-dataset batches.")

        all_batches = []  # Start empty

        for _ in range(self.num_repeats):  # Iterate K times to include the first epoch as well
            epoch_order = torch.randperm(len(ds_batches), generator=generator).tolist()
            shuffled_batches = [ds_batches[i] for i in epoch_order]  # Keep batch structure
            all_batches.append(shuffled_batches)

        # Flatten across all epochs
        ds_batches = [batch for epoch_batches in all_batches for batch in epoch_batches]

        # Convert to list of indices
        ds_batches = [int(i) for i in torch.cat(ds_batches).tolist()]

        print(">>>>>>>>>>>>>>>>>>>>ds batches", ds_batches[:128], self.ordered_datasets, self.total_batch_size, flush=True)

        # Yield the indices
        yield from ds_batches


@dataclass
class VLM2VecInBatchSampler(torch.utils.data.Sampler[int]):
    """Map-style equivalent of VLM2Vec's weighted source-block interleaving.

    The public VLM2Vec image-only recipe uses ``interleave_batch_size=64``.
    VLM2Vec interprets that as 64 examples *per device*, hence 512 consecutive
    examples from one randomly selected source with 8 processes.  Sources are
    consumed sequentially and restarted when exhausted.  Reproducing those
    source blocks matters because it determines which examples become mutual
    in-batch negatives.
    """

    total_batch_size: int
    ds_lens: List[int]
    probabilities: List[float]
    source_block_size: int
    num_steps: int
    data_source: Dataset
    seed: int = 42
    match_vlm2vec_rank_sharding: bool = False

    def __post_init__(self):
        if not self.ds_lens or any(length <= 0 for length in self.ds_lens):
            raise ValueError(f"Invalid VLM2Vec dataset lengths: {self.ds_lens}")
        if len(self.probabilities) != len(self.ds_lens):
            raise ValueError(
                "VLM2Vec probabilities/datasets mismatch: "
                f"{len(self.probabilities)} != {len(self.ds_lens)}"
            )
        if any(weight < 0 for weight in self.probabilities):
            raise ValueError(f"Negative VLM2Vec dataset weight: {self.probabilities}")
        weight_sum = float(sum(self.probabilities))
        if weight_sum <= 0:
            raise ValueError("VLM2Vec dataset weights must have a positive sum")
        self.probabilities = [float(weight) / weight_sum for weight in self.probabilities]
        self._num_samples = int(self.total_batch_size) * int(self.num_steps)
        self.source_block_size = int(self.source_block_size)
        if self.source_block_size <= 0:
            raise ValueError(
                f"VLM2Vec source_block_size must be positive: {self.source_block_size}"
            )
        if self.total_batch_size % self.source_block_size != 0:
            raise ValueError(
                "VLM2Vec source block must divide the global batch: "
                f"{self.total_batch_size} % {self.source_block_size} != 0"
            )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.total_batch_size % world_size != 0:
            raise ValueError(
                "VLM2Vec global batch must be divisible by world size: "
                f"{self.total_batch_size} % {world_size} != 0"
            )

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        # Match VLM2Vec's RandomlyCyclingMultiSourcesExamplesIterable, which
        # uses numpy.default_rng(seed) for the weighted source stream.
        import numpy as np

        rng = np.random.default_rng(self.seed)
        num_source_blocks = int(np.ceil(self._num_samples / self.source_block_size))
        source_blocks = rng.choice(
            len(self.ds_lens),
            size=num_source_blocks,
            p=self.probabilities,
        )
        source_ids = np.repeat(source_blocks, self.source_block_size)[: self._num_samples]
        offsets = np.cumsum([0] + self.ds_lens[:-1], dtype=np.int64)
        cursors = np.zeros(len(self.ds_lens), dtype=np.int64)
        indices = []
        for source_id in source_ids:
            source_id = int(source_id)
            local_index = int(cursors[source_id] % self.ds_lens[source_id])
            indices.append(int(offsets[source_id]) + local_index)
            cursors[source_id] += 1

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        layout_message = ""
        if self.match_vlm2vec_rank_sharding and world_size > 1:
            # VLM2Vec defines this source-block size per device, constructs the
            # corresponding global interleaved stream, and splits that stream
            # across ranks. HF Trainer instead shards this map-style DataLoader
            # by whole local batches. Transpose each global batch into rank-major
            # order so BatchSamplerShard reconstructs the intended per-rank
            # 64-example source blocks.
            reordered_indices = []
            reordered_source_ids = []
            for batch_start in range(0, self._num_samples, self.total_batch_size):
                batch_end = min(batch_start + self.total_batch_size, self._num_samples)
                batch_indices = np.asarray(indices[batch_start:batch_end], dtype=np.int64)
                batch_sources = source_ids[batch_start:batch_end]
                for process_index in range(world_size):
                    reordered_indices.extend(batch_indices[process_index::world_size].tolist())
                    reordered_source_ids.extend(batch_sources[process_index::world_size].tolist())
            indices = reordered_indices
            source_ids = np.asarray(reordered_source_ids, dtype=np.int64)
            layout_message = "layout=vlm2vec_example_shard_rank_major "

        import hashlib

        local_batch_size = self.total_batch_size // world_size
        first_batch_sources = source_ids[: self.total_batch_size]
        first_batch_indices = np.asarray(indices[: self.total_batch_size], dtype=np.int64)
        local_start = rank * local_batch_size
        local_end = local_start + local_batch_size
        first_local_sources = first_batch_sources[local_start:local_end]
        first_local_indices = first_batch_indices[local_start:local_end]
        counts = np.bincount(first_batch_sources, minlength=len(self.ds_lens)).tolist()
        local_counts = np.bincount(
            first_local_sources, minlength=len(self.ds_lens)
        ).tolist()
        global_sha = hashlib.sha256(first_batch_indices.tobytes()).hexdigest()[:20]
        local_sha = hashlib.sha256(first_local_indices.tobytes()).hexdigest()[:20]
        print(
            "[VLM2VEC_INBATCH_SAMPLER] "
            f"rank={rank}/{world_size} seed={self.seed} "
            f"global_batch={self.total_batch_size} local_batch={local_batch_size} "
            f"{layout_message}"
            f"source_block_size={self.source_block_size} "
            f"blocks_per_global_batch={self.total_batch_size // self.source_block_size} "
            f"num_steps={self.num_steps} probabilities={self.probabilities} "
            f"first_global_batch_sources={source_blocks[:self.total_batch_size // self.source_block_size].tolist()} "
            f"first_global_batch_source_counts={counts} "
            f"first_local_batch_source_counts={local_counts} "
            f"global_unique={np.unique(first_batch_indices).size}/{self.total_batch_size} "
            f"local_unique={np.unique(first_local_indices).size}/{local_batch_size} "
            f"global_indices_sha256={global_sha} local_indices_sha256={local_sha} "
            f"first_local_indices={first_local_indices[:16].tolist()}",
            flush=True,
        )
        yield from indices
