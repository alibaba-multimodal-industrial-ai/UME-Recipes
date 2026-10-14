"""Build the one-time target embedding/index/Top-K cache for FN-risk training."""

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import transformers
import yaml
from torch.utils.data import DataLoader, Sampler
from transformers import HfArgumentParser, set_seed

from src.training.arguments import DataArguments, ModelArguments, TrainingArguments
from src.training.data.collator import MultimodalDataCollator, process_vlm_inputs_fns
from src.training.data.loader import init_concat_dataset
from src.training.model import MMEBModel, get_backbone_name, load_processor
from src.training.utils import batch_to_device, print_master


class _RankStrideSampler(Sampler):
    def __init__(self, size, rank, world_size):
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return max(0, (self.size - self.rank + self.world_size - 1) // self.world_size)


class _TargetOnlyCollator(MultimodalDataCollator):
    def __init__(self, *args, dataset_offsets, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_offsets = dataset_offsets

    def __call__(self, examples):
        corpus_ids = np.asarray(
            [
                self.dataset_offsets[example["global_dataset_name"]]
                + int(example["idx"])
                for example in examples
            ],
            dtype=np.int64,
        )
        target_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
        processed = process_fn(
            target_inputs,
            processor=self.processor,
            max_length=self.data_args.max_len,
        )
        processed["text"] = [field for example in examples for field in example["pos_text"]]
        processed["global_dataset_name"] = [
            example["global_dataset_name"]
            for example in examples
            for _ in example["pos_text"]
        ]
        if len(processed["text"]) != len(examples):
            raise RuntimeError("risk cache requires exactly one positive target per row")
        return corpus_ids, processed


def _dataset_offsets(dataset_config, dataset_lens):
    offsets = {}
    cursor = 0
    for (_, config), length in zip(dataset_config.items(), dataset_lens):
        name = f"mmeb/{config['subset_name']}"
        offsets[name] = cursor
        cursor += int(length)
    return offsets, cursor


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_published_file(source_sha, source_size, destination):
    """Wait for an OSS/FUSE write to become readable and verify its contents."""
    attempts = int(os.environ.get("CACHE_PUBLISH_VERIFY_ATTEMPTS", "4"))
    delay = float(os.environ.get("CACHE_PUBLISH_VERIFY_DELAY_SECONDS", "5"))
    if attempts <= 0 or delay < 0:
        raise ValueError("cache publish verification attempts/delay must be valid")
    observed = "missing"
    for attempt in range(1, attempts + 1):
        try:
            observed_size = destination.stat().st_size
            if observed_size != source_size:
                observed = f"size={observed_size}, expected={source_size}"
            else:
                destination_sha = _sha256_file(destination)
                if destination_sha == source_sha:
                    return True
                observed = f"sha256={destination_sha}, expected={source_sha}"
        except (FileNotFoundError, OSError) as exc:
            observed = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            print(
                "[CACHE_PUBLISH_VERIFY_RETRY] "
                f"path={destination} attempt={attempt}/{attempts} observed={observed}",
                flush=True,
            )
            time.sleep(delay)
    print(
        "[CACHE_PUBLISH_VERIFY_FAILED] "
        f"path={destination} attempts={attempts} observed={observed}",
        flush=True,
    )
    return False


def _publish_regular_file(source, destination, reuse_existing=True):
    """Stream a local artifact to OSS/FUSE and tolerate delayed read visibility."""
    source = Path(source)
    destination = Path(destination)
    source_size = source.stat().st_size
    source_sha = _sha256_file(source)

    # A previous process can have completed the multipart upload but failed its
    # immediate FUSE read-back. Reuse it when the bytes are now visible instead
    # of retransmitting a multi-gigabyte artifact.
    if reuse_existing and destination.is_file() and _verify_published_file(
        source_sha, source_size, destination
    ):
        print(f"[CACHE_PUBLISH_REUSED] path={destination}", flush=True)
        return source_sha

    with source.open("rb") as source_handle, destination.open("wb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
        target_handle.flush()
    if not _verify_published_file(source_sha, source_size, destination):
        raise RuntimeError(
            f"published artifact SHA256 mismatch: {source} -> {destination}"
        )
    return source_sha


def _build_faiss_cache(cache_dir, embeddings, manifest):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required only for this isolated online-FN-risk experiment"
        ) from exc

    num_cases, dimension = embeddings.shape
    top_k = int(manifest["top_k"])
    pq_m = int(os.environ.get("FNRW_FAISS_PQ_M", "96"))
    nbits = int(os.environ.get("FNRW_FAISS_PQ_BITS", "8"))
    if pq_m <= 0:
        raise ValueError(f"FAISS PQ m must be positive, got {pq_m}")
    if dimension % pq_m != 0:
        raise ValueError(f"embedding dimension {dimension} is not divisible by PQ m={pq_m}")
    faiss.omp_set_num_threads(int(os.environ.get("FNRW_FAISS_THREADS", "32")))
    neighbors = np.lib.format.open_memmap(
        cache_dir / "neighbors.npy",
        mode="w+",
        dtype=np.int32,
        shape=(num_cases, top_k),
    )
    add_chunk = int(os.environ.get("FNRW_FAISS_ADD_CHUNK", "16384"))
    search_chunk = int(os.environ.get("FNRW_FAISS_SEARCH_CHUNK", "4096"))
    faiss_metadata = []
    for dataset_id, partition in enumerate(manifest["dataset_partitions"]):
        offset = int(partition["offset"])
        length = int(partition["length"])
        end_offset = offset + length
        if length <= top_k:
            raise ValueError(
                f"dataset {partition['name']} has {length} cases, insufficient for "
                f"an internal Top-{top_k} excluding self"
            )
        # Keep enough vectors per inverted list that nprobe can reliably return
        # K+1 results even for the smaller MMEB sources.
        min_list_size = int(os.environ.get("FNRW_FAISS_MIN_LIST_SIZE", "64"))
        nlist = min(
            int(os.environ.get("FNRW_FAISS_NLIST", "4096")),
            max(1, length // min_list_size),
        )
        nprobe = min(int(os.environ.get("FNRW_FAISS_NPROBE", "32")), nlist)
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(
            quantizer, dimension, nlist, pq_m, nbits, faiss.METRIC_INNER_PRODUCT
        )
        index.nprobe = nprobe
        train_size = min(
            int(os.environ.get("FNRW_FAISS_TRAIN_SIZE", "100000")), length
        )
        local_train_indices = np.linspace(0, length - 1, train_size, dtype=np.int64)
        train_vectors = np.asarray(
            embeddings[offset + local_train_indices], dtype=np.float32
        )
        print(
            f"[ONLINE_FN_CACHE] dataset={partition['name']} training FAISS "
            f"on {train_size}/{length} vectors",
            flush=True,
        )
        index.train(train_vectors)
        index.set_direct_map_type(faiss.DirectMap.Hashtable)

        for start in range(offset, end_offset, add_chunk):
            end = min(start + add_chunk, end_offset)
            vectors = np.asarray(embeddings[start:end], dtype=np.float32)
            ids = np.arange(start, end, dtype=np.int64)
            index.add_with_ids(vectors, ids)
        if index.ntotal != length:
            raise RuntimeError(
                f"dataset {partition['name']} FAISS ntotal={index.ntotal}, "
                f"expected {length}"
            )

        for start in range(offset, end_offset, search_chunk):
            end = min(start + search_chunk, end_offset)
            vectors = np.asarray(embeddings[start:end], dtype=np.float32)
            _, found = index.search(vectors, top_k + 1)
            query_ids = np.arange(start, end, dtype=np.int64)
            keep = found != query_ids[:, None]
            order = np.argsort(~keep, axis=1, kind="stable")
            compact = np.take_along_axis(found, order, axis=1)[:, :top_k]
            if np.any(compact < offset) or np.any(compact >= end_offset):
                raise RuntimeError(
                    f"dataset {partition['name']} Top-K escaped its dataset partition"
                )
            neighbors[start:end] = compact.astype(np.int32)

        index_file = f"index_dataset_{dataset_id:03d}.faiss"
        faiss.write_index(index, str(cache_dir / index_file))
        partition["index_file"] = index_file
        faiss_metadata.append(
            {
                "dataset_id": dataset_id,
                "name": partition["name"],
                "index_file": index_file,
                "nlist": nlist,
                "nprobe": nprobe,
                "min_list_size": min_list_size,
                "train_size": train_size,
            }
        )
        print(
            f"[ONLINE_FN_CACHE] dataset={partition['name']} internal Top-{top_k} complete",
            flush=True,
        )
    neighbors.flush()
    manifest["faiss"] = {
        "type": "per_dataset_IndexIVFPQ",
        "metric": "inner_product",
        "pq_m": pq_m,
        "pq_bits": nbits,
        "indexes": faiss_metadata,
    }


def main():
    if transformers.__version__ != "5.9.0":
        raise RuntimeError("cache builder requires transformers==5.9.0")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    timeout = timedelta(seconds=int(training_args.ddp_timeout))
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timeout)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(int(training_args.seed))

    publish_dir = Path(os.environ["FNRW_CACHE_DIR"])
    stage_dir = Path(os.environ["FNRW_CACHE_STAGE_DIR"])
    manifest_path = publish_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete"):
            print_master(f"[ONLINE_FN_CACHE] reusing complete cache: {publish_dir}")
            dist.destroy_process_group()
            return
    setup_error = [None]
    if rank == 0:
        try:
            if publish_dir.exists() and any(publish_dir.iterdir()):
                raise RuntimeError(
                    "refusing to overwrite incomplete/nonempty publish directory: "
                    f"{publish_dir}"
                )
            if stage_dir.exists() and any(stage_dir.iterdir()):
                raise RuntimeError(
                    f"refusing to overwrite incomplete/nonempty local stage: {stage_dir}"
                )
            stage_dir.mkdir(parents=True, exist_ok=True)
            (stage_dir / "shards").mkdir(exist_ok=False)
            probe_path = stage_dir / ".mmap_probe.npy"
            probe = np.lib.format.open_memmap(
                probe_path, mode="w+", dtype=np.int32, shape=(1,)
            )
            probe[0] = 1
            probe.flush()
            del probe
            probe_path.unlink()
            print(
                f"[ONLINE_FN_CACHE] mmap staging on local filesystem: {stage_dir}",
                flush=True,
            )
        except Exception as exc:
            setup_error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(setup_error, src=0)
    if setup_error[0] is not None:
        dist.destroy_process_group()
        raise RuntimeError(f"cache staging preflight failed: {setup_error[0]}")
    dist.barrier()

    model = MMEBModel.build(model_args).to(device).eval()
    model_backbone = get_backbone_name(hf_config=model.config)
    model_args.model_backbone = model_backbone
    training_args.model_backbone = model_backbone
    processor = load_processor(model_args, data_args)
    model.model_backbone = model_backbone
    model.processor = processor

    config_bytes = Path(data_args.dataset_config).read_bytes()
    dataset_config = yaml.safe_load(config_bytes.decode("utf-8"))
    train_dataset, dataset_lens = init_concat_dataset(
        dataset_config, model_args, data_args, training_args
    )
    offsets, num_cases = _dataset_offsets(dataset_config, dataset_lens)
    collator = _TargetOnlyCollator(
        processor,
        model_args,
        data_args,
        training_args,
        dataset_offsets=offsets,
    )
    sampler = _RankStrideSampler(num_cases, rank, world_size)
    encode_batch_size = int(
        os.environ.get(
            "FNRW_ENCODE_BATCH_SIZE",
            str(training_args.per_device_train_batch_size),
        )
    )
    if encode_batch_size <= 0:
        raise ValueError("FNRW_ENCODE_BATCH_SIZE must be positive")
    print_master(
        "[ONLINE_FN_CACHE_CONFIG] "
        f"backbone={model_backbone} cases={num_cases} world_size={world_size} "
        f"encode_batch_per_rank={encode_batch_size} "
        f"pixels={data_args.resize_min_pixels}:{data_args.resize_max_pixels} "
        f"lora_enabled={bool(model_args.lora)}"
    )
    loader = DataLoader(
        train_dataset,
        batch_size=encode_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    shard_size = len(sampler)
    shard_ids = np.lib.format.open_memmap(
        stage_dir / "shards" / f"ids_rank{rank}.npy",
        mode="w+",
        dtype=np.int64,
        shape=(shard_size,),
    )
    shard_embeddings = None
    cursor = 0
    with torch.no_grad():
        for batch_index, (corpus_ids, target_inputs) in enumerate(loader):
            target_inputs = batch_to_device(target_inputs, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                reps = model(tgt=target_inputs)["tgt_reps"]
            reps_np = reps.detach().float().cpu().numpy().astype(np.float16)
            if shard_embeddings is None:
                shard_embeddings = np.lib.format.open_memmap(
                    stage_dir / "shards" / f"embeddings_rank{rank}.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(shard_size, reps_np.shape[1]),
                )
            end = cursor + len(corpus_ids)
            shard_ids[cursor:end] = corpus_ids
            shard_embeddings[cursor:end] = reps_np
            cursor = end
            if batch_index % 100 == 0:
                print(
                    f"[ONLINE_FN_CACHE] rank={rank} encoded={cursor}/{shard_size}",
                    flush=True,
                )
    if cursor != shard_size or shard_embeddings is None:
        raise RuntimeError(f"rank {rank} encoded {cursor}/{shard_size} rows")
    shard_ids.flush()
    shard_embeddings.flush()
    dimension = int(shard_embeddings.shape[1])
    dimensions = [None for _ in range(world_size)]
    dist.all_gather_object(dimensions, dimension)
    if len(set(dimensions)) != 1:
        raise RuntimeError(f"embedding dimensions differ across ranks: {dimensions}")
    # FAISS construction below is CPU-only and can take substantial time.
    # Release the encoder and the final processed GPU batch first so this stage
    # cannot be mistaken for a low-power GPU/NCCL hang.
    del target_inputs, reps
    del model, processor, loader, collator, train_dataset
    torch.cuda.empty_cache()
    print(
        f"[ONLINE_FN_CACHE_ENCODER_RELEASED] rank={rank} "
        f"encoded={cursor} dimension={dimension}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()
    if rank != 0:
        return

    embeddings = np.lib.format.open_memmap(
        stage_dir / "embeddings.npy",
        mode="w+",
        dtype=np.float16,
        shape=(num_cases, dimension),
    )
    coverage = np.zeros(num_cases, dtype=bool)
    for shard_rank in range(world_size):
        ids = np.load(stage_dir / "shards" / f"ids_rank{shard_rank}.npy", mmap_mode="r")
        reps = np.load(
            stage_dir / "shards" / f"embeddings_rank{shard_rank}.npy", mmap_mode="r"
        )
        if np.any(coverage[ids]):
            raise RuntimeError(f"duplicate corpus IDs in cache shard {shard_rank}")
        embeddings[ids] = reps
        coverage[ids] = True
    if not coverage.all():
        raise RuntimeError(f"cache is missing {int((~coverage).sum())} corpus rows")
    embeddings.flush()

    dataset_partitions = [
        {
            "dataset_id": dataset_id,
            "name": name,
            "offset": int(offsets[name]),
            "length": int(length),
        }
        for dataset_id, (name, length) in enumerate(zip(offsets.keys(), dataset_lens))
    ]
    manifest = {
        "version": 2,
        "complete": False,
        "representation": "normalized_positive_target",
        "neighbor_scope": "within_dataset",
        "num_cases": num_cases,
        "dimension": dimension,
        "top_k": int(os.environ.get("FNRW_TOP_K", "100")),
        "dataset_offsets": offsets,
        "dataset_lengths": {
            name: int(length)
            for name, length in zip(offsets.keys(), dataset_lens)
        },
        "dataset_partitions": dataset_partitions,
        "dataset_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_name": os.path.realpath(model_args.model_name),
        "model_backbone": model_backbone,
        "pooling": model_args.pooling,
        "normalize": bool(model_args.normalize),
        "temperature": float(model_args.temperature),
        "resize_use_processor": bool(data_args.resize_use_processor),
        "resize_min_pixels": int(data_args.resize_min_pixels),
        "resize_max_pixels": int(data_args.resize_max_pixels),
        "encode_batch_size_per_rank": encode_batch_size,
        "cache_model_initialization": (
            "checkpoint_plus_fresh_zero_effect_lora"
            if model_args.lora
            else "checkpoint_base_model"
        ),
        "lora_enabled_during_cache_build": bool(model_args.lora),
        "lora_r": int(model_args.lora_r),
        "lora_alpha": int(model_args.lora_alpha),
        "lora_dropout": float(model_args.lora_dropout),
        "lora_target_modules": model_args.lora_target_modules,
        "transformers_version": transformers.__version__,
        "seed": int(training_args.seed),
        "world_size": world_size,
    }
    _build_faiss_cache(stage_dir, embeddings, manifest)
    manifest["complete"] = True
    publish_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = ["neighbors.npy"] + [
        partition["index_file"] for partition in manifest["dataset_partitions"]
    ]
    manifest["artifact_sha256"] = {}
    for artifact_file in artifact_files:
        print(f"[ONLINE_FN_CACHE] publishing {artifact_file}", flush=True)
        manifest["artifact_sha256"][artifact_file] = _publish_regular_file(
            stage_dir / artifact_file, publish_dir / artifact_file
        )
    local_manifest = stage_dir / "manifest.json"
    local_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    # Publish the complete manifest last. Its presence is the commit marker.
    _publish_regular_file(local_manifest, manifest_path)
    print(
        f"[ONLINE_FN_CACHE] local staging retained for recovery/inspection: {stage_dir}",
        flush=True,
    )
    print(f"[ONLINE_FN_CACHE_COMPLETE] {json.dumps(manifest, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
