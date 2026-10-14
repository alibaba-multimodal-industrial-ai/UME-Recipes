"""Build initial query/positive caches and fixed diverse seeds for hard batches."""

import hashlib
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import transformers
import yaml
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed

from build_online_false_negative_cache import (
    _RankStrideSampler,
    _dataset_offsets,
    _publish_regular_file,
)
from src.training.arguments import DataArguments, ModelArguments, TrainingArguments
from src.training.data.collator import MultimodalDataCollator
from src.training.data.loader import init_concat_dataset
from src.training.model import MMEBModel, get_backbone_name, load_processor
from src.training.dynamic_hard_batch_v2_core import (
    teacher_adapter_fingerprint as compute_teacher_adapter_fingerprint,
)
from src.training.utils import batch_to_device, print_master


class _PairCollator(MultimodalDataCollator):
    def __init__(self, *args, dataset_offsets, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_offsets = dataset_offsets

    def __call__(self, examples):
        ids = np.asarray([
            self.dataset_offsets[e["global_dataset_name"]] + int(e["idx"])
            for e in examples
        ], dtype=np.int64)
        queries, positives = super().__call__(examples)
        return ids, queries, positives


class _EmbeddingOnlyLMHead(torch.nn.Module):
    """Avoid materializing unused vocabulary logits during cache encoding.

    Qwen2-VL computes the Transformer hidden states before invoking lm_head.
    MMEBModel consumes hidden_states[-1] and never reads logits, so returning a
    one-channel view preserves the embedding exactly while avoiding a
    batch*sequence*vocabulary allocation.
    """

    def forward(self, hidden_states):
        return hidden_states[..., :1]


def main():
    if transformers.__version__ != "5.9.0":
        raise RuntimeError("dynamic hard-batch cache builder requires transformers==5.9.0")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=int(training_args.ddp_timeout)))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(int(training_args.seed))
    publish_dir = Path(os.environ["DHBG_CACHE_DIR"])
    stage_dir = Path(os.environ["DHBG_CACHE_STAGE_DIR"])
    if (publish_dir / "manifest.json").is_file():
        manifest = json.loads((publish_dir / "manifest.json").read_text())
        if manifest.get("complete"):
            print_master(f"[DYNAMIC_HARD_BATCH_CACHE] reusing {publish_dir}")
            dist.destroy_process_group()
            return
    error = [None]
    if rank == 0:
        try:
            if publish_dir.exists() and any(publish_dir.iterdir()):
                raise RuntimeError(f"refusing to overwrite nonempty cache: {publish_dir}")
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            (stage_dir / "shards").mkdir(parents=True)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    dist.barrier()

    model = MMEBModel.build(model_args).to(device).eval()
    embedding_encoder = (
        model.encoder.get_base_model()
        if hasattr(model.encoder, "get_base_model")
        else model.encoder
    )
    model_backbone = getattr(model, "model_backbone", None)
    if model_backbone is None:
        model_backbone = get_backbone_name(hf_config=model.config)
    if hasattr(embedding_encoder, "lm_head"):
        embedding_encoder.lm_head = _EmbeddingOnlyLMHead()
        print(
            f"[DYNAMIC_HARD_BATCH_CACHE] rank={rank} disabled unused vocabulary logits",
            flush=True,
        )
    elif model_backbone == "qwen3_5_vl":
        # The verified Qwen3.5 path intentionally loads Qwen3_5Model rather
        # than Qwen3_5ForConditionalGeneration, so there is no vocabulary
        # projection to disable. MMEBModel consumes last_hidden_state directly.
        print(
            f"[DYNAMIC_HARD_BATCH_CACHE] rank={rank} "
            "Qwen3_5Model has no vocabulary LM head (embedding-only path)",
            flush=True,
        )
    else:
        raise RuntimeError(
            "Embedding cache builder could not locate the causal LM head for "
            f"backbone={model_backbone!r}"
        )
    model_args.model_backbone = model_backbone
    training_args.model_backbone = model_args.model_backbone
    processor = load_processor(model_args, data_args)
    model.model_backbone, model.processor = model_args.model_backbone, processor
    config_bytes = Path(data_args.dataset_config).read_bytes()
    dataset_config = yaml.safe_load(config_bytes.decode())
    dataset, lengths = init_concat_dataset(dataset_config, model_args, data_args, training_args)
    offsets, total = _dataset_offsets(dataset_config, lengths)
    encode_batch_size = int(os.environ.get("DHBG_ENCODE_BATCH_SIZE", "16"))
    if encode_batch_size <= 0:
        raise ValueError(f"DHBG_ENCODE_BATCH_SIZE must be positive: {encode_batch_size}")
    loader = DataLoader(
        dataset,
        batch_size=encode_batch_size,
        sampler=_RankStrideSampler(total, rank, world),
        collate_fn=_PairCollator(processor, model_args, data_args, training_args, dataset_offsets=offsets),
        num_workers=0,
        pin_memory=True,
    )
    shard_size = len(loader.sampler)
    ids_out = np.lib.format.open_memmap(stage_dir / "shards" / f"ids_{rank}.npy", mode="w+", dtype=np.int64, shape=(shard_size,))
    q_out = p_out = None
    cursor = 0
    with torch.no_grad():
        for batch_no, (ids, queries, positives) in enumerate(loader):
            queries, positives = batch_to_device(queries, device), batch_to_device(positives, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = model(qry=queries)["qry_reps"]
                p = model(tgt=positives)["tgt_reps"]
            q, p = q.float().cpu().numpy().astype(np.float16), p.float().cpu().numpy().astype(np.float16)
            if q_out is None:
                q_out = np.lib.format.open_memmap(stage_dir / "shards" / f"q_{rank}.npy", mode="w+", dtype=np.float16, shape=(shard_size, q.shape[1]))
                p_out = np.lib.format.open_memmap(stage_dir / "shards" / f"p_{rank}.npy", mode="w+", dtype=np.float16, shape=(shard_size, p.shape[1]))
            end = cursor + len(ids)
            ids_out[cursor:end], q_out[cursor:end], p_out[cursor:end] = ids, q, p
            cursor = end
            if batch_no % 100 == 0:
                print(f"[DYNAMIC_HARD_BATCH_CACHE] rank={rank} encoded={cursor}/{shard_size}", flush=True)
    if cursor != shard_size or q_out is None:
        raise RuntimeError(f"rank {rank} encoded {cursor}/{shard_size}")
    ids_out.flush(); q_out.flush(); p_out.flush()
    dimensions = [None] * world
    dist.all_gather_object(dimensions, int(q_out.shape[1]))
    if len(set(dimensions)) != 1:
        raise RuntimeError(f"embedding dimensions differ: {dimensions}")
    dist.barrier()
    dist.destroy_process_group()
    if rank:
        return

    dimension = dimensions[0]
    q_all = np.lib.format.open_memmap(stage_dir / "query_embeddings.npy", mode="w+", dtype=np.float16, shape=(total, dimension))
    p_all = np.lib.format.open_memmap(stage_dir / "positive_embeddings.npy", mode="w+", dtype=np.float16, shape=(total, dimension))
    covered = np.zeros(total, dtype=bool)
    for shard in range(world):
        ids = np.load(stage_dir / "shards" / f"ids_{shard}.npy", mmap_mode="r")
        if covered[ids].any():
            raise RuntimeError("duplicate cache IDs")
        q_all[ids] = np.load(stage_dir / "shards" / f"q_{shard}.npy", mmap_mode="r")
        p_all[ids] = np.load(stage_dir / "shards" / f"p_{shard}.npy", mmap_mode="r")
        covered[ids] = True
    if not covered.all():
        raise RuntimeError("incomplete embedding cache")
    q_all.flush(); p_all.flush()

    rng = np.random.default_rng(int(training_args.seed))
    schedule = rng.choice(len(lengths), size=int(training_args.max_steps), p=np.full(len(lengths), 1 / len(lengths)))
    seed_counts = np.bincount(schedule, minlength=len(lengths))
    defer_seeds_to_v2 = os.environ.get("DHBG_DEFER_SEEDS_TO_V2", "0") == "1"
    # Seed selection is the one expensive global operation. Keep one dataset's
    # pair vectors on GPU and update its min-max envelope with GEMV; no NxN
    # similarity matrix is materialized.
    def farthest_cuda(local_q, local_p, count, initial):
        vectors = 0.5 * (
            torch.from_numpy(np.asarray(local_q)).to(device)
            + torch.from_numpy(np.asarray(local_p)).to(device)
        )
        count = min(int(count), len(vectors))
        selected = []
        chosen = torch.zeros(len(vectors), dtype=torch.bool, device=device)
        max_similarity = torch.full((len(vectors),), -torch.inf, dtype=torch.float32, device=device)
        current = int(initial)
        with torch.no_grad():
            for _ in range(count):
                selected.append(current)
                chosen[current] = True
                similarity = (vectors @ vectors[current]).float()
                max_similarity = torch.maximum(max_similarity, similarity)
                max_similarity[chosen] = torch.inf
                if len(selected) < count:
                    current = int(torch.argmin(max_similarity).item())
        del vectors, chosen, max_similarity
        torch.cuda.empty_cache()
        return np.asarray(selected, dtype=np.int64)

    partitions, seed_lists = [], []
    cursor = 0
    for dataset_id, ((_, config), length, count) in enumerate(zip(dataset_config.items(), lengths, seed_counts)):
        local_q = q_all[cursor:cursor + length]
        local_p = p_all[cursor:cursor + length]
        initial = int(np.random.default_rng(int(training_args.seed) + dataset_id).integers(length))
        seeds = (
            np.empty(0, dtype=np.int64)
            if defer_seeds_to_v2
            else farthest_cuda(local_q, local_p, int(count), initial)
        )
        seed_lists.append(seeds.tolist())
        partitions.append({"dataset_id": dataset_id, "name": f"mmeb/{config['subset_name']}", "offset": cursor, "length": int(length)})
        if defer_seeds_to_v2:
            print(
                f"[DYNAMIC_HARD_BATCH_SEEDS_DEFERRED] dataset={partitions[-1]['name']} "
                "owner=dynamic_hard_batch_v2_B3_proportional_preprocessing",
                flush=True,
            )
        else:
            print(f"[DYNAMIC_HARD_BATCH_SEEDS] dataset={partitions[-1]['name']} seeds={len(seeds)} initial={initial}", flush=True)
        cursor += length
    teacher_fingerprint = (
        compute_teacher_adapter_fingerprint(model_args.model_name)
        if model_args.init_lora_model else None
    )
    if teacher_fingerprint:
        print(
            "[DYNAMIC_HARD_BATCH_TEACHER_OK] "
            f"adapter={os.path.realpath(model_args.model_name)} "
            f"fingerprint={teacher_fingerprint}",
            flush=True,
        )
    manifest = {
        "version": 1, "complete": True, "method": "per_dataset_dynamic_hard_batch_greedy",
        "num_cases": total, "dimension": dimension, "dataset_partitions": partitions,
        "dataset_offsets": offsets, "dataset_lengths": {p["name"]: p["length"] for p in partitions},
        "dataset_schedule": schedule.tolist(), "seeds": seed_lists,
        "seed_build_policy": (
            "deferred_to_dynamic_hard_batch_v2"
            if defer_seeds_to_v2 else "v1_uniform_schedule_farthest_point"
        ),
        "max_steps": int(training_args.max_steps), "global_batch_size": int(training_args.per_device_train_batch_size) * world,
        "candidate_multiplier": int(os.environ.get("DHBG_CANDIDATE_MULTIPLIER", "8")),
        "dataset_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_backbone": model_args.model_backbone,
        "model_name": os.path.realpath(model_args.model_name), "pooling": model_args.pooling,
        "init_lora_model": bool(model_args.init_lora_model),
        "init_lora_base_model": (
            os.path.realpath(model_args.init_lora_base_model)
            if model_args.init_lora_base_model else None
        ),
        "processor_name": (
            os.path.realpath(model_args.processor_name)
            if model_args.processor_name else None
        ),
        "teacher_adapter_fingerprint": teacher_fingerprint,
        "qwen2_model_impl": model_args.qwen2_model_impl,
        "qwen2_b3_attention_impl": model_args.qwen2_b3_attention_impl,
        "qwen2_b3_legacy_position_ids": model_args.qwen2_b3_legacy_position_ids,
        "resize_use_processor": bool(data_args.resize_use_processor),
        "resize_min_pixels": int(data_args.resize_min_pixels),
        "resize_max_pixels": int(data_args.resize_max_pixels),
        "transformers_version": transformers.__version__,
        "normalize": bool(model_args.normalize), "seed": int(training_args.seed), "world_size": world,
        "encode_batch_size_per_device": encode_batch_size,
    }
    publish_dir.mkdir(parents=True, exist_ok=True)
    manifest["artifact_sha256"] = {}
    for name in ("query_embeddings.npy", "positive_embeddings.npy"):
        manifest["artifact_sha256"][name] = _publish_regular_file(stage_dir / name, publish_dir / name)
    (publish_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[DYNAMIC_HARD_BATCH_CACHE] complete: {publish_dir}", flush=True)


if __name__ == "__main__":
    main()
