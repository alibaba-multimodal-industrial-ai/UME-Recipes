"""Build rank-weighted CSR graphs from the existing v1 embedding/seed cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from build_online_false_negative_cache import _publish_regular_file
from src.training.dynamic_hard_batch_v2_core import (
    b3_rank_start,
    build_rank_delta_csr,
    make_b3_proportional_schedule,
)


def _sha256_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parquet_rows(config_path, item):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("B3 schedule derivation requires pyarrow") from exc
    root = Path(item["dataset_name"])
    if not root.is_absolute():
        root = (Path(config_path).resolve().parents[3] / root).resolve()
        if not root.exists():
            root = (Path.cwd() / item["dataset_name"]).resolve()
    path = root / item["subset_name"] / "train-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"B3 schedule parquet is missing: {path}")
    return int(pq.ParquetFile(path).metadata.num_rows)


def _top_ranks(local_query, local_positive, depth, query_chunk, device):
    n = len(local_query)
    if depth >= n:
        raise ValueError(f"rank_depth={depth} must be smaller than dataset size={n}")
    positive = torch.from_numpy(np.asarray(local_positive)).to(device=device, dtype=torch.float32)
    output = np.empty((n, depth), dtype=np.int32)
    with torch.no_grad():
        for start in range(0, n, query_chunk):
            end = min(start + query_chunk, n)
            query = torch.from_numpy(np.asarray(local_query[start:end])).to(
                device=device, dtype=torch.float32
            )
            scores = query @ positive.T
            output[start:end] = torch.topk(
                scores, k=depth, dim=1, largest=True, sorted=True
            ).indices.cpu().numpy().astype(np.int32, copy=False)
            if start == 0 or end == n or (start // query_chunk) % 100 == 0:
                print(f"[DHBG_V2_RANK] encoded={end}/{n}", flush=True)
    return output


def _farthest_seeds(local_query, local_positive, count, initial, device):
    """Match the v1 farthest-point seed construction for a requested K."""
    vectors = 0.5 * (
        torch.from_numpy(np.asarray(local_query)).to(device)
        + torch.from_numpy(np.asarray(local_positive)).to(device)
    )
    count = min(int(count), len(vectors))
    selected = []
    chosen = torch.zeros(len(vectors), dtype=torch.bool, device=device)
    max_similarity = torch.full(
        (len(vectors),), -torch.inf, dtype=torch.float32, device=device
    )
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
    return selected


def _validate_teacher_embeddings(query, positive, chunk_size=16384):
    """Reject nonfinite, zero or non-normalized teacher representations."""
    summaries = {}
    for side, embeddings in (("query", query), ("positive", positive)):
        norm_min, norm_max, norm_sum, rows = math.inf, -math.inf, 0.0, 0
        value_min, value_max = math.inf, -math.inf
        for start in range(0, len(embeddings), chunk_size):
            block = np.asarray(embeddings[start : start + chunk_size], dtype=np.float32)
            if not np.isfinite(block).all():
                bad = np.argwhere(~np.isfinite(block))[0]
                raise FloatingPointError(
                    f"teacher {side} embedding is nonfinite at "
                    f"row={start + int(bad[0])}, col={int(bad[1])}"
                )
            norms = np.linalg.norm(block, axis=1)
            norm_min = min(norm_min, float(norms.min()))
            norm_max = max(norm_max, float(norms.max()))
            norm_sum += float(norms.sum(dtype=np.float64))
            rows += len(norms)
            value_min = min(value_min, float(block.min()))
            value_max = max(value_max, float(block.max()))
        if norm_min < 0.98 or norm_max > 1.02:
            raise FloatingPointError(
                f"teacher {side} embeddings violate normalized-representation "
                f"contract: norm_min={norm_min}, norm_max={norm_max}"
            )
        summaries[side] = {
            "rows": rows,
            "norm_min": norm_min,
            "norm_mean": norm_sum / rows,
            "norm_max": norm_max,
            "value_min": value_min,
            "value_max": value_max,
        }
    print(f"[DHBG_V2_TEACHER_EMBEDDINGS_OK] {summaries}", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--b3-dataset-config", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--resize-min-pixels", type=int, required=True)
    parser.add_argument("--resize-max-pixels", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--rank-depth", type=int, default=256)
    parser.add_argument("--rank-alpha", type=float, default=0.03)
    parser.add_argument("--rank-query-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_steps <= 0 or args.global_batch_size <= 0:
        raise ValueError("max_steps and global_batch_size must be positive")
    if args.rank_depth <= 70 or args.rank_alpha <= 0:
        raise ValueError("rank_depth must exceed the largest safety boundary (70) and alpha must be positive")
    if args.rank_query_chunk <= 0:
        raise ValueError("rank_query_chunk must be positive")

    source_dir, output_dir, stage_dir = map(
        Path, (args.source_cache, args.output_cache, args.stage_dir)
    )
    source_manifest_path = source_dir / "manifest.json"
    source = json.loads(source_manifest_path.read_text())
    if not source.get("complete") or source.get("version") != 1:
        raise RuntimeError("source dynamic-hard-batch v1 cache is incomplete")
    if int(source.get("seed", -1)) != args.seed:
        raise RuntimeError("v2 must reuse a v1 seed set built with the same seed")
    if int(source.get("global_batch_size", -1)) != args.global_batch_size:
        raise RuntimeError("v1 and v2 global batch sizes differ")
    if not source.get("init_lora_model"):
        raise RuntimeError(
            "v2 source embeddings must come from a loaded teacher adapter, "
            "not a fresh/base model cache"
        )
    if not source.get("teacher_adapter_fingerprint"):
        raise RuntimeError("teacher cache is missing its trained-adapter fingerprint")
    if source.get("seed_build_policy") != "deferred_to_dynamic_hard_batch_v2":
        raise RuntimeError(
            "teacher cache was not built in v2 embedding-only preprocessing mode"
        )
    preprocessing_errors = []
    if source.get("qwen2_model_impl") != "b3":
        preprocessing_errors.append("qwen2_model_impl is not b3")
    if source.get("qwen2_b3_attention_impl") != "flash_attention_2":
        preprocessing_errors.append("attention implementation is not flash_attention_2")
    if source.get("qwen2_b3_legacy_position_ids") is not False:
        preprocessing_errors.append("legacy position IDs are enabled")
    if source.get("resize_use_processor") is not True:
        preprocessing_errors.append("processor resize is disabled")
    if int(source.get("resize_min_pixels", -1)) != args.resize_min_pixels:
        preprocessing_errors.append("resize_min_pixels differs")
    if int(source.get("resize_max_pixels", -1)) != args.resize_max_pixels:
        preprocessing_errors.append("resize_max_pixels differs")
    if source.get("transformers_version") != "5.9.0":
        preprocessing_errors.append("Transformers version is not 5.9.0")
    if preprocessing_errors:
        raise RuntimeError(
            "teacher preprocessing contract mismatch: "
            + "; ".join(preprocessing_errors)
        )
    if source.get("dataset_config_sha256") != _sha256_bytes(args.dataset_config):
        raise RuntimeError("v1 vectors were built from a different dataset config")

    original_config = yaml.safe_load(Path(args.dataset_config).read_text())
    b3_config = yaml.safe_load(Path(args.b3_dataset_config).read_text())
    if list(original_config) != list(b3_config):
        raise RuntimeError("original and B3 dataset config orders differ")
    original_weights = {
        name: float(item["weight"]) for name, item in original_config.items()
    }
    b3_weights = {name: float(item["weight"]) for name, item in b3_config.items()}
    if b3_weights != original_weights:
        raise RuntimeError(
            "B3 and original YAML weight fields differ; even though B3's "
            "ordered sampler ignores them, refusing an ambiguous config"
        )
    print(
        "[DHBG_V2_B3_YAML_WEIGHTS_VERIFIED_IGNORED] "
        f"weights={b3_weights} effective_schedule=complete_batch_inventory",
        flush=True,
    )
    partitions = source["dataset_partitions"]
    expected_names = [f"mmeb/{item['subset_name']}" for item in original_config.values()]
    if [item["name"] for item in partitions] != expected_names:
        raise RuntimeError("v1 partition order differs from the original dataset config")

    b3_parquet_rows = [
        _parquet_rows(args.b3_dataset_config, item) for item in b3_config.values()
    ]
    # Match load_b3_dataset and CustomRandomSampler: each B3 subset is capped
    # by num_sample_per_subset, then only complete global batches participate.
    b3_effective_rows = [
        min(rows, int(item.get("num_sample_per_subset", rows)))
        for rows, item in zip(b3_parquet_rows, b3_config.values())
    ]
    b3_batch_counts = [rows // args.global_batch_size for rows in b3_effective_rows]
    schedule = make_b3_proportional_schedule(
        b3_batch_counts, args.max_steps, args.seed
    )
    rank_starts = [b3_rank_start(item["subset_name"]) for item in b3_config.values()]
    print(
        "[DHBG_V2_B3_SCHEDULE] "
        + " ".join(
            f"{name}:{count}"
            for name, count in zip(original_config, b3_batch_counts)
        ),
        flush=True,
    )

    query = np.load(source_dir / "query_embeddings.npy", mmap_mode="r")
    positive = np.load(source_dir / "positive_embeddings.npy", mmap_mode="r")
    expected_shape = (int(source["num_cases"]), int(source["dimension"]))
    if query.shape != expected_shape or positive.shape != expected_shape:
        raise RuntimeError("v1 embedding arrays do not match their manifest")
    teacher_embedding_summary = _validate_teacher_embeddings(query, positive)
    if args.rank_depth >= min(int(item["length"]) for item in partitions):
        raise ValueError("rank_depth must be smaller than every dataset partition")

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    device = torch.device("cuda", 0)
    graph_artifacts = []
    partition_out = []
    seed_lists = []
    for dataset_id, (partition, start_rank) in enumerate(zip(partitions, rank_starts)):
        offset, length = int(partition["offset"]), int(partition["length"])
        initial_seed = int(
            np.random.default_rng(args.seed + dataset_id).integers(length)
        )
        dataset_seeds = _farthest_seeds(
            query[offset : offset + length],
            positive[offset : offset + length],
            b3_batch_counts[dataset_id],
            initial_seed,
            device,
        )
        if len(dataset_seeds) != b3_batch_counts[dataset_id]:
            raise RuntimeError(
                f"failed to build the B3-sized seed set for {partition['name']}"
            )
        seed_lists.append(dataset_seeds)
        print(
            f"[DHBG_V2_SEEDS] dataset={partition['name']} "
            f"seed_set_size={len(dataset_seeds)} initial={initial_seed} "
            "policy=b3_complete_batch_count",
            flush=True,
        )
        print(
            f"[DHBG_V2_BUILD] dataset={partition['name']} length={length} "
            f"safe_window={start_rank}-{start_rank + 100} "
            f"rank_depth={args.rank_depth} alpha={args.rank_alpha}",
            flush=True,
        )
        top_ids = _top_ranks(
            query[offset : offset + length],
            positive[offset : offset + length],
            args.rank_depth,
            args.rank_query_chunk,
            device,
        )
        graph, tail = build_rank_delta_csr(top_ids, start_rank, args.rank_alpha)
        prefix = f"graph_dataset_{dataset_id:03d}"
        files = {
            "indptr_file": f"{prefix}_indptr.npy",
            "indices_file": f"{prefix}_indices.npy",
            "delta_file": f"{prefix}_delta.npy",
        }
        np.save(stage_dir / files["indptr_file"], graph.indptr.astype(np.int64))
        np.save(stage_dir / files["indices_file"], graph.indices.astype(np.int32))
        np.save(stage_dir / files["delta_file"], graph.data.astype(np.float16))
        graph_artifacts.extend(files.values())
        degree = np.diff(graph.indptr)
        item = dict(partition)
        item.update(
            files,
            rank_start=start_rank,
            primary_rank_end=start_rank + 100,
            tail_weight=tail,
            graph_nnz=int(graph.nnz),
            degree_mean=float(degree.mean()),
            degree_max=int(degree.max(initial=0)),
        )
        partition_out.append(item)
        print(
            f"[DHBG_V2_GRAPH_OK] dataset={partition['name']} nnz={graph.nnz} "
            f"degree_mean={degree.mean():.2f} tail_weight={tail:.8g}",
            flush=True,
        )
        del top_ids, graph
        torch.cuda.empty_cache()

    manifest = {
        "version": 2,
        "complete": False,
        "method": "b3_safe_smooth_rank_full_dataset_batch_greedy",
        "loss": "unchanged_cross_device_info_nce",
        "source_v1_manifest_sha256": _sha256_bytes(source_manifest_path),
        "source_v1_artifact_sha256": source.get("artifact_sha256"),
        "dataset_config_sha256": _sha256_bytes(args.dataset_config),
        "b3_dataset_config_sha256": _sha256_bytes(args.b3_dataset_config),
        "model_name": os.path.realpath(args.student_model),
        "student_model_name": os.path.realpath(args.student_model),
        "teacher_model_name": source["model_name"],
        "teacher_base_model_name": source.get("init_lora_base_model"),
        "teacher_processor_name": source.get("processor_name"),
        "teacher_init_lora_model": source["init_lora_model"],
        "teacher_adapter_fingerprint": source["teacher_adapter_fingerprint"],
        "teacher_embedding_summary": teacher_embedding_summary,
        "teacher_preprocessing": {
            key: source[key]
            for key in (
                "qwen2_model_impl",
                "qwen2_b3_attention_impl",
                "qwen2_b3_legacy_position_ids",
                "resize_use_processor",
                "resize_min_pixels",
                "resize_max_pixels",
                "transformers_version",
            )
        },
        "pooling": source["pooling"],
        "normalize": source["normalize"],
        "world_size": source["world_size"],
        "seed": args.seed,
        "max_steps": args.max_steps,
        "global_batch_size": args.global_batch_size,
        "num_cases": source["num_cases"],
        "dimension": source["dimension"],
        "dataset_offsets": source["dataset_offsets"],
        "dataset_lengths": source["dataset_lengths"],
        "dataset_partitions": partition_out,
        "seeds": seed_lists,
        "seed_set_sizes": dict(zip(original_config, b3_batch_counts)),
        "seed_set_sizes_ordered": b3_batch_counts,
        "seed_policy": "farthest_point_K_equals_B3_complete_batch_count_then_cycle",
        "dataset_schedule": schedule.tolist(),
        "schedule_policy": "shuffle_complete_B3_batch_inventory_per_pass",
        "b3_parquet_rows": dict(zip(original_config, b3_parquet_rows)),
        "b3_effective_rows": dict(zip(original_config, b3_effective_rows)),
        "b3_complete_batch_counts": dict(zip(original_config, b3_batch_counts)),
        "b3_yaml_weights_verified_but_ignored": b3_weights,
        "rank_alpha": args.rank_alpha,
        "rank_depth": args.rank_depth,
        "rank_tail_policy": "ranks_beyond_depth_share_w(depth)_baseline",
        "rank_indexing": "zero_based_matching_B3_array_slices",
        "unsafe_prefix_encoding": "negative_delta_symmetric_either_direction",
        "artifact_sha256": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in graph_artifacts:
        manifest["artifact_sha256"][name] = _publish_regular_file(
            stage_dir / name, output_dir / name
        )
    manifest["complete"] = True
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"[DHBG_V2_CACHE_OK] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
