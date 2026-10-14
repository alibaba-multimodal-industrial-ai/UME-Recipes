#!/usr/bin/env python3
"""Rebuild the missing B3 Qwen2-VL-2B teacher ranks efficiently.

The released B3 tree contains the final ``MMEB-train`` parquet shards but not
the ``MMEB-evaloutputs`` rank arrays used to create them.  This builder starts
from a verified VLM2Vec-Qwen2VL-2B query/positive embedding cache and writes
only the ranks needed by the isolated no-FN ablation:

* exact Top-100 targets for the 15 VQA/retrieval/general sources;
* exact rankings over unique class-label representatives for each recovered
  <=10k classification block.

It never materializes an N x N score matrix.  Queries are multiplied in small
chunks, datasets are distributed across all torchrun ranks, output arrays are
int32, and large embedding arrays are expected on node-local storage so NumPy
mmap does not hit OSS/FUSE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml

from build_online_false_negative_cache import _publish_regular_file


CLASSIFICATION = ("ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397")
VQA = ("OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W")
GENERAL = (
    "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i",
    "MSCOCO_i2t", "NIGHTS", "WebQA", "MSCOCO",
)
ALL_DATASETS = CLASSIFICATION + VQA + GENERAL
CLASS_PART_SIZE = 10_000
CLUSTER_SIZE = 16
GENERAL_DEPTH = 100
CLASS_GRAPH_PERCENT = 30
CLASS_INSTANCE_UNIQUE = 10
CLASS_PREFIX = "Represent the class label: "
# Identical text is encoded in different distributed BF16 batches while the
# reusable pair cache is built.  Flash-attention reductions can therefore
# produce slightly different float16 rows for the same class label.  B3 ranks
# classification candidates by unique label, so accept only numerically
# equivalent duplicates and use the first occurrence as the canonical label
# vector.  This remains deliberately strict enough to catch a mismatched or
# corrupted cache.
CLASS_REPRESENTATIVE_MIN_COSINE = 0.99


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_dataset_root(config_path: Path, configured: str) -> Path:
    root = Path(configured)
    if root.is_absolute():
        return root
    candidates = [
        (Path.cwd() / root).resolve(),
        (config_path.resolve().parents[3] / root).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parquet_path(config_path: Path, item: dict, subset: str | None = None) -> Path:
    root = resolve_dataset_root(config_path, str(item["dataset_name"]))
    subset_name = str(subset if subset is not None else item["subset_name"])
    files = sorted((root / subset_name).glob("train-*.parquet"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one parquet for {subset_name}, found {files} under {root}"
        )
    return files[0]


def _normal(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def source_identity(row: pd.Series, *, processed: bool) -> tuple[str, str, str, str]:
    label = _normal(row.get("pos_text"))
    if processed and label.startswith(CLASS_PREFIX):
        label = label[len(CLASS_PREFIX) :]
    return (
        _normal(row.get("qry")),
        _normal(row.get("qry_image_path")),
        label,
        _normal(row.get("pos_image_path")),
    )


def expected_part_lengths(total: int) -> list[int]:
    return [
        min(CLASS_PART_SIZE, total - start)
        for start in range(0, total, CLASS_PART_SIZE)
    ]


def recover_classification_partitions(
    source: pd.DataFrame,
    baseline: pd.DataFrame,
    dataset: str,
) -> list[np.ndarray]:
    """Recover B3's original <=10k block membership from its successful shard.

    Each official block is emitted contiguously after Metis ordering.  Blocks
    are padded independently to a multiple of 16, so their boundaries are
    recoverable even though the original ``batch_indices*.npy`` files are gone.
    """
    identity_to_source: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in source.iterrows():
        identity_to_source[source_identity(row, processed=False)].append(int(index))
    ambiguous = [key for key, values in identity_to_source.items() if len(values) != 1]
    if ambiguous:
        raise RuntimeError(
            f"Cannot recover {dataset} classification blocks unambiguously: "
            f"{len(ambiguous)} source identities are duplicated"
        )

    lengths = expected_part_lengths(len(source))
    emitted_lengths = [math.ceil(length / CLUSTER_SIZE) * CLUSTER_SIZE for length in lengths]
    if sum(emitted_lengths) != len(baseline):
        raise RuntimeError(
            f"Successful B3 shard cannot recover {dataset} blocks: "
            f"expected emitted rows={sum(emitted_lengths)}, found={len(baseline)}"
        )

    partitions: list[np.ndarray] = []
    seen_global: set[int] = set()
    cursor = 0
    for part, (source_length, emitted_length) in enumerate(
        zip(lengths, emitted_lengths), start=1
    ):
        segment = baseline.iloc[cursor : cursor + emitted_length]
        cursor += emitted_length
        recovered: list[int] = []
        seen_part: set[int] = set()
        for _, row in segment.iterrows():
            key = source_identity(row, processed=True)
            candidates = identity_to_source.get(key)
            if not candidates:
                raise RuntimeError(
                    f"B3 shard row in {dataset} part {part} is absent from the raw source: {key}"
                )
            index = candidates[0]
            if index not in seen_part:
                recovered.append(index)
                seen_part.add(index)
        if len(recovered) != source_length:
            raise RuntimeError(
                f"Recovered {len(recovered)}/{source_length} unique source rows "
                f"for {dataset} part {part}"
            )
        overlap = seen_global.intersection(seen_part)
        if overlap:
            raise RuntimeError(
                f"Recovered classification blocks overlap for {dataset}: {sorted(overlap)[:8]}"
            )
        seen_global.update(seen_part)
        partitions.append(np.asarray(recovered, dtype=np.int64))
    if len(seen_global) != len(source):
        raise RuntimeError(
            f"Recovered classification blocks cover {len(seen_global)}/{len(source)} rows for {dataset}"
        )
    return partitions


def exact_topk(
    query: np.ndarray,
    positive: np.ndarray,
    depth: int,
    query_chunk: int,
    device: torch.device,
    progress_label: str,
) -> np.ndarray:
    if not 0 < depth <= len(positive):
        raise ValueError(
            f"Invalid rank depth for {progress_label}: depth={depth}, targets={len(positive)}"
        )
    target = torch.from_numpy(np.asarray(positive)).to(device=device, dtype=torch.float32)
    output = np.empty((len(query), depth), dtype=np.int32)
    with torch.no_grad():
        for start in range(0, len(query), query_chunk):
            end = min(start + query_chunk, len(query))
            query_tensor = torch.from_numpy(np.asarray(query[start:end])).to(
                device=device, dtype=torch.float32
            )
            indices = torch.topk(
                query_tensor @ target.T,
                k=depth,
                dim=1,
                largest=True,
                sorted=True,
            ).indices
            output[start:end] = indices.cpu().numpy().astype(np.int32, copy=False)
            if start == 0 or end == len(query) or (start // query_chunk) % 100 == 0:
                print(
                    f"[B3_NO_FN_RANK_PROGRESS] {progress_label} rows={end}/{len(query)}",
                    flush=True,
                )
            del query_tensor, indices
    del target
    torch.cuda.empty_cache()
    return output


def validate_class_label_representatives(
    positive: np.ndarray,
    source_indices: np.ndarray,
    labels: list[str],
    representatives: np.ndarray,
    dataset: str,
    part_number: int,
    chunk_size: int = 4096,
) -> None:
    """Validate that duplicate rows are safe to collapse by class label.

    B3 classification candidates are unique text-only class labels.  The pair
    cache encodes repeated occurrences across different distributed BF16
    batches, where reduction order can change the final float16 values by a
    small amount.  Bitwise identity is therefore not a valid contract.  We
    instead require every occurrence to be cosine-equivalent to the first
    occurrence, which is used as that label's canonical candidate embedding.
    """
    representative_by_label = {
        labels[int(local_index)]: int(local_index) for local_index in representatives
    }
    representative_vectors = np.asarray(positive[source_indices[representatives]])
    vector_owners: dict[bytes, str] = {}
    for label, vector in zip(representative_by_label, representative_vectors):
        vector_key = vector.tobytes()
        previous = vector_owners.setdefault(vector_key, label)
        if previous != label:
            raise RuntimeError(
                f"Cannot use exact unique-label ranking for {dataset} part {part_number}: "
                f"distinct labels {previous!r} and {label!r} have identical target embeddings"
            )
    representative_rows = np.asarray(
        [representative_by_label[label] for label in labels], dtype=np.int64
    )
    bitwise_mismatched = 0
    minimum_cosine = 1.0
    maximum_absolute_delta = 0.0
    invalid_rows = 0
    for start in range(0, len(source_indices), chunk_size):
        end = min(start + chunk_size, len(source_indices))
        rows = np.asarray(
            positive[source_indices[start:end]], dtype=np.float32
        )
        reference_rows = source_indices[representative_rows[start:end]]
        references = np.asarray(positive[reference_rows], dtype=np.float32)
        different = np.any(rows != references, axis=1)
        bitwise_mismatched += int(np.count_nonzero(different))
        if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(references)):
            invalid_rows += int(
                np.count_nonzero(
                    ~np.all(np.isfinite(rows), axis=1)
                    | ~np.all(np.isfinite(references), axis=1)
                )
            )
            continue
        row_norm = np.linalg.norm(rows, axis=1)
        reference_norm = np.linalg.norm(references, axis=1)
        valid_norm = (row_norm > 0.0) & (reference_norm > 0.0)
        invalid_rows += int(np.count_nonzero(~valid_norm))
        if np.any(valid_norm):
            cosine = np.sum(
                rows[valid_norm] * references[valid_norm], axis=1
            ) / (row_norm[valid_norm] * reference_norm[valid_norm])
            minimum_cosine = min(minimum_cosine, float(np.min(cosine)))
            maximum_absolute_delta = max(
                maximum_absolute_delta,
                float(
                    np.max(
                        np.abs(
                            rows[valid_norm] - references[valid_norm]
                        )
                    )
                ),
            )
    if invalid_rows or minimum_cosine < CLASS_REPRESENTATIVE_MIN_COSINE:
        raise RuntimeError(
            f"Cannot use canonical unique-label ranking for {dataset} part "
            f"{part_number}: invalid_rows={invalid_rows}, "
            f"minimum_cosine={minimum_cosine:.8f}, "
            f"required>={CLASS_REPRESENTATIVE_MIN_COSINE:.8f}, "
            f"maximum_absolute_delta={maximum_absolute_delta:.8g}"
        )
    print(
        "[B3_NO_FN_CLASS_REPRESENTATIVES_OK] "
        f"dataset={dataset} part={part_number} rows={len(labels)} "
        f"unique_labels={len(representatives)} "
        f"bitwise_mismatched={bitwise_mismatched} "
        f"minimum_cosine={minimum_cosine:.8f} "
        f"maximum_absolute_delta={maximum_absolute_delta:.8g} "
        "canonical=first_occurrence",
        flush=True,
    )


def validate_vector_manifest(
    manifest: dict,
    source_config: Path,
    min_pixels: int,
    max_pixels: int,
) -> None:
    errors: list[str] = []
    if manifest.get("complete") is not True or int(manifest.get("version", -1)) != 1:
        errors.append("cache is incomplete")
    if manifest.get("dataset_config_sha256") != sha256_bytes(source_config):
        errors.append("dataset config hash differs")
    if manifest.get("model_backbone") != "qwen2_vl":
        errors.append("backbone is not qwen2_vl")
    if manifest.get("qwen2_model_impl") != "b3":
        errors.append("qwen2_model_impl is not b3")
    if manifest.get("qwen2_b3_attention_impl") != "flash_attention_2":
        errors.append("attention implementation is not flash_attention_2")
    if manifest.get("qwen2_b3_legacy_position_ids") is not False:
        errors.append("legacy position IDs are enabled")
    if manifest.get("normalize") is not True:
        errors.append("teacher embeddings are not normalized")
    if manifest.get("init_lora_model") is not True:
        errors.append("teacher adapter was not loaded")
    if not manifest.get("teacher_adapter_fingerprint"):
        errors.append("teacher adapter fingerprint is missing")
    if int(manifest.get("resize_min_pixels", -1)) != min_pixels:
        errors.append("minimum pixels differ")
    if int(manifest.get("resize_max_pixels", -1)) != max_pixels:
        errors.append("maximum pixels differ")
    if errors:
        raise RuntimeError("B3 no-FN teacher-vector contract failed: " + "; ".join(errors))


def validate_array(path: Path, shape: tuple[int, int]) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.shape != shape or array.dtype != np.float16:
        raise RuntimeError(
            f"Teacher vector array differs from manifest: {path}, "
            f"shape={array.shape}, dtype={array.dtype}, expected={shape}/float16"
        )
    return array


def dataset_stage_success(stage_dataset: Path, expected: dict) -> bool:
    success_path = stage_dataset / "_SUCCESS.json"
    if not success_path.is_file():
        return False
    actual = json.loads(success_path.read_text(encoding="utf-8"))
    if actual.get("contract") != expected:
        raise RuntimeError(
            f"Incompatible resumable rank stage at {stage_dataset}: {actual.get('contract')}"
        )
    for item in actual.get("artifacts", []):
        path = stage_dataset / item["name"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            return False
        if sha256_file(path) != item["sha256"]:
            return False
    return True


def write_stage_success(stage_dataset: Path, contract: dict, files: Iterable[Path]) -> None:
    artifacts = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    (stage_dataset / "_SUCCESS.json").write_text(
        json.dumps({"contract": contract, "artifacts": artifacts}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--resize-min-pixels", type=int, default=3136)
    parser.add_argument("--resize-max-pixels", type=int, default=1003520)
    args = parser.parse_args()
    if args.query_chunk <= 0:
        raise ValueError("query-chunk must be positive")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    source_manifest_path = args.source_cache / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    validate_vector_manifest(
        source_manifest,
        args.source_config,
        args.resize_min_pixels,
        args.resize_max_pixels,
    )
    source_config = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    baseline_config = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8"))
    if tuple(source_config) != ALL_DATASETS or tuple(baseline_config) != ALL_DATASETS:
        raise RuntimeError("B3/source dataset order differs from the verified 20-source order")

    partitions = source_manifest["dataset_partitions"]
    expected_partition_names = [f"mmeb/{entry['subset_name']}" for entry in source_config.values()]
    if [entry["name"] for entry in partitions] != expected_partition_names:
        raise RuntimeError("Teacher-vector partition order differs from the source config")
    expected_shape = (
        int(source_manifest["num_cases"]),
        int(source_manifest["dimension"]),
    )
    query = validate_array(args.source_cache / "query_embeddings.npy", expected_shape)
    positive = validate_array(args.source_cache / "positive_embeddings.npy", expected_shape)

    if rank == 0:
        args.stage_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[B3_NO_FN_RANK_INPUT_OK] "
            f"teacher={source_manifest.get('model_name')} vectors={expected_shape} "
            f"world_size={world} query_chunk={args.query_chunk}",
            flush=True,
        )
    if distributed:
        dist.barrier()

    for dataset_id in range(rank, len(ALL_DATASETS), world):
        dataset = ALL_DATASETS[dataset_id]
        partition = partitions[dataset_id]
        offset, length = int(partition["offset"]), int(partition["length"])
        stage_dataset = args.stage_dir / dataset
        stage_dataset.mkdir(parents=True, exist_ok=True)
        contract = {
            "format": "b3_qwen2_no_fn_rank_dataset_v1",
            "dataset": dataset,
            "offset": offset,
            "length": length,
            "dimension": expected_shape[1],
            "teacher_adapter_fingerprint": source_manifest["teacher_adapter_fingerprint"],
            "source_vector_artifact_sha256": source_manifest.get("artifact_sha256"),
            "source_dataset_config_sha256": sha256_bytes(args.source_config),
            "baseline_dataset_config_sha256": sha256_bytes(args.baseline_config),
            "resize_min_pixels": args.resize_min_pixels,
            "resize_max_pixels": args.resize_max_pixels,
            "classification_blocks_recovered_from_successful_b3": dataset in CLASSIFICATION,
        }
        if dataset_stage_success(stage_dataset, contract):
            print(f"[B3_NO_FN_RANK_STAGE_HIT] rank={rank} dataset={dataset}", flush=True)
            continue

        artifacts: list[Path] = []
        if dataset not in CLASSIFICATION:
            ranks = exact_topk(
                query[offset : offset + length],
                positive[offset : offset + length],
                GENERAL_DEPTH,
                args.query_chunk,
                device,
                dataset,
            )
            rank_path = stage_dataset / f"{dataset}_pred_rank.npy"
            np.save(rank_path, ranks)
            artifacts.append(rank_path)
            del ranks
        else:
            source_path = parquet_path(args.source_config, source_config[dataset])
            baseline_path = parquet_path(args.baseline_config, baseline_config[dataset])
            source_frame = pd.read_parquet(source_path)
            baseline_frame = pd.read_parquet(baseline_path)
            if len(source_frame) != length:
                raise RuntimeError(
                    f"Raw source/vector length mismatch for {dataset}: "
                    f"source={len(source_frame)} vectors={length}"
                )
            recovered_parts = recover_classification_partitions(
                source_frame, baseline_frame, dataset
            )
            graph_unique = math.ceil(
                source_frame["pos_text"].nunique() * CLASS_GRAPH_PERCENT / 100
            )
            for part_number, source_indices in enumerate(recovered_parts, start=1):
                subset = source_frame.iloc[source_indices].reset_index(drop=True)
                labels = subset["pos_text"].astype(str).tolist()
                representative_by_label: dict[str, int] = {}
                for local_index, label in enumerate(labels):
                    representative_by_label.setdefault(label, local_index)
                representatives = np.asarray(
                    list(representative_by_label.values()), dtype=np.int64
                )
                validate_class_label_representatives(
                    positive[offset : offset + length],
                    source_indices,
                    labels,
                    representatives,
                    dataset,
                    part_number,
                )
                depth = min(
                    len(representatives), max(graph_unique, CLASS_INSTANCE_UNIQUE)
                )
                ranked_representatives = exact_topk(
                    query[offset + source_indices],
                    positive[offset + source_indices[representatives]],
                    depth,
                    args.query_chunk,
                    device,
                    f"{dataset}:part{part_number}",
                )
                ranks = representatives[ranked_representatives].astype(np.int32, copy=False)
                rank_path = stage_dataset / f"{dataset}_pred_rank{part_number}.npy"
                index_path = stage_dataset / f"{dataset}_batch_indices{part_number}.npy"
                np.save(rank_path, ranks)
                np.save(index_path, source_indices)
                artifacts.extend((rank_path, index_path))
                del ranked_representatives, ranks
        write_stage_success(stage_dataset, contract, artifacts)
        print(
            f"[B3_NO_FN_RANK_DATASET_OK] rank={rank} dataset={dataset} "
            f"artifacts={len(artifacts)}",
            flush=True,
        )

    if distributed:
        dist.barrier()
    if rank == 0:
        output_artifacts: dict[str, dict[str, object]] = {}
        for dataset in ALL_DATASETS:
            stage_dataset = args.stage_dir / dataset
            success = json.loads((stage_dataset / "_SUCCESS.json").read_text(encoding="utf-8"))
            for item in success["artifacts"]:
                source_path = stage_dataset / item["name"]
                destination = args.output_cache / dataset / item["name"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                published_sha = _publish_regular_file(source_path, destination)
                if published_sha != item["sha256"]:
                    raise RuntimeError(f"Published rank hash differs for {destination}")
                output_artifacts[str(destination.relative_to(args.output_cache))] = item
        manifest = {
            "format": "b3_qwen2_no_fn_rank_cache_v1",
            "complete": True,
            "method": "exact_chunked_cosine_topk_from_vlm2vec_qwen2vl_2b",
            "source_vector_manifest_sha256": sha256_bytes(source_manifest_path),
            "source_vector_artifact_sha256": source_manifest.get("artifact_sha256"),
            "teacher_model": source_manifest.get("model_name"),
            "teacher_adapter_fingerprint": source_manifest["teacher_adapter_fingerprint"],
            "source_dataset_config_sha256": sha256_bytes(args.source_config),
            "baseline_dataset_config_sha256": sha256_bytes(args.baseline_config),
            "resize_min_pixels": args.resize_min_pixels,
            "resize_max_pixels": args.resize_max_pixels,
            "general_rank_depth": GENERAL_DEPTH,
            "classification_rank_unit": "unique_label_representative",
            "classification_representative_embeddings": (
                "same_label_cosine_equivalent_first_occurrence_canonical"
            ),
            "classification_representative_min_cosine": (
                CLASS_REPRESENTATIVE_MIN_COSINE
            ),
            "classification_blocks": "recovered_from_successful_b3_bs16_shards",
            "classification_false_negative_label_exclusion": False,
            "dataset_order": list(ALL_DATASETS),
            "world_size": world,
            "query_chunk": args.query_chunk,
            "artifacts": output_artifacts,
        }
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        # Keep a node-local completion marker as well.  The immediately
        # following shard build can consume this stage directly and avoid
        # reading ~GB of rank arrays back through OSS/FUSE.
        (args.stage_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
        args.output_cache.mkdir(parents=True, exist_ok=True)
        (args.output_cache / "manifest.json").write_text(manifest_text, encoding="utf-8")
        print(
            f"[B3_NO_FN_RANK_CACHE_OK] cache={args.output_cache} "
            f"artifacts={len(output_artifacts)}",
            flush=True,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
