#!/usr/bin/env python3
"""Build the isolated B3 ``no_fn_filter`` ablation shards.

This is a parameterized copy of B3's two official Metis preprocessing paths.
The ablation changes only false-negative protection:

* non-classification sources use teacher ranks [0:100] instead of [30:130]
  or [70:170], and do not subtract the skipped prefix when sampling the five
  per-query instance negatives;
* classification sources do not remove the anchor's class label when building
  the graph or sampling instance negatives.

Metis batching, cluster size 16, five explicit instance negatives, dataset
weights, prompt prefixes and source rows remain aligned with the reproduced
Qwen2-VL-2B B3 baseline. Outputs live under a separate cache root and never
overwrite B3's released shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml


CLASSIFICATION = ("ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397")
VQA = ("OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W")
GENERAL = (
    "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i",
    "MSCOCO_i2t", "NIGHTS", "WebQA", "MSCOCO",
)
ALL_DATASETS = CLASSIFICATION + VQA + GENERAL
I2T_RETRIEVAL = {"MSCOCO_i2t", "VisualNews_i2t"}

CLUSTER_SIZE = 16
GRAPH_NEIGHBORS = 100
INSTANCE_NEGATIVES = 5
CLASS_GRAPH_PERCENT = 30
CLASS_INSTANCE_UNIQUE = 2 * INSTANCE_NEGATIVES
CLASS_PART_SIZE = 10_000
B3_RESIZE_MIN_PIXELS = 3136
B3_RESIZE_MAX_PIXELS = 1003520


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_explicit_qwen2b_rank_root(path: Path) -> bool:
    lowered = str(path).lower().replace("_", "-")
    return any(
        marker in lowered
        for marker in (
            "qwen2b",
            "qwen2-2b",
            "qwen2vl-2b",
            "qwen2-vl-2b",
        )
    )


def discover_rank_root(search_root: Path, datasets: Sequence[str], classification: bool) -> Path:
    coverage: dict[Path, set[str]] = defaultdict(set)
    suffix = "_pred_rank1.npy" if classification else "_pred_rank.npy"
    for file_path in search_root.rglob(f"*{suffix}"):
        dataset = file_path.parent.name
        expected = f"{dataset}{suffix}"
        if dataset in datasets and file_path.name == expected:
            if classification and not (file_path.parent / f"{dataset}_batch_indices1.npy").is_file():
                continue
            coverage[file_path.parent.parent].add(dataset)
    complete = [root for root, found in coverage.items() if found == set(datasets)]
    if not complete:
        summary = {str(root): sorted(found) for root, found in coverage.items()}
        raise FileNotFoundError(
            "Could not find one Qwen2 teacher rank root covering all datasets. "
            f"classification={classification}, search_root={search_root}, candidates={summary}"
        )
    explicit_qwen2b = [path for path in complete if is_explicit_qwen2b_rank_root(path)]
    if len(explicit_qwen2b) == 1:
        return explicit_qwen2b[0]
    if len(complete) == 1:
        return complete[0]
    # Never guess among CLIP/Qwen-7B/ambiguous teacher roots: using a
    # different teacher changes both the Metis graph and five explicit HNs,
    # making this a compound ablation.
    if len(explicit_qwen2b) > 1:
        candidates = explicit_qwen2b
    else:
        candidates = complete
    if len(candidates) != 1:
        raise RuntimeError(
            "Teacher rank-root discovery is ambiguous; set the explicit rank-root "
            f"argument. Candidates: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def discover_source_root(b3_root: Path, baseline: dict[str, dict]) -> Path:
    configured_roots = {
        Path(str(entry["dataset_name"]))
        for entry in baseline.values()
        if entry.get("dataset_name")
    }
    candidates = list(configured_roots) + [
        b3_root / "MMEB-train",
        b3_root / "MMEB-train",
    ]
    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if str(candidate) in checked:
            continue
        checked.append(str(candidate))
        if all(
            len(list((candidate / dataset).glob("train-*.parquet"))) == 1
            for dataset in ALL_DATASETS
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the 20 raw MMEB training subsets used by B3. "
        f"Checked {checked}; pass --source-root explicitly if they live elsewhere."
    )


def validate_generated_rank_cache(
    general_rank_root: Path,
    classification_rank_root: Path,
    baseline_config: Path | None = None,
) -> dict[str, object] | None:
    """Validate the compact rank cache before any expensive Metis work.

    Explicit legacy B3 rank roots do not have a manifest and remain supported.
    The newly regenerated cache is required to be self-describing and every
    node-local artifact is checked against its published size and SHA-256.
    """
    if general_rank_root.resolve() != classification_rank_root.resolve():
        return None
    manifest_path = general_rank_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("complete") is not True:
        errors.append("cache is incomplete")
    if manifest.get("format") != "b3_qwen2_no_fn_rank_cache_v1":
        errors.append(f"unexpected format={manifest.get('format')!r}")
    if manifest.get("dataset_order") != list(ALL_DATASETS):
        errors.append("dataset order differs")
    if int(manifest.get("general_rank_depth", -1)) != GRAPH_NEIGHBORS:
        errors.append("general rank depth differs from 100")
    if manifest.get("classification_rank_unit") != "unique_label_representative":
        errors.append("classification rank unit differs")
    if manifest.get("classification_false_negative_label_exclusion") is not False:
        errors.append("classification golden-label filtering is still enabled")
    if int(manifest.get("resize_min_pixels", -1)) != B3_RESIZE_MIN_PIXELS:
        errors.append("minimum pixels differ from B3")
    if int(manifest.get("resize_max_pixels", -1)) != B3_RESIZE_MAX_PIXELS:
        errors.append("maximum pixels differ from B3")
    if not manifest.get("teacher_adapter_fingerprint"):
        errors.append("teacher adapter fingerprint is missing")
    if not manifest.get("source_vector_artifact_sha256"):
        errors.append("source-vector artifact hashes are missing")
    if not manifest.get("source_dataset_config_sha256"):
        errors.append("source dataset-config hash is missing")
    if (
        baseline_config is not None
        and manifest.get("baseline_dataset_config_sha256")
        != sha256_file(baseline_config)
    ):
        errors.append("baseline dataset-config hash differs")
    if errors:
        raise RuntimeError("B3 no-FN rank-cache contract failed: " + "; ".join(errors))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("B3 no-FN rank-cache manifest has no artifacts")
    for relative, item in artifacts.items():
        path = general_rank_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"B3 no-FN rank artifact is missing: {path}")
        if path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"B3 no-FN rank artifact size differs: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"B3 no-FN rank artifact SHA-256 differs: {path}")
    print(
        "[B3_NO_FN_RANK_CACHE_VERIFIED] "
        f"root={general_rank_root} artifacts={len(artifacts)} "
        f"teacher={manifest.get('teacher_model')}",
        flush=True,
    )
    return manifest


def load_source(source_root: Path, dataset: str) -> tuple[pd.DataFrame, Path]:
    files = sorted((source_root / dataset).glob("train-*.parquet"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source parquet for {dataset}, found {files}"
        )
    return pd.read_parquet(files[0]), files[0]


def metis_partition(adjacency: list[list[int]], cluster_size: int) -> list[int]:
    try:
        import metis
    except ImportError as exc:
        raise RuntimeError(
            "B3 no-FN preprocessing requires the same Python metis package used "
            "by the official B3 preprocessing environment."
        ) from exc
    n_clusters = len(adjacency) // cluster_size
    if n_clusters <= 0:
        raise ValueError(f"Dataset is too small for cluster_size={cluster_size}")
    _, parts = metis.part_graph(adjacency, n_clusters)
    return list(parts)


def finalize_clusters(parts: Sequence[int], cluster_size: int, rng: random.Random) -> list[list[int]]:
    clusters_by_id: dict[int, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(parts):
        clusters_by_id[int(cluster_id)].append(index)
    clusters = sorted(clusters_by_id.values(), key=len, reverse=True)
    final: list[list[int]] = []
    remaining: list[int] = []
    for cluster in clusters:
        cluster = list(cluster)
        while len(cluster) > cluster_size:
            remaining.append(cluster.pop())
        while len(cluster) < cluster_size and remaining:
            cluster.append(remaining.pop())
        final.append(cluster)
    final.sort(key=min)
    if remaining:
        final.append(remaining)
    if final and len(final[-1]) != cluster_size:
        tail = final.pop()
        pieces = [tail[i : i + cluster_size] for i in range(0, len(tail), cluster_size)]
        if pieces and len(pieces[-1]) < cluster_size:
            leftover = pieces.pop()
            while len(leftover) < cluster_size:
                # Preserve B3's released post-processing literally. It walks
                # the earlier clusters and appends a sampled *position*
                # (0..cluster_size-1), rather than the source row stored at
                # that position.  This oddity is part of the verified B3
                # shard construction and is therefore retained in the
                # ablation instead of being silently "fixed".
                for donor in final:
                    if len(leftover) == cluster_size:
                        break
                    leftover.append(rng.randrange(len(donor)))
            pieces.append(leftover)
        final.extend(pieces)
    if not final or any(len(cluster) != cluster_size for cluster in final):
        raise RuntimeError("Metis post-processing failed to create fixed-size B3 clusters")
    return final


def mutual_knn_adjacency(preferences: np.ndarray) -> list[list[int]]:
    n_rows = preferences.shape[0]
    edges = [
        (row, int(candidate))
        for row, ranked in enumerate(preferences)
        for candidate in ranked
        if int(candidate) != row and 0 <= int(candidate) < n_rows
    ]
    edge_set = set(edges)
    adjacency = [[] for _ in range(n_rows)]
    # Match B3's implementation exactly: each reciprocal directed edge adds
    # both directions, so a mutual pair occurs twice in each adjacency list.
    # The duplicate multiplicity acts as the same uniform Metis edge weight
    # used to produce the successful baseline shards.
    for row, candidate in edges:
        if (candidate, row) in edge_set:
            adjacency[row].append(candidate)
            adjacency[candidate].append(row)
    return adjacency


def validate_built_frame(frame: pd.DataFrame, dataset: str) -> None:
    if frame.empty:
        raise RuntimeError(f"B3 no-FN preprocessing produced no rows for {dataset}")
    required = {
        "qry", "qry_image_path", "pos_text", "pos_image_path",
        "neg_text", "neg_image_path",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"B3 no-FN shard for {dataset} is missing columns: {missing}")
    bad = [
        row
        for row, (texts, images) in enumerate(
            zip(frame["neg_text"], frame["neg_image_path"])
        )
        if len(texts) != INSTANCE_NEGATIVES or len(images) != INSTANCE_NEGATIVES
    ]
    if bad:
        raise RuntimeError(
            f"B3 no-FN shard for {dataset} does not have exactly "
            f"{INSTANCE_NEGATIVES} explicit negatives at rows {bad[:8]}"
        )


def sample_general_instance_negatives(
    clusters: Sequence[Sequence[int]],
    preferences: np.ndarray,
    per_query: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    samples_in_cluster_order: list[list[int]] = []
    for cluster in clusters:
        cluster_array = np.asarray(cluster, dtype=np.int64)
        flat = preferences[cluster_array].reshape(-1)
        valid_flat = flat[(flat >= 0) & (flat < preferences.shape[0])]
        scores = np.bincount(
            np.concatenate((valid_flat, cluster_array)), minlength=preferences.shape[0]
        ).astype(np.int64)
        scores[cluster_array] = 0
        valid = np.flatnonzero(scores > 0)
        if valid.size == 0:
            raise RuntimeError("No candidate remains for B3 instance-level mining")
        probabilities = scores[valid].astype(np.float64)
        probabilities /= probabilities.sum()
        sample_size = len(cluster) * per_query
        repeat = max(1, math.ceil(sample_size / valid.size))
        pool = np.repeat(valid, repeat)
        pool_prob = np.repeat(probabilities, repeat)
        pool_prob /= pool_prob.sum()
        chosen = rng.choice(pool, size=sample_size, replace=False, p=pool_prob)
        samples_in_cluster_order.extend(
            [int(value) for value in negatives]
            for negatives in chosen.reshape(len(cluster), per_query)
        )
    return samples_in_cluster_order


def first_n_unique(values: Iterable[str], count: int) -> list[str]:
    # Deliberately no golden-label removal: this is the all-FN-protection-off
    # classification ablation requested by the experiment definition.
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
            if len(result) == count:
                break
    return result


def classification_adjacency(
    preferences: np.ndarray, frame: pd.DataFrame, unique_count: int
) -> list[list[int]]:
    label_to_rows: dict[str, list[int]] = defaultdict(list)
    for row, label in enumerate(frame["pos_text"].tolist()):
        label_to_rows[label].append(row)
    directed: set[tuple[int, int]] = set()
    for row, ranked in enumerate(preferences):
        labels = frame.iloc[np.asarray(ranked, dtype=np.int64)]["pos_text"].tolist()
        for label in first_n_unique(labels, unique_count):
            for candidate in label_to_rows[label]:
                if candidate != row:
                    directed.add((row, candidate))
    adjacency = [[] for _ in range(len(frame))]
    for row, candidate in directed:
        if (candidate, row) in directed:
            adjacency[row].append(candidate)
    return adjacency


def sample_class_instance_negatives(
    clusters: Sequence[Sequence[int]],
    preferences: np.ndarray,
    frame: pd.DataFrame,
    unique_count: int,
    per_query: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    samples_in_cluster_order: list[list[str]] = []
    for cluster in clusters:
        score = Counter()
        for row in cluster:
            labels = frame.iloc[np.asarray(preferences[row], dtype=np.int64)]["pos_text"].tolist()
            score.update(first_n_unique(labels, unique_count))
        labels = np.asarray(list(score.keys()), dtype=object)
        weights = np.asarray([score[label] for label in labels], dtype=np.float64)
        weights /= weights.sum()
        sample_size = len(cluster) * per_query
        repeat = max(1, math.ceil(sample_size / len(labels)))
        pool = np.repeat(labels, repeat)
        pool_prob = np.repeat(weights, repeat)
        pool_prob /= pool_prob.sum()
        chosen = rng.choice(pool, size=sample_size, replace=False, p=pool_prob)
        samples_in_cluster_order.extend(
            [str(value) for value in negatives]
            for negatives in chosen.reshape(len(cluster), per_query)
        )
    return samples_in_cluster_order


def add_b3_candidate_prefixes(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame = frame.copy()
    if dataset in CLASSIFICATION:
        prefix = "Represent the class label: "
    elif dataset in VQA:
        prefix = "Represent the answer: "
    elif dataset in I2T_RETRIEVAL:
        prefix = "Represent the image caption: "
    else:
        return frame
    frame["pos_text"] = frame["pos_text"].map(lambda text: prefix + str(text))
    frame["neg_text"] = frame["neg_text"].map(
        lambda values: [prefix + str(text) for text in values]
    )
    return frame


def write_dataset(
    frame: pd.DataFrame, output_dir: Path, metadata: dict[str, object]
) -> dict[str, object]:
    validate_built_frame(frame, str(metadata["dataset"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    final_parquet = output_dir / "train-00000-of-00001.parquet"
    success = output_dir / "_B3_NO_FN_SUCCESS.json"
    with tempfile.TemporaryDirectory(prefix="b3_no_fn_") as temporary:
        temporary_path = Path(temporary) / final_parquet.name
        frame.to_parquet(temporary_path, index=False)
        shutil.copyfile(temporary_path, final_parquet)
    metadata = {
        **metadata,
        "format": "b3_qwen2_no_fn_filter_dataset_v1",
        "all_false_negative_protections_disabled": True,
        "rows": len(frame),
        "parquet_bytes": final_parquet.stat().st_size,
        "parquet_sha256": sha256_file(final_parquet),
    }
    success.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def build_general(
    dataset: str,
    source_root: Path,
    rank_root: Path,
    output_root: Path,
    seed: int,
    rank_cache_manifest_sha256: str | None,
) -> tuple[str, dict[str, object]]:
    print(f"[B3_NO_FN_STAGE] dataset={dataset} stage=load_source_and_top1_ranks", flush=True)
    frame, source_file = load_source(source_root, dataset)
    rank_file = rank_root / dataset / f"{dataset}_pred_rank.npy"
    ranks = np.load(rank_file, mmap_mode="r")
    if ranks.shape[0] != len(frame) or ranks.shape[1] < GRAPH_NEIGHBORS:
        raise RuntimeError(
            f"Teacher ranks do not match {dataset}: ranks={ranks.shape}, rows={len(frame)}"
        )
    preferences = np.asarray(ranks[:, :GRAPH_NEIGHBORS], dtype=np.int64)
    print(f"[B3_NO_FN_STAGE] dataset={dataset} stage=build_mutual_top1_graph", flush=True)
    parts = metis_partition(mutual_knn_adjacency(preferences), CLUSTER_SIZE)
    print(f"[B3_NO_FN_STAGE] dataset={dataset} stage=mine_unfiltered_instance_hn", flush=True)
    clusters = finalize_clusters(parts, CLUSTER_SIZE, random.Random(seed))
    negatives = sample_general_instance_negatives(
        clusters, preferences, INSTANCE_NEGATIVES, np.random.default_rng(seed)
    )
    ordered = [row for cluster in clusters for row in cluster]
    result = frame.iloc[ordered].reset_index(drop=True)
    result["neg_text"] = [
        frame.iloc[indices]["pos_text"].tolist() for indices in negatives
    ]
    result["neg_image_path"] = [
        frame.iloc[indices]["pos_image_path"].tolist() for indices in negatives
    ]
    result = add_b3_candidate_prefixes(result, dataset)
    subset = f"{dataset}_HNPS_Metis_bs16bi_0.100_qwen2b_no_fn"
    metadata = write_dataset(
        result,
        output_root / subset,
        {
            "dataset": dataset,
            "task_family": "general",
            "source": str(source_file),
            "teacher_rank": str(rank_file),
            "teacher_rank_shape": list(ranks.shape),
            "rank_cache_manifest_sha256": rank_cache_manifest_sha256,
            "graph_rank_slice": [0, 100],
            "classification_golden_label_exclusion": False,
            "cluster_size": CLUSTER_SIZE,
            "instance_hard_negatives": INSTANCE_NEGATIVES,
            "seed": seed,
        },
    )
    return subset, metadata


def build_classification(
    dataset: str,
    source_root: Path,
    rank_root: Path,
    output_root: Path,
    seed: int,
    rank_cache_manifest_sha256: str | None,
) -> tuple[str, dict[str, object]]:
    print(f"[B3_NO_FN_STAGE] dataset={dataset} stage=load_source", flush=True)
    frame, source_file = load_source(source_root, dataset)
    graph_unique_count = math.ceil(
        frame["pos_text"].nunique() * CLASS_GRAPH_PERCENT / 100
    )
    pieces: list[pd.DataFrame] = []
    rank_files: list[str] = []
    part_count = math.ceil(len(frame) / CLASS_PART_SIZE)
    for part in range(1, part_count + 1):
        print(
            f"[B3_NO_FN_STAGE] dataset={dataset} part={part}/{part_count} "
            "stage=build_same_label_allowed_graph_and_hn",
            flush=True,
        )
        dataset_dir = rank_root / dataset
        rank_file = dataset_dir / f"{dataset}_pred_rank{part}.npy"
        index_file = dataset_dir / f"{dataset}_batch_indices{part}.npy"
        ranks = np.load(rank_file, mmap_mode="r")
        source_indices = np.asarray(np.load(index_file), dtype=np.int64)
        subset_frame = frame.iloc[source_indices].reset_index(drop=True)
        if ranks.shape[0] != len(subset_frame):
            raise RuntimeError(
                f"Classification rank rows do not match {dataset} part {part}: "
                f"ranks={ranks.shape}, rows={len(subset_frame)}"
            )
        preferences = np.asarray(ranks, dtype=np.int64)
        parts = metis_partition(
            classification_adjacency(
                preferences, subset_frame, graph_unique_count
            ),
            CLUSTER_SIZE,
        )
        part_seed = seed + part
        clusters = finalize_clusters(parts, CLUSTER_SIZE, random.Random(part_seed))
        negatives = sample_class_instance_negatives(
            clusters,
            preferences,
            subset_frame,
            CLASS_INSTANCE_UNIQUE,
            INSTANCE_NEGATIVES,
            np.random.default_rng(part_seed),
        )
        ordered = [row for cluster in clusters for row in cluster]
        result = subset_frame.iloc[ordered].reset_index(drop=True)
        result["neg_text"] = negatives
        result["neg_image_path"] = [[""] * INSTANCE_NEGATIVES for _ in range(len(result))]
        pieces.append(result)
        rank_files.extend([str(rank_file), str(index_file)])
    combined = add_b3_candidate_prefixes(pd.concat(pieces, ignore_index=True), dataset)
    # B3's own no-prefix-filter naming uses 0.30P: start from the first
    # teacher-ranked class and retain the original 30%-of-classes graph
    # budget.  Keep that convention while adding an isolated cache suffix.
    subset = f"{dataset}_HNPS_Metis_bs16bi_0.30P_10_5_qwen2b_no_fn"
    metadata = write_dataset(
        combined,
        output_root / subset,
        {
            "dataset": dataset,
            "task_family": "classification",
            "source": str(source_file),
            "teacher_rank_files": rank_files,
            "rank_cache_manifest_sha256": rank_cache_manifest_sha256,
            "graph_unique_class_percent": CLASS_GRAPH_PERCENT,
            "instance_candidate_unique_classes": CLASS_INSTANCE_UNIQUE,
            "classification_golden_label_exclusion": False,
            "cluster_size": CLUSTER_SIZE,
            "instance_hard_negatives": INSTANCE_NEGATIVES,
            "seed": seed,
        },
    )
    return subset, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3-root", type=Path, default=Path("./B3"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--general-rank-root", type=Path)
    parser.add_argument("--classification-rank-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("exp/cache/b3_qwen2_no_fn_filter/MMEB-train"),
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/data_configs/mmeb_new/mmeb20_HNPS_bs16bi_30.130_30P.10.5_70.170_qwen2b.yaml"),
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("exp/cache/b3_qwen2_no_fn_filter/mmeb20_b3_no_fn_filter.yaml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8"))
    if tuple(baseline.keys()) != ALL_DATASETS:
        raise RuntimeError(
            "Baseline B3 dataset order changed; refusing to build an incomparable ablation: "
            f"found={tuple(baseline.keys())} expected={ALL_DATASETS}"
        )
    source_root = args.source_root or discover_source_root(args.b3_root, baseline)
    search_root = args.b3_root / "MMEB-evaloutputs"
    general_rank_root = args.general_rank_root or discover_rank_root(
        search_root, VQA + GENERAL, classification=False
    )
    class_rank_root = args.classification_rank_root or discover_rank_root(
        search_root, CLASSIFICATION, classification=True
    )
    generated_rank_manifest = validate_generated_rank_cache(
        general_rank_root, class_rank_root, args.baseline_config
    )
    generated_rank_manifest_sha256 = (
        sha256_file(general_rank_root / "manifest.json")
        if generated_rank_manifest is not None else None
    )
    print(
        "[B3_NO_FN_INPUTS] "
        f"source_root={source_root} general_rank_root={general_rank_root} "
        f"classification_rank_root={class_rank_root}",
        flush=True,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    subsets: dict[str, str] = {}
    metadata: dict[str, object] = {}
    for dataset_index, dataset in enumerate(ALL_DATASETS):
        expected_suffix = (
            "HNPS_Metis_bs16bi_0.30P_10_5_qwen2b_no_fn"
            if dataset in CLASSIFICATION
            else "HNPS_Metis_bs16bi_0.100_qwen2b_no_fn"
        )
        subset = f"{dataset}_{expected_suffix}"
        success_path = args.output_root / subset / "_B3_NO_FN_SUCCESS.json"
        if success_path.is_file() and not args.force:
            cached = json.loads(success_path.read_text(encoding="utf-8"))
            parquet_path = args.output_root / subset / "train-00000-of-00001.parquet"
            cache_ok = (
                cached.get("format") == "b3_qwen2_no_fn_filter_dataset_v1"
                and cached.get("all_false_negative_protections_disabled") is True
                and cached.get("classification_golden_label_exclusion") is False
                and int(cached.get("cluster_size", -1)) == CLUSTER_SIZE
                and int(cached.get("instance_hard_negatives", -1)) == INSTANCE_NEGATIVES
                and cached.get("rank_cache_manifest_sha256")
                == generated_rank_manifest_sha256
                and parquet_path.is_file()
                and int(cached.get("parquet_bytes", -1)) == parquet_path.stat().st_size
                and cached.get("parquet_sha256") == sha256_file(parquet_path)
            )
            if cache_ok:
                subsets[dataset] = subset
                metadata[dataset] = cached
                print(f"[B3_NO_FN_CACHE_HIT] dataset={dataset} subset={subset}", flush=True)
                continue
            raise RuntimeError(
                f"Existing no-FN cache has an incompatible contract: {success_path}. "
                "Use --force only after moving or intentionally replacing that isolated cache."
            )
        dataset_seed = args.seed + 10_000 * dataset_index
        if dataset in CLASSIFICATION:
            subset, details = build_classification(
                dataset,
                source_root,
                class_rank_root,
                args.output_root,
                dataset_seed,
                generated_rank_manifest_sha256,
            )
        else:
            subset, details = build_general(
                dataset,
                source_root,
                general_rank_root,
                args.output_root,
                dataset_seed,
                generated_rank_manifest_sha256,
            )
        subsets[dataset] = subset
        metadata[dataset] = details
        print(
            f"[B3_NO_FN_DATASET_OK] dataset={dataset} subset={subset} "
            f"rows={details['rows']} sha256={details['parquet_sha256']}",
            flush=True,
        )
    config = {}
    for dataset, entry in baseline.items():
        config[dataset] = dict(entry)
        config[dataset]["dataset_name"] = str(args.output_root.resolve())
        config[dataset]["subset_name"] = subsets[dataset]
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "format": "b3_qwen2_no_fn_filter_v1",
        "all_false_negative_protections_disabled": True,
        "rank_prefix_start": 0,
        "rank_prefix_width": GRAPH_NEIGHBORS,
        "classification_golden_label_exclusion": False,
        "cluster_size": CLUSTER_SIZE,
        "instance_hard_negatives": INSTANCE_NEGATIVES,
        "dataset_order": list(ALL_DATASETS),
        "baseline_config": str(args.baseline_config),
        "baseline_config_sha256": sha256_file(args.baseline_config),
        "generated_config": str(args.config_output),
        "general_rank_root": str(general_rank_root),
        "classification_rank_root": str(class_rank_root),
        "generated_rank_manifest_sha256": generated_rank_manifest_sha256,
        "teacher_adapter_fingerprint": (
            generated_rank_manifest.get("teacher_adapter_fingerprint")
            if generated_rank_manifest is not None else None
        ),
        "datasets": metadata,
    }
    manifest_path = args.config_output.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[B3_NO_FN_CACHE_OK] config={args.config_output} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
