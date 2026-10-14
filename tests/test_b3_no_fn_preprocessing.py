import json

import numpy as np
import pandas as pd
import pytest

import build_b3_no_fn_rank_cache as rank_cache
from build_b3_no_fn_filter_shards import validate_generated_rank_cache


def _row(index):
    return {
        "qry": f"query-{index}",
        "qry_image_path": f"image-{index}.jpg",
        "pos_text": f"label-{index % 3}",
        "pos_image_path": "",
    }


def test_recover_classification_blocks_from_successful_b3(monkeypatch):
    monkeypatch.setattr(rank_cache, "CLASS_PART_SIZE", 4)
    monkeypatch.setattr(rank_cache, "CLUSTER_SIZE", 4)
    source = pd.DataFrame([_row(index) for index in range(7)])
    # Two independently emitted blocks: four rows, then three rows plus B3's
    # padding duplicate.  Both are in arbitrary Metis order.
    baseline = pd.concat(
        [
            source.iloc[[2, 0, 3, 1]],
            source.iloc[[5, 6, 4, 5]],
        ],
        ignore_index=True,
    )
    baseline["pos_text"] = rank_cache.CLASS_PREFIX + baseline["pos_text"]
    recovered = rank_cache.recover_classification_partitions(
        source, baseline, "synthetic"
    )
    assert [part.tolist() for part in recovered] == [[2, 0, 3, 1], [5, 6, 4]]
    assert sorted(np.concatenate(recovered).tolist()) == list(range(7))


def test_recover_classification_blocks_rejects_cross_block_overlap(monkeypatch):
    monkeypatch.setattr(rank_cache, "CLASS_PART_SIZE", 4)
    monkeypatch.setattr(rank_cache, "CLUSTER_SIZE", 4)
    source = pd.DataFrame([_row(index) for index in range(7)])
    baseline = pd.concat(
        [source.iloc[[0, 1, 2, 3]], source.iloc[[3, 4, 5, 4]]],
        ignore_index=True,
    )
    baseline["pos_text"] = rank_cache.CLASS_PREFIX + baseline["pos_text"]
    with pytest.raises(RuntimeError, match="Recovered 3/3|overlap"):
        rank_cache.recover_classification_partitions(source, baseline, "synthetic")


def test_generated_rank_cache_contract_and_hash(tmp_path):
    dataset_dir = tmp_path / rank_cache.ALL_DATASETS[0]
    dataset_dir.mkdir()
    artifact = dataset_dir / "ranks.npy"
    np.save(artifact, np.asarray([[0, 1]], dtype=np.int32))
    relative = str(artifact.relative_to(tmp_path))
    manifest = {
        "format": "b3_qwen2_no_fn_rank_cache_v1",
        "complete": True,
        "dataset_order": list(rank_cache.ALL_DATASETS),
        "general_rank_depth": 100,
        "classification_rank_unit": "unique_label_representative",
        "classification_false_negative_label_exclusion": False,
        "teacher_model": "synthetic-teacher",
        "teacher_adapter_fingerprint": {"adapter_config_sha256": "synthetic"},
        "source_vector_artifact_sha256": {"query_embeddings.npy": "synthetic"},
        "source_dataset_config_sha256": "synthetic",
        "resize_min_pixels": 3136,
        "resize_max_pixels": 1003520,
        "artifacts": {
            relative: {
                "name": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": rank_cache.sha256_file(artifact),
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_generated_rank_cache(tmp_path, tmp_path) == manifest
    artifact.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="size differs"):
        validate_generated_rank_cache(tmp_path, tmp_path)


def test_class_label_representative_collapse_requires_cosine_equivalence():
    positive = np.asarray(
        [[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float16
    )
    source_indices = np.asarray([2, 1, 0, 3], dtype=np.int64)
    labels = ["a", "b", "a", "b"]
    representatives = np.asarray([0, 1], dtype=np.int64)
    rank_cache.validate_class_label_representatives(
        positive, source_indices, labels, representatives, "synthetic", 1
    )
    # Distributed BF16 encoding may make identical text differ slightly across
    # batches; such rows must remain valid unique-label candidates.
    positive[3, 0] = np.float16(1e-3)
    rank_cache.validate_class_label_representatives(
        positive, source_indices, labels, representatives, "synthetic", 1
    )
    positive[3, 0] = np.float16(0.25)
    with pytest.raises(RuntimeError, match="minimum_cosine"):
        rank_cache.validate_class_label_representatives(
            positive, source_indices, labels, representatives, "synthetic", 1
        )
