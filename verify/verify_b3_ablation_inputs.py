#!/usr/bin/env python3
"""Cheap, read-only contract check before launching a B3 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


EXPECTED_DATASETS = (
    "ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397",
    "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA",
    "Visual7W", "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t",
    "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "MSCOCO",
)


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Dataset config is not a mapping: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    args = parser.parse_args()
    baseline = load_yaml(args.baseline_config)
    actual = load_yaml(args.dataset_config)
    if tuple(baseline) != EXPECTED_DATASETS or tuple(actual) != EXPECTED_DATASETS:
        raise RuntimeError(
            "B3 ablation dataset order/coverage differs from the verified 20-source baseline"
        )
    if args.ablation in {"random_batch", "no_instance_hn"}:
        if args.dataset_config.read_bytes() != args.baseline_config.read_bytes():
            raise RuntimeError(
                f"{args.ablation} must use the byte-identical B3 baseline dataset config"
            )
    elif args.ablation == "no_fn_filter":
        dataset_rank_manifest_hashes = set()
        for dataset in EXPECTED_DATASETS:
            before = baseline[dataset]
            after = actual[dataset]
            for key, value in before.items():
                if key not in {"dataset_name", "subset_name"} and after.get(key) != value:
                    raise RuntimeError(
                        f"no_fn_filter changed non-ablation field {dataset}.{key}: "
                        f"baseline={value!r}, actual={after.get(key)!r}"
                    )
            subset = str(after.get("subset_name", ""))
            if not subset.endswith("_qwen2b_no_fn"):
                raise RuntimeError(f"Unexpected no-FN subset for {dataset}: {subset}")
            dataset_root = Path(str(after["dataset_name"]))
            success_path = dataset_root / subset / "_B3_NO_FN_SUCCESS.json"
            parquet_path = dataset_root / subset / "train-00000-of-00001.parquet"
            if not success_path.is_file() or not parquet_path.is_file():
                raise FileNotFoundError(
                    f"Incomplete no-FN cache for {dataset}: {success_path}, {parquet_path}"
                )
            success = json.loads(success_path.read_text(encoding="utf-8"))
            expected = {
                "format": "b3_qwen2_no_fn_filter_dataset_v1",
                "all_false_negative_protections_disabled": True,
                "classification_golden_label_exclusion": False,
                "cluster_size": 16,
                "instance_hard_negatives": 5,
            }
            for key, value in expected.items():
                if success.get(key) != value:
                    raise RuntimeError(
                        f"Invalid no-FN cache contract for {dataset}: "
                        f"{key}={success.get(key)!r}, expected {value!r}"
                    )
            dataset_rank_manifest_hashes.add(
                success.get("rank_cache_manifest_sha256")
            )
            if (
                success.get("parquet_bytes") != parquet_path.stat().st_size
                or success.get("parquet_sha256") != sha256_file(parquet_path)
            ):
                raise RuntimeError(
                    f"No-FN parquet does not match its completed-cache marker: {parquet_path}"
                )
            if dataset not in EXPECTED_DATASETS[:5] and success.get("graph_rank_slice") != [0, 100]:
                raise RuntimeError(
                    f"{dataset} does not use teacher ranks from Top-1: "
                    f"{success.get('graph_rank_slice')!r}"
                )
            if dataset in EXPECTED_DATASETS[:5]:
                if success.get("graph_unique_class_percent") != 30:
                    raise RuntimeError(
                        f"{dataset} changed B3's classification graph budget"
                    )
                if success.get("instance_candidate_unique_classes") != 10:
                    raise RuntimeError(
                        f"{dataset} changed B3's classification instance-HN budget"
                    )
        manifest_path = args.dataset_config.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format") != "b3_qwen2_no_fn_filter_v1"
            or manifest.get("all_false_negative_protections_disabled") is not True
            or manifest.get("classification_golden_label_exclusion") is not False
            or manifest.get("rank_prefix_start") != 0
            or manifest.get("rank_prefix_width") != 100
            or manifest.get("cluster_size") != 16
            or manifest.get("instance_hard_negatives") != 5
            or manifest.get("dataset_order") != list(EXPECTED_DATASETS)
            or manifest.get("baseline_config_sha256")
            != sha256_file(args.baseline_config)
        ):
            raise RuntimeError(f"Invalid no-FN root manifest: {manifest_path}")
        expected_rank_manifest_hash = manifest.get(
            "generated_rank_manifest_sha256"
        )
        if dataset_rank_manifest_hashes != {expected_rank_manifest_hash}:
            raise RuntimeError(
                "No-FN datasets were not all generated from the rank cache "
                f"recorded by the root manifest: datasets={dataset_rank_manifest_hashes}, "
                f"root={expected_rank_manifest_hash!r}"
            )
    else:
        raise ValueError(f"Unknown ablation: {args.ablation}")
    print(
        "[B3_ABLATION_INPUTS_OK] "
        f"ablation={args.ablation} datasets={len(actual)} config={args.dataset_config}",
        flush=True,
    )


if __name__ == "__main__":
    main()
