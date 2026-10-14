"""Isolated sampler/trainer for rank-weighted full-dataset batch greedy v2."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from src.training.data.collator import MultimodalDataCollator
from src.training.dynamic_hard_batch_v2_core import (
    full_dataset_greedy,
    teacher_adapter_fingerprint,
)
from src.training.trainer import GradCacheLateProcessTrainer


_IDS_KEY = "_dynamic_hard_batch_v2_corpus_ids"
_MANAGER = None


class DynamicHardBatchV2Manager:
    def __init__(self):
        self.cache_dir = Path(os.environ["DHBG_V2_RUNTIME_CACHE_DIR"])
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        if not self.manifest.get("complete") or self.manifest.get("version") != 2:
            raise RuntimeError("dynamic hard-batch v2 cache is incomplete or incompatible")
        if self.manifest.get("method") != "b3_safe_smooth_rank_full_dataset_batch_greedy":
            raise RuntimeError("unexpected dynamic hard-batch v2 method")
        self.log_path = Path(
            os.environ.get(
                "DHBG_V2_LOG_PATH", "dynamic_hard_batch_v2_steps.jsonl"
            )
        )
        # Accelerate's DataLoaderShard reads one batch ahead. Keep proofs keyed
        # by logical sampler step instead of a mutable "last batch" value, or
        # step 1 will be incorrectly compared with the prefetched step 2.
        self.constructed_batch_proofs = {}
        self.graphs = []
        for partition in self.manifest["dataset_partitions"]:
            arrays = (
                np.load(self.cache_dir / partition["indptr_file"], mmap_mode="r"),
                np.load(self.cache_dir / partition["indices_file"], mmap_mode="r"),
                np.load(self.cache_dir / partition["delta_file"], mmap_mode="r"),
            )
            length = int(partition["length"])
            if len(arrays[0]) != length + 1 or len(arrays[1]) != len(arrays[2]):
                raise RuntimeError(f"invalid v2 graph arrays for {partition['name']}")
            self.graphs.append(arrays)

    @property
    def rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def schedule(self):
        return self.manifest["dataset_schedule"]

    def _construct_rank0(self, step, dataset_id, seed_local, batch_size):
        partition = self.manifest["dataset_partitions"][dataset_id]
        indptr, indices, delta = self.graphs[dataset_id]
        started = time.perf_counter()
        selected, marginal, greedy_diagnostics = full_dataset_greedy(
            indptr,
            indices,
            delta,
            seed_local,
            batch_size,
            return_diagnostics=True,
        )
        elapsed = time.perf_counter() - started
        selected_mask = np.zeros(int(partition["length"]), dtype=bool)
        selected_mask[selected] = True
        positive_edges = unsafe_edges = 0
        for row in selected:
            begin, end = int(indptr[row]), int(indptr[row + 1])
            values = np.asarray(delta[begin:end], dtype=np.float32)[
                selected_mask[np.asarray(indices[begin:end], dtype=np.int64)]
            ]
            positive_edges += int((values > 0).sum())
            unsafe_edges += int((values < 0).sum())
        positive_pairs, unsafe_pairs = positive_edges // 2, unsafe_edges // 2
        if unsafe_pairs and greedy_diagnostics["fallback_selection_count"] == 0:
            raise RuntimeError(
                "hard-prefix exclusion failed: constructed batch contains "
                f"{unsafe_pairs} symmetric unsafe pairs"
            )
        pair_count = batch_size * (batch_size - 1) // 2
        valid_marginal = marginal[1:].astype(np.float64)
        correction = float(valid_marginal.sum())
        tail = float(partition["tail_weight"])
        record = {
            "step": int(step),
            "dataset": partition["name"],
            "seed_local": int(seed_local),
            "candidate_universe": int(partition["length"]),
            "batch_size": int(batch_size),
            "selected_unique": int(np.unique(selected).size),
            "candidate_policy": (
                "strict_prefix_when_available_with_position_level_legacy_fallback"
            ),
            "unsafe_relation": "symmetric_either_direction_rank_prefix",
            **greedy_diagnostics,
            "selected_global_ids_sha256": hashlib.sha256(
                (selected + int(partition["offset"])).astype(np.int64).tobytes()
            ).hexdigest()[:20],
            "selected_local_ids_first16": selected[:16].tolist(),
            "rank_start_primary_end_depth": [
                int(partition["rank_start"]),
                int(partition["primary_rank_end"]),
                int(self.manifest["rank_depth"]),
            ],
            "rank_alpha": float(self.manifest["rank_alpha"]),
            "tail_weight": tail,
            "marginal_gain_min_p50_p90_max": [
                float(valid_marginal.min()),
                float(np.quantile(valid_marginal, 0.5)),
                float(np.quantile(valid_marginal, 0.9)),
                float(valid_marginal.max()),
            ],
            "nonpositive_marginal_selections": int((valid_marginal <= 0).sum()),
            "weighted_hard_pairs": positive_pairs,
            "unsafe_prefix_pairs": unsafe_pairs,
            "weighted_hard_pair_fraction": positive_pairs / pair_count,
            "unsafe_prefix_pair_fraction": unsafe_pairs / pair_count,
            "batch_pair_score_mean_clipped_tail": (
                (pair_count * tail + correction) / pair_count
            ),
            "batch_objective_correction": correction,
            "construction_seconds": elapsed,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if os.environ.get("DHBG_V2_DEBUG", os.environ.get("B3_DEBUG", "1")) == "1":
            print(
                "[B3_DEBUG][DYNAMIC_HARD_BATCH_V2] "
                + " ".join(f"{key}={value}" for key, value in record.items()),
                flush=True,
            )
        return (selected + int(partition["offset"])).tolist()

    def construct(self, step, dataset_id, seed_local, batch_size):
        result = [None]
        if self.rank == 0:
            result[0] = self._construct_rank0(
                step, dataset_id, seed_local, batch_size
            )
        if dist.is_initialized():
            dist.broadcast_object_list(result, src=0)
        global_ids = np.asarray(result[0], dtype=np.int64)
        logical_step = int(step)
        self.constructed_batch_proofs[logical_step] = {
            "count": int(global_ids.size),
            "sorted_sha256": hashlib.sha256(
                np.sort(global_ids).tobytes()
            ).hexdigest(),
            "ordered_sha256": hashlib.sha256(global_ids.tobytes()).hexdigest(),
        }
        # This remains bounded during normal look-ahead and also during
        # Trainer's checkpoint data-skip replay.
        while len(self.constructed_batch_proofs) > 8:
            del self.constructed_batch_proofs[min(self.constructed_batch_proofs)]
        return global_ids.tolist()


def get_v2_manager():
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = DynamicHardBatchV2Manager()
    return _MANAGER


class DynamicHardBatchV2Sampler(torch.utils.data.Sampler[int]):
    """One B3-proportional, full-dataset-greedy global batch per step."""

    def __init__(
        self,
        total_batch_size,
        ds_lens,
        probabilities,
        source_block_size,
        num_steps,
        data_source,
        seed=42,
        **kwargs,
    ):
        self.total_batch_size = int(total_batch_size)
        self.num_steps = int(num_steps)
        self.data_source = data_source
        self.manager = get_v2_manager()
        manifest = self.manager.manifest
        if list(map(int, ds_lens)) != [
            int(value) for value in manifest["dataset_lengths"].values()
        ]:
            raise RuntimeError("runtime dataset lengths differ from v2 graph cache")
        if self.num_steps != int(manifest["max_steps"]):
            raise RuntimeError("max_steps differs from v2 graph cache")
        if self.total_batch_size != int(manifest["global_batch_size"]):
            raise RuntimeError("global batch size differs from v2 graph cache")
        if int(seed) != int(manifest["seed"]):
            raise RuntimeError("sampler seed differs from v2 graph cache")
        expected_seed_sizes = manifest.get("seed_set_sizes_ordered", [])
        if len(expected_seed_sizes) != len(manifest["seeds"]) or any(
            int(expected) <= 0 or len(seeds) != int(expected)
            for seeds, expected in zip(manifest["seeds"], expected_seed_sizes)
        ):
            raise RuntimeError("v2 cache seed sizes differ from B3 batch counts")

    def __len__(self):
        return self.total_batch_size * self.num_steps

    def __iter__(self):
        positions = [0] * len(self.manager.manifest["seeds"])
        for step, dataset_id in enumerate(self.manager.schedule, start=1):
            dataset_id = int(dataset_id)
            seeds = self.manager.manifest["seeds"][dataset_id]
            if not seeds:
                raise RuntimeError(f"dataset {dataset_id} has an empty v1 seed set")
            seed = int(seeds[positions[dataset_id] % len(seeds)])
            positions[dataset_id] += 1
            yield from self.manager.construct(
                step, dataset_id, seed, self.total_batch_size
            )


class DynamicHardBatchV2Collator(MultimodalDataCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offsets = get_v2_manager().manifest["dataset_offsets"]

    def __call__(self, examples):
        ids = [
            self.offsets[example["global_dataset_name"]] + int(example["idx"])
            for example in examples
        ]
        queries, positives = super().__call__(examples)
        queries[_IDS_KEY] = ids
        return queries, positives


class DynamicHardBatchV2Trainer(GradCacheLateProcessTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.data_args.negative_strategy != "vlm2vec_inbatch"
            or self.args.embedding_gradient_strategy != "info_nce"
        ):
            raise ValueError(
                "dynamic hard-batch v2 requires ordinary vlm2vec_inbatch InfoNCE"
            )
        self.dynamic_v2_manager = get_v2_manager()
        self._dynamic_v2_layout_verified = False
        manifest = self.dynamic_v2_manager.manifest
        errors = []
        fingerprint_result = [None]
        if self.dynamic_v2_manager.rank == 0:
            try:
                fingerprint_result[0] = {
                    "value": teacher_adapter_fingerprint(
                        os.environ["DHBG_V2_TEACHER_MODEL"]
                    )
                }
            except Exception as exc:
                fingerprint_result[0] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        if dist.is_initialized():
            dist.broadcast_object_list(fingerprint_result, src=0)
        if fingerprint_result[0].get("error"):
            errors.append(
                "failed to fingerprint current teacher adapter: "
                + fingerprint_result[0]["error"]
            )
        elif fingerprint_result[0]["value"] != manifest.get(
            "teacher_adapter_fingerprint"
        ):
            errors.append("teacher adapter contents changed after preprocessing")
        config_sha = hashlib.sha256(
            Path(self.data_args.dataset_config).read_bytes()
        ).hexdigest()
        b3_config_path = Path(os.environ["DHBG_V2_B3_DATASET_CONFIG"])
        b3_config_sha = hashlib.sha256(b3_config_path.read_bytes()).hexdigest()
        if manifest.get("dataset_config_sha256") != config_sha:
            errors.append("original dataset config SHA256 differs")
        if manifest.get("b3_dataset_config_sha256") != b3_config_sha:
            errors.append("B3 proportion config SHA256 differs")
        if os.path.realpath(manifest.get("model_name", "")) != os.path.realpath(
            self.model_args.model_name
        ):
            errors.append("base model differs")
        expected_teacher = os.environ.get("DHBG_V2_TEACHER_MODEL")
        if not expected_teacher or os.path.realpath(
            manifest.get("teacher_model_name", "")
        ) != os.path.realpath(expected_teacher):
            errors.append("teacher model differs")
        if not manifest.get("teacher_init_lora_model"):
            errors.append("teacher adapter was not loaded during preprocessing")
        if not manifest.get("teacher_adapter_fingerprint"):
            errors.append("teacher adapter fingerprint is missing")
        teacher_cache_dir = os.environ.get("DHBG_CACHE_DIR")
        if not teacher_cache_dir:
            errors.append("DHBG_CACHE_DIR is missing")
        else:
            try:
                teacher_cache_manifest = json.loads(
                    (Path(teacher_cache_dir) / "manifest.json").read_text()
                )
                if not teacher_cache_manifest.get("complete"):
                    errors.append("teacher embedding cache is incomplete")
                elif teacher_cache_manifest.get("artifact_sha256") != manifest.get(
                    "source_v1_artifact_sha256"
                ):
                    errors.append(
                        "teacher embedding artifacts differ from those used to build the graph"
                    )
                elif self.dynamic_v2_manager.rank == 0:
                    print(
                        "[DHBG_V2_TEACHER_ARTIFACT_LINK_OK] "
                        f"artifacts={teacher_cache_manifest['artifact_sha256']}",
                        flush=True,
                    )
            except Exception as exc:
                errors.append(
                    "failed to validate teacher embedding cache linkage: "
                    f"{type(exc).__name__}: {exc}"
                )
        if manifest.get("pooling") != self.model.pooling:
            errors.append("pooling differs")
        if bool(manifest.get("normalize")) != bool(self.model.normalize):
            errors.append("normalization differs")
        if int(manifest.get("world_size", -1)) != self.dynamic_v2_manager.world:
            errors.append("world size differs")
        if len(self.train_dataset) != int(manifest.get("num_cases", -1)):
            errors.append("training corpus length differs")
        if errors:
            raise RuntimeError(
                "dynamic hard-batch v2 cache is incompatible: " + "; ".join(errors)
            )

    def training_step(self, model, inputs, *args, **kwargs):
        queries, _ = inputs
        local_ids = list(map(int, queries.pop(_IDS_KEY)))
        if len(local_ids) != int(self.args.per_device_train_batch_size):
            raise RuntimeError("v2 collator produced an unexpected local batch size")
        if not self._dynamic_v2_layout_verified:
            local_tensor = torch.tensor(
                local_ids, dtype=torch.int64, device=self.args.device
            )
            if dist.is_initialized():
                gathered = [
                    torch.empty_like(local_tensor)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered, local_tensor.contiguous())
                actual = torch.cat(gathered).cpu().numpy()
            else:
                actual = local_tensor.cpu().numpy()
            logical_step = int(self.state.global_step) + 1
            expected_proof = self.dynamic_v2_manager.constructed_batch_proofs.get(
                logical_step
            )
            actual_sorted_sha = hashlib.sha256(
                np.sort(actual).astype(np.int64, copy=False).tobytes()
            ).hexdigest()
            if expected_proof is None:
                raise RuntimeError(
                    "missing dynamic batch proof for logical step "
                    f"{logical_step}; available_steps="
                    f"{sorted(self.dynamic_v2_manager.constructed_batch_proofs)}"
                )
            if (
                len(actual) != int(expected_proof["count"])
                or actual_sorted_sha != expected_proof["sorted_sha256"]
            ):
                raise RuntimeError(
                    "DataLoader/DDP sharding changed the v2 global batch IDs: "
                    f"logical_step={logical_step} actual_count={len(actual)} "
                    f"expected_count={expected_proof['count']} "
                    f"actual_sorted_sha256={actual_sorted_sha[:20]} "
                    f"expected_sorted_sha256="
                    f"{expected_proof['sorted_sha256'][:20]} "
                    f"constructed_steps="
                    f"{sorted(self.dynamic_v2_manager.constructed_batch_proofs)}"
                )
            if np.unique(actual).size != len(actual):
                raise RuntimeError("v2 global batch contains duplicate IDs")
            if self.dynamic_v2_manager.rank == 0:
                digest = hashlib.sha256(actual.tobytes()).hexdigest()[:20]
                print(
                    "[DHBG_V2_DDP_LAYOUT_OK] "
                    f"logical_step={logical_step} "
                    f"world={self.dynamic_v2_manager.world} "
                    f"local_batch={len(local_ids)} global_batch={len(actual)} "
                    f"actual_rank_major_sha256={digest} "
                    f"selected_set_sha256={actual_sorted_sha[:20]} "
                    f"prefetched_steps="
                    f"{sorted(self.dynamic_v2_manager.constructed_batch_proofs)}",
                    flush=True,
                )
            for constructed_step in list(
                self.dynamic_v2_manager.constructed_batch_proofs
            ):
                if constructed_step <= logical_step:
                    del self.dynamic_v2_manager.constructed_batch_proofs[
                        constructed_step
                    ]
            self._dynamic_v2_layout_verified = True
        return super().training_step(model, inputs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        manifest = self.dynamic_v2_manager.manifest
        config = {
            "method": "b3_hard_prefix_preferred_smooth_rank_full_dataset_batch_greedy_with_legacy_fallback",
            "graph_cache_method": manifest["method"],
            "loss": manifest["loss"],
            "candidate_policy": "strict_prefix_when_available_then_position_level_original_v2_gain",
            "unsafe_relation": "symmetric_either_direction_rank_prefix",
            "rank_alpha": manifest["rank_alpha"],
            "rank_depth": manifest["rank_depth"],
            "seed_set_sizes": manifest["seed_set_sizes"],
            "teacher_model_name": manifest["teacher_model_name"],
            "teacher_adapter_fingerprint": manifest[
                "teacher_adapter_fingerprint"
            ],
            "rank_tail_policy": manifest["rank_tail_policy"],
            "schedule_policy": manifest["schedule_policy"],
            "seed_policy": manifest["seed_policy"],
            "cache_manifest": os.path.realpath(
                self.dynamic_v2_manager.cache_dir / "manifest.json"
            ),
        }
        (Path(output_dir) / "dynamic_hard_batch_v2_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
