"""Online false-negative-risk deletion for an isolated in-batch experiment."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from src.training.data.collator import MultimodalDataCollator
from src.training.loss import DistributedContrastiveLoss
from src.training.online_false_negative_risk_core import (
    compact_nonself_neighbors,
    false_negative_risk_energy,
    probabilistic_risk_info_nce,
    risk_deletion_probabilities,
    systematic_probability_mask,
)
from src.training.trainer import GradCacheLateProcessTrainer


_CORPUS_IDS_KEY = "_online_fn_risk_corpus_ids"


def _exclude_self(neighbor_ids, query_ids, top_k):
    """Remove each row's own corpus ID while preserving search order."""
    return compact_nonself_neighbors(neighbor_ids, query_ids, top_k)


class OnlineFalseNegativeRiskCollator(MultimodalDataCollator):
    """Attach stable integer corpus IDs without changing model inputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache_dir = Path(os.environ["FNRW_CACHE_DIR"])
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        self.dataset_offsets = manifest["dataset_offsets"]

    def __call__(self, examples):
        corpus_ids = []
        for example in examples:
            dataset_name = example["global_dataset_name"]
            if dataset_name not in self.dataset_offsets:
                raise KeyError(f"dataset absent from risk-cache manifest: {dataset_name}")
            corpus_ids.append(self.dataset_offsets[dataset_name] + int(example["idx"]))
        queries, targets = super().__call__(examples)
        queries[_CORPUS_IDS_KEY] = corpus_ids
        return queries, targets


class FaissMutableNeighborStore:
    """Rank-0 mutable IVF index and asynchronous per-case neighbor table."""

    def __init__(self, cache_dir, expected_top_k):
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "online FN risk weighting requires FAISS in this isolated environment"
            ) from exc
        self.faiss = faiss
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads(
            (self.cache_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if not self.manifest.get("complete"):
            raise RuntimeError("false-negative risk cache is incomplete")
        self.top_k = int(self.manifest["top_k"])
        if self.top_k != int(expected_top_k):
            raise ValueError(
                f"cache top_k={self.top_k} differs from requested {expected_top_k}"
            )
        faiss.omp_set_num_threads(int(os.environ.get("FNRW_FAISS_THREADS", "16")))
        self.neighbors = np.load(self.cache_dir / "neighbors.npy").copy()
        self.num_cases = int(self.manifest["num_cases"])
        self.dimension = int(self.manifest["dimension"])
        if self.manifest.get("version") != 2 or self.manifest.get(
            "neighbor_scope"
        ) != "within_dataset":
            raise RuntimeError(
                "risk cache must use version-2 within-dataset neighbor indexes"
            )
        self.partitions = self.manifest["dataset_partitions"]
        self.case_dataset_ids = np.empty(self.num_cases, dtype=np.int32)
        self.indexes = []
        expected_offset = 0
        for dataset_id, partition in enumerate(self.partitions):
            offset = int(partition["offset"])
            length = int(partition["length"])
            if int(partition["dataset_id"]) != dataset_id or offset != expected_offset:
                raise RuntimeError("dataset partitions are not contiguous and ordered")
            end = offset + length
            if length <= self.top_k or end > self.num_cases:
                raise RuntimeError("dataset partition cannot supply its internal Top-K")
            index = faiss.read_index(str(self.cache_dir / partition["index_file"]))
            if index.ntotal != length or index.d != self.dimension:
                raise RuntimeError(
                    f"dataset index mismatch for {partition['name']}"
                )
            self.indexes.append(index)
            self.case_dataset_ids[offset:end] = dataset_id
            expected_offset = end
        if expected_offset != self.num_cases:
            raise RuntimeError("dataset partitions do not cover the training corpus")
        if self.top_k <= 0 or self.top_k >= self.num_cases:
            raise ValueError(
                f"top_k must be in [1, {self.num_cases - 1}], got {self.top_k}"
            )
        if self.neighbors.shape != (
            self.num_cases,
            self.top_k,
        ):
            raise RuntimeError("risk cache index/table shape mismatch")
        self.last_updated = np.zeros(self.num_cases, dtype=np.int32)
        self.sum_last_updated = 0

    def _search_without_self(self, index, vectors, case_ids, dataset_name):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        while True:
            _, found = index.search(vectors, self.top_k + 1)
            try:
                return _exclude_self(found, case_ids, self.top_k)
            except RuntimeError as exc:
                current_nprobe = int(index.nprobe)
                if current_nprobe >= int(index.nlist):
                    raise RuntimeError(
                        f"dataset={dataset_name} exhausted all {index.nlist} IVF lists "
                        f"without Top-{self.top_k}: {exc}"
                    ) from exc
                next_nprobe = min(int(index.nlist), max(current_nprobe + 1, current_nprobe * 2))
                index.nprobe = next_nprobe
                print(
                    "[B3_DEBUG][ONLINE_FN_RISK_FAISS_RETRY] "
                    f"dataset={dataset_name} nprobe={current_nprobe}->{next_nprobe} "
                    f"reason={exc}",
                    flush=True,
                )

    def update_and_query(self, query_reps, target_reps, case_ids, step):
        query_reps = np.ascontiguousarray(query_reps, dtype=np.float32)
        target_reps = np.ascontiguousarray(target_reps, dtype=np.float32)
        if query_reps.ndim != 2 or target_reps.shape != query_reps.shape:
            raise ValueError("query/target cache updates must have equal rank-2 shapes")
        if query_reps.shape[1] != self.dimension:
            raise ValueError(
                f"current embedding dimension={query_reps.shape[1]}, "
                f"cache dimension={self.dimension}"
            )
        case_ids = np.asarray(case_ids, dtype=np.int64)
        if np.any(case_ids < 0) or np.any(case_ids >= self.num_cases):
            raise IndexError("batch contains an out-of-range corpus ID")
        # If a row is duplicated, the representations are equivalent positives;
        # use the last occurrence for the single canonical cache entry.
        last_position = {}
        for position, case_id in enumerate(case_ids.tolist()):
            last_position[case_id] = position
        unique_ids = np.fromiter(last_position.keys(), dtype=np.int64)
        unique_positions = np.fromiter(last_position.values(), dtype=np.int64)
        unique_targets = target_reps[unique_positions]
        unique_dataset_ids = self.case_dataset_ids[unique_ids]
        for dataset_id in np.unique(unique_dataset_ids):
            selected = np.flatnonzero(unique_dataset_ids == dataset_id)
            dataset_case_ids = unique_ids[selected]
            dataset_targets = unique_targets[selected]
            index = self.indexes[int(dataset_id)]
            index.update_vectors(dataset_case_ids, dataset_targets)
            updated_neighbors = self._search_without_self(
                index,
                dataset_targets,
                dataset_case_ids,
                self.partitions[int(dataset_id)]["name"],
            )
            self.neighbors[dataset_case_ids] = updated_neighbors

        old_steps = self.last_updated[unique_ids].astype(np.int64)
        self.sum_last_updated += int((int(step) - old_steps).sum())
        self.last_updated[unique_ids] = int(step)
        mean_staleness = float(step) - self.sum_last_updated / self.num_cases

        batch_dataset_ids = self.case_dataset_ids[case_ids]
        anchor_neighbors = np.empty((len(case_ids), self.top_k), dtype=np.int32)
        for dataset_id in np.unique(batch_dataset_ids):
            selected = np.flatnonzero(batch_dataset_ids == dataset_id)
            anchor_neighbors[selected] = self._search_without_self(
                self.indexes[int(dataset_id)],
                query_reps[selected],
                case_ids[selected],
                self.partitions[int(dataset_id)]["name"],
            )
        candidate_neighbors = self.neighbors[case_ids]
        return (
            anchor_neighbors,
            candidate_neighbors,
            batch_dataset_ids,
            mean_staleness,
        )


class _OnlineRiskLossBase:
    def _configure(self, *, temperature, risk_lambda, mutual_coefficient, top_k, cache_dir, log_path):
        self.temperature = float(temperature)
        self.risk_lambda = float(risk_lambda)
        self.mutual_coefficient = float(mutual_coefficient)
        self.max_pair_drop_probability = float(
            os.environ.get("FNRW_MAX_PAIR_DROP_PROB", "0.95")
        )
        self.max_drop_ratio = float(os.environ.get("FNRW_MAX_DROP_RATIO", "0.40"))
        self.mask_seed = int(os.environ.get("FNRW_MASK_SEED", "42"))
        self.top_k = int(top_k)
        self.log_path = Path(log_path)
        self.step = None
        self.local_case_ids = None
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.store = (
            FaissMutableNeighborStore(cache_dir, top_k) if self.rank == 0 else None
        )

    def set_batch_context(self, step, local_case_ids):
        self.step = int(step)
        self.local_case_ids = list(map(int, local_case_ids))

    def _global_case_ids(self):
        if self.local_case_ids is None:
            raise RuntimeError("online FN risk batch context was not set")
        if not dist.is_initialized():
            return self.local_case_ids
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, self.local_case_ids)
        return [case_id for rank_ids in gathered for case_id in rank_ids]

    @staticmethod
    def _summary(values, probabilities=(0.50, 0.90, 0.99), max_samples=65536):
        values = values.float().reshape(-1)
        if values.numel() == 0:
            return (float("nan"),) * (len(probabilities) + 1)
        stride = max(1, math.ceil(values.numel() / max_samples))
        sample = values[::stride]
        return (
            values.mean().item(),
            *(torch.quantile(sample, probability).item() for probability in probabilities),
        )

    def _energy(self, query, target, case_ids):
        batch = len(case_ids)
        started = time.perf_counter()
        rank_zero_error = [None]
        rank_zero_exception = None
        if self.rank == 0:
            try:
                (
                    anchor_neighbors,
                    candidate_neighbors,
                    batch_dataset_ids,
                    mean_staleness,
                ) = self.store.update_and_query(
                    query.detach().float().cpu().numpy(),
                    target.detach().float().cpu().numpy(),
                    case_ids,
                    self.step,
                )
                components = false_negative_risk_energy(
                    target.detach().float(),
                    case_ids,
                    case_ids,
                    anchor_neighbors,
                    candidate_neighbors,
                    mutual_coefficient=self.mutual_coefficient,
                    anchor_dataset_ids=batch_dataset_ids,
                    candidate_dataset_ids=batch_dataset_ids,
                )
                energy = torch.from_numpy(components["energy"]).to(query.device)
                structural = torch.stack(
                    [
                        torch.from_numpy(components["positive_candidate_similarity"]),
                        torch.from_numpy(components["jaccard"]),
                        torch.from_numpy(components["mutual"]),
                    ],
                    dim=0,
                )
                dataset_ids = torch.from_numpy(batch_dataset_ids.astype(np.int64))
            except Exception as exc:
                rank_zero_exception = exc
                rank_zero_error[0] = f"{type(exc).__name__}: {exc}"
                energy = torch.empty(
                    (batch, batch), device=query.device, dtype=torch.float32
                )
                structural = None
                dataset_ids = None
                mean_staleness = None
        else:
            energy = torch.empty((batch, batch), device=query.device, dtype=torch.float32)
            structural = None
            dataset_ids = None
            mean_staleness = None
        if dist.is_initialized():
            # Rank 0 owns the mutable FAISS state.  Propagate its failure before
            # the tensor collective so peer ranks never wait until NCCL timeout.
            dist.broadcast_object_list(rank_zero_error, src=0)
        if rank_zero_error[0] is not None:
            error = RuntimeError(
                "online FN-risk rank-0 FAISS/energy stage failed coherently: "
                + rank_zero_error[0]
            )
            if rank_zero_exception is not None:
                raise error from rank_zero_exception
            raise error
        if dist.is_initialized():
            dist.broadcast(energy, src=0)
        elapsed = time.perf_counter() - started
        return energy.detach(), structural, dataset_ids, mean_staleness, elapsed

    def _step_mask_seed(self, case_ids):
        digest = hashlib.sha256()
        digest.update(str(self.mask_seed).encode("ascii"))
        digest.update(str(self.step).encode("ascii"))
        digest.update(np.asarray(case_ids, dtype=np.int64).tobytes())
        return int.from_bytes(digest.digest()[:8], "little") % (2**63 - 1)

    def _loss(self, query, target, case_ids):
        risk_started = time.perf_counter()
        energy, structural, dataset_ids, mean_staleness, elapsed = self._energy(
            query, target, case_ids
        )
        drop_probability, probability_diagnostics = risk_deletion_probabilities(
            energy,
            risk_lambda=self.risk_lambda,
            max_pair_drop_probability=self.max_pair_drop_probability,
            max_drop_ratio=self.max_drop_ratio,
        )
        mask_seed = self._step_mask_seed(case_ids)
        rank_zero_error = [None]
        rank_zero_exception = None
        if self.rank == 0:
            try:
                drop_mask = systematic_probability_mask(
                    drop_probability, seed=mask_seed
                ).to(query.device)
            except Exception as exc:
                rank_zero_exception = exc
                rank_zero_error[0] = f"{type(exc).__name__}: {exc}"
                drop_mask = torch.empty(
                    energy.shape, device=query.device, dtype=torch.bool
                )
        else:
            drop_mask = torch.empty(
                energy.shape, device=query.device, dtype=torch.bool
            )
        if dist.is_initialized():
            dist.broadcast_object_list(rank_zero_error, src=0)
        if rank_zero_error[0] is not None:
            error = RuntimeError(
                "online FN-risk rank-0 probability sampler failed coherently: "
                + rank_zero_error[0]
            )
            if rank_zero_exception is not None:
                raise error from rank_zero_exception
            raise error
        if dist.is_initialized():
            dist.broadcast(drop_mask, src=0)
        actual_drop_count = drop_mask.sum(dim=1)
        max_drop_count = int(probability_diagnostics["max_drop_count"])
        if torch.any(actual_drop_count > max_drop_count):
            raise RuntimeError(
                "probabilistic FN-risk sampler exceeded its per-anchor budget"
            )
        loss, diagnostics = probabilistic_risk_info_nce(
            query, target, drop_mask, temperature=self.temperature
        )
        if self.rank == 0:
            off_diagonal = ~torch.eye(len(case_ids), dtype=torch.bool)
            same_dataset = dataset_ids[:, None] == dataset_ids[None, :]
            cpu_mask = off_diagonal & same_dataset
            device_mask = cpu_mask.to(query.device)
            similarity = structural[0][cpu_mask]
            jaccard = structural[1][cpu_mask]
            mutual = structural[2][cpu_mask]
            risk_energy = energy[device_mask]
            within_drop_probability = drop_probability[device_mask]
            within_drop_mask = drop_mask[device_mask]
            deleted_energy = risk_energy[within_drop_mask]
            retained_energy = risk_energy[~within_drop_mask]
            within_model_similarity = (
                diagnostics["plain_logits"][device_mask] * self.temperature
            )
            deleted_model_similarity = within_model_similarity[within_drop_mask]
            retained_model_similarity = within_model_similarity[~within_drop_mask]
            rows = torch.arange(query.size(0), device=query.device)
            negative_logits = diagnostics["plain_logits"].clone()
            negative_logits[rows, rows] = float("-inf")
            hardest_indices = negative_logits.argmax(dim=1)
            hardest_deleted = drop_mask[rows, hardest_indices]
            mutual_bool = mutual.bool().to(query.device)
            plain_accuracy = (
                diagnostics["plain_logits"].argmax(dim=1) == rows
            ).float().mean()
            sampled_accuracy = (
                diagnostics["sampled_logits"].argmax(dim=1) == rows
            ).float().mean()
            mask_sha256 = hashlib.sha256(
                drop_mask.detach().cpu().numpy().tobytes()
            ).hexdigest()
            record = {
                "step": self.step,
                "sampling": "systematic_dependent_bernoulli",
                "mask_seed": mask_seed,
                "mask_sha256": mask_sha256,
                "total_negative_pairs": int(off_diagonal.sum().item()),
                "within_dataset_negative_pairs": int(cpu_mask.sum().item()),
                "cross_dataset_unadjusted_pairs": int(
                    (off_diagonal & ~same_dataset).sum().item()
                ),
                "a_mean_p50_p90_p99": self._summary(similarity),
                "jaccard_mean_p50_p90_p99": self._summary(jaccard),
                "mutual_rate": mutual.mean().item(),
                "energy_mean_p50_p90_p99": self._summary(risk_energy),
                "drop_probability_mean_p50_p90_p99": self._summary(
                    within_drop_probability
                ),
                "raw_expected_drop_per_anchor_mean": float(
                    probability_diagnostics["raw_expected_drop_count"].mean().item()
                ),
                "expected_drop_per_anchor_mean": float(
                    probability_diagnostics["expected_drop_count"].mean().item()
                ),
                "actual_drop_per_anchor_mean": float(actual_drop_count.float().mean().item()),
                "actual_drop_per_anchor_min": int(actual_drop_count.min().item()),
                "actual_drop_per_anchor_max": int(actual_drop_count.max().item()),
                "actual_drop_ratio": float(
                    actual_drop_count.sum().item() / off_diagonal.sum().item()
                ),
                "actual_within_dataset_drop_ratio": float(
                    within_drop_mask.float().mean().item()
                ),
                "max_drop_count": max_drop_count,
                "budget_cap_anchor_rate": float(
                    (probability_diagnostics["budget_scale"] < 1.0).float().mean().item()
                ),
                "anchors_with_zero_drop": int((actual_drop_count == 0).sum().item()),
                "deleted_energy_mean_p50_p90_p99": self._summary(deleted_energy),
                "retained_energy_mean_p50_p90_p99": self._summary(retained_energy),
                "deleted_model_similarity_mean_p50_p90_p99": self._summary(
                    deleted_model_similarity
                ),
                "retained_model_similarity_mean_p50_p90_p99": self._summary(
                    retained_model_similarity
                ),
                "hardest_negative_deleted_fraction": float(
                    hardest_deleted.float().mean().item()
                ),
                "mutual_pairs_deleted_fraction": (
                    float(within_drop_mask[mutual_bool].float().mean().item())
                    if mutual_bool.any().item()
                    else None
                ),
                "removed_softmax_probability_mass_mean": float(
                    diagnostics["removed_probability_mass"].mean().item()
                ),
                "retained_negatives_per_anchor_mean": float(
                    diagnostics["retained_negative_count"].float().mean().item()
                ),
                "loss": float(loss.detach().item()),
                "unfiltered_infonce_loss": float(diagnostics["plain_loss"].item()),
                "loss_delta_from_deletion": float(
                    loss.detach().item() - diagnostics["plain_loss"].item()
                ),
                "unfiltered_train_accuracy": float(plain_accuracy.item()),
                "post_deletion_train_accuracy": float(sampled_accuracy.item()),
                "quantile_sample_cap": 65536,
                "mean_cache_staleness_steps": mean_staleness,
                "risk_overhead_seconds": elapsed,
                "risk_and_sampling_overhead_seconds": time.perf_counter()
                - risk_started,
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if os.environ.get("FNRW_DEBUG", os.environ.get("B3_DEBUG", "1")) == "1":
                print(
                    "[B3_DEBUG][ONLINE_FN_RISK] "
                    f"step={self.step} "
                    + " ".join(
                        f"{key}={value}" for key, value in record.items() if key != "step"
                    ),
                    flush=True,
                )
        return loss


class OnlineRiskSimpleLoss(_OnlineRiskLossBase):
    def __init__(self, **kwargs):
        self._configure(**kwargs)

    def __call__(self, query, target, **kwargs):
        return self._loss(query, target, self._global_case_ids())


class OnlineRiskDistributedLoss(_OnlineRiskLossBase, DistributedContrastiveLoss):
    def __init__(self, **kwargs):
        DistributedContrastiveLoss.__init__(self, temperature=kwargs["temperature"])
        self._configure(**kwargs)

    def __call__(self, query, target, **kwargs):
        global_query = self.gather_tensor(query)
        global_target = self.gather_tensor(target)
        loss = self._loss(global_query, global_target, self._global_case_ids())
        return loss * self.word_size if self.scale_loss else loss


class OnlineFalseNegativeRiskTrainer(GradCacheLateProcessTrainer):
    """Isolated GradCache trainer using probabilistic FN-risk deletion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_args.negative_strategy != "vlm2vec_inbatch":
            raise ValueError("online FN risk requires vlm2vec_inbatch")
        if self.args.embedding_gradient_strategy != "info_nce":
            raise ValueError("online FN risk requires info_nce GradCache")
        cache_dir = os.environ["FNRW_CACHE_DIR"]
        top_k = int(os.environ.get("FNRW_TOP_K", "100"))
        risk_lambda = float(os.environ.get("FNRW_LAMBDA", "1.5"))
        mutual_coefficient = float(os.environ.get("FNRW_MUTUAL_COEFFICIENT", "0.5"))
        log_path = os.environ.get(
            "FNRW_LOG_PATH",
            os.path.join(self.args.output_dir, "online_fn_risk_steps.jsonl"),
        )
        manifest = json.loads(
            (Path(cache_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        dataset_config_bytes = Path(self.data_args.dataset_config).read_bytes()
        expected_dataset_sha = hashlib.sha256(dataset_config_bytes).hexdigest()
        cache_errors = []
        if not manifest.get("complete"):
            cache_errors.append("manifest is incomplete")
        if manifest.get("version") != 2:
            cache_errors.append("cache version is not per-dataset v2")
        if manifest.get("neighbor_scope") != "within_dataset":
            cache_errors.append("neighbor scope is not within-dataset")
        if manifest.get("dataset_config_sha256") != expected_dataset_sha:
            cache_errors.append("dataset config SHA256 differs")
        if int(manifest.get("top_k", -1)) != top_k:
            cache_errors.append("Top-K differs")
        if manifest.get("pooling") != self.model.pooling:
            cache_errors.append("pooling differs")
        if bool(manifest.get("normalize")) != bool(self.model.normalize):
            cache_errors.append("normalization differs")
        if not math.isclose(
            float(manifest.get("temperature", -1.0)),
            float(self.model.temperature),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            cache_errors.append("temperature differs")
        if os.path.realpath(manifest.get("model_name", "")) != os.path.realpath(
            self.model_args.model_name
        ):
            cache_errors.append("base model differs")
        supported_backbones = {"qwen2_vl", "qwen3_5_vl"}
        training_backbone = self.model_args.model_backbone
        if training_backbone not in supported_backbones:
            cache_errors.append(
                "online FN risk supports only qwen2_vl and qwen3_5_vl backbones"
            )
        cached_backbone = manifest.get("model_backbone")
        if cached_backbone is None:
            # Keep already-built Qwen2 caches usable. Qwen3.5 caches are never
            # accepted without an explicit backbone identity because both
            # models can otherwise share the same method/cache layout.
            if training_backbone != "qwen2_vl":
                cache_errors.append("cache does not identify its model backbone")
        elif cached_backbone != training_backbone:
            cache_errors.append(
                "model backbone differs: "
                f"cache={cached_backbone!r}, training={training_backbone!r}"
            )
        model_config = self.model.config
        text_config = getattr(model_config, "text_config", None)
        hidden_size = getattr(
            text_config,
            "hidden_size",
            getattr(model_config, "hidden_size", None),
        )
        if hidden_size is None:
            cache_errors.append("could not determine model hidden size")
        elif int(manifest.get("dimension", -1)) != int(hidden_size):
            cache_errors.append(
                "embedding dimension differs: "
                f"cache={manifest.get('dimension')!r}, training={hidden_size!r}"
            )
        # New manifests record all preprocessing inputs. Missing fields remain
        # accepted only for the legacy Qwen2 cache that predates this contract.
        require_explicit_preprocessing = training_backbone == "qwen3_5_vl"
        preprocessing_fields_present = "resize_use_processor" in manifest
        if require_explicit_preprocessing or preprocessing_fields_present:
            if not bool(manifest.get("resize_use_processor")):
                cache_errors.append("cache did not use processor-controlled resizing")
            if int(manifest.get("resize_min_pixels", -1)) != int(
                self.data_args.resize_min_pixels
            ):
                cache_errors.append("minimum image pixels differ")
            if int(manifest.get("resize_max_pixels", -1)) != int(
                self.data_args.resize_max_pixels
            ):
                cache_errors.append("maximum image pixels differ")
        if training_backbone == "qwen3_5_vl":
            if manifest.get("cache_model_initialization") != "checkpoint_base_model":
                cache_errors.append(
                    "Qwen3.5 cache was not encoded by the untouched checkpoint backbone"
                )
            if bool(manifest.get("lora_enabled_during_cache_build", True)):
                cache_errors.append("Qwen3.5 cache incorrectly enabled LoRA")
        faiss_config = manifest.get("faiss", {})
        expected_pq_m = int(os.environ.get("FNRW_FAISS_PQ_M", "96"))
        if expected_pq_m <= 0:
            cache_errors.append(f"FAISS PQ m must be positive, got {expected_pq_m}")
        elif int(faiss_config.get("pq_m", -1)) != expected_pq_m:
            cache_errors.append(
                "FAISS PQ m differs: "
                f"cache={faiss_config.get('pq_m')!r}, requested={expected_pq_m}"
            )
        if (
            hidden_size is not None
            and expected_pq_m > 0
            and int(hidden_size) % expected_pq_m != 0
        ):
            cache_errors.append(
                f"embedding dimension {hidden_size} is not divisible by "
                f"FAISS PQ m={expected_pq_m}"
            )
        if int(manifest.get("num_cases", -1)) != len(self.train_dataset):
            cache_errors.append("training corpus length differs")
        if cache_errors:
            raise RuntimeError(
                "online FN-risk cache is incompatible with this run: "
                + "; ".join(cache_errors)
            )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                "[ONLINE_FN_CACHE_CONTRACT_OK] "
                f"backbone={training_backbone} dimension={hidden_size} "
                f"cases={len(self.train_dataset)} top_k={top_k} "
                f"pq_m={faiss_config.get('pq_m')} "
                f"initialization={manifest.get('cache_model_initialization')} "
                f"pixels={self.data_args.resize_min_pixels}:"
                f"{self.data_args.resize_max_pixels} seed={self.args.seed}",
                flush=True,
            )
        loss_class = OnlineRiskDistributedLoss if self.is_ddp else OnlineRiskSimpleLoss
        self.online_risk_loss = loss_class(
            temperature=self.model.temperature,
            risk_lambda=risk_lambda,
            mutual_coefficient=mutual_coefficient,
            top_k=top_k,
            cache_dir=cache_dir,
            log_path=log_path,
        )
        self.gc.loss_fn = self.online_risk_loss
        self._online_risk_config = {
            "method": "online_false_negative_risk_probability_deletion",
            "backbone": training_backbone,
            "embedding_dimension": int(hidden_size),
            "model": os.path.realpath(self.model_args.model_name),
            "formula": "p_drop=min(1-exp(-lambda*(a^2+jaccard^2+mutual_coefficient*mutual)),max_pair_probability),then_row_budget_scale",
            "risk_lambda": risk_lambda,
            "mutual_coefficient": mutual_coefficient,
            "max_pair_drop_probability": self.online_risk_loss.max_pair_drop_probability,
            "max_drop_ratio": self.online_risk_loss.max_drop_ratio,
            "sampling": "systematic_dependent_bernoulli_with_exact_pair_marginals_and_floor_or_ceil_row_count",
            "mask_seed": self.online_risk_loss.mask_seed,
            "positive_deletion_probability": 0.0,
            "cross_dataset_deletion_probability": 0.0,
            "importance_correction": False,
            "top_k": top_k,
            "neighbor_scope": "within_dataset",
            "temperature": float(self.model.temperature),
            "risk_gradient": "detached",
            "negative_treatment": "sampled_hard_deletion_without_reweighting",
            "cache_dir": os.path.realpath(cache_dir),
            "cache_manifest_sha256": hashlib.sha256(
                (Path(cache_dir) / "manifest.json").read_bytes()
            ).hexdigest(),
        }

    def training_step(self, model, inputs, *args, **kwargs):
        queries, _ = inputs
        corpus_ids = queries.pop(_CORPUS_IDS_KEY)
        self.online_risk_loss.set_batch_context(
            self.state.global_step + 1, corpus_ids
        )
        return super().training_step(model, inputs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        (Path(output_dir) / "online_false_negative_risk_config.json").write_text(
            json.dumps(self._online_risk_config, indent=2) + "\n",
            encoding="utf-8",
        )
