"""Isolated HOBIT-C sampler and trainer for supported VLM backbones."""

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
from src.training.hobit_core import (
    HOBITConfig,
    construct_hobit_source_epoch,
    rank_major_global_batch,
)
from src.training.trainer import GradCacheLateProcessTrainer


_IDS_KEY = "_hobit_global_case_ids"
_MANAGER = None


class HOBITManager:
    """Own the rolling student cache and deterministic global batch schedule."""

    def __init__(self):
        self.cache_dir = Path(os.environ["HOBIT_RUNTIME_CACHE_DIR"])
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        if not self.manifest.get("complete"):
            raise RuntimeError("HOBIT initial student cache is incomplete")
        if self.manifest.get("method") != "hobit_initial_student_embeddings":
            raise RuntimeError(
                "HOBIT cache has the wrong owner/method: "
                f"{self.manifest.get('method')!r}"
            )
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device(
            "cuda", int(os.environ.get("LOCAL_RANK", "0"))
        ) if torch.cuda.is_available() else torch.device("cpu")
        self.config = HOBITConfig(
            block_size=int(os.environ.get("HOBIT_SOURCE_BLOCK_SIZE", "512")),
            seed_size=int(os.environ.get("HOBIT_SEED_SIZE", "32")),
            candidate_k=int(os.environ.get("HOBIT_CANDIDATE_K", "128")),
            alpha=float(os.environ.get("HOBIT_ALPHA", "1.0")),
            temperature=float(os.environ.get("HOBIT_TEMPERATURE", "0.02")),
        )
        self.config.validate()
        self.partitions = self.manifest["dataset_partitions"]
        self.offsets = self.manifest["dataset_offsets"]
        self.source_orders: dict[int, np.ndarray] = {}
        self.source_cursors = [0] * len(self.partitions)
        self.source_epochs = [0] * len(self.partitions)
        self.expected_batches: dict[int, dict] = {}
        self.current_step = 0
        self.update_count = 0
        self.log_path = Path(os.environ["HOBIT_DIAGNOSTICS_PATH"])

        self.query_cache = self.target_cache = self.last_update = None
        init_status = [None]
        if self.rank == 0:
            try:
                self.query_cache = np.load(
                    self.cache_dir / "query_embeddings.npy", mmap_mode="r+"
                )
                self.target_cache = np.load(
                    self.cache_dir / "positive_embeddings.npy", mmap_mode="r+"
                )
                expected_shape = (
                    int(self.manifest["num_cases"]),
                    int(self.manifest["dimension"]),
                )
                if (
                    self.query_cache.shape != expected_shape
                    or self.target_cache.shape != expected_shape
                ):
                    raise RuntimeError(
                        "HOBIT cache shape mismatch: "
                        f"query={self.query_cache.shape} target={self.target_cache.shape} "
                        f"expected={expected_shape}"
                    )
                self.last_update = np.full(expected_shape[0], -1, dtype=np.int32)
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "event": "hobit_runtime_start",
                        "config": self.config.__dict__,
                        "num_cases": expected_shape[0],
                        "dimension": expected_shape[1],
                    }, sort_keys=True) + "\n")
            except Exception as exc:
                init_status[0] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(init_status, src=0, device=self.device)
        if init_status[0] is not None:
            raise RuntimeError(f"HOBIT runtime cache initialization failed: {init_status[0]}")

    def _broadcast_array(self, array: np.ndarray | None, length: int) -> np.ndarray:
        if not dist.is_initialized():
            if array is None:
                raise RuntimeError("rank 0 did not construct a HOBIT source epoch")
            return np.asarray(array, dtype=np.int64)
        tensor = torch.empty(length, dtype=torch.int64, device=self.device)
        if self.rank == 0:
            tensor.copy_(torch.from_numpy(np.asarray(array, dtype=np.int64)).to(self.device))
        dist.broadcast(tensor, src=0)
        return tensor.cpu().numpy()

    def _plan_source_epoch(
        self, source_id: int, avoid_first_global: np.ndarray | None = None
    ) -> None:
        partition = self.partitions[source_id]
        offset, length = int(partition["offset"]), int(partition["length"])
        epoch = self.source_epochs[source_id]
        plan_seed = int(self.manifest["seed"]) + 1_000_003 * source_id + epoch
        avoid_first = None
        if avoid_first_global is not None:
            avoid_first = np.asarray(avoid_first_global, dtype=np.int64) - offset
        expected_length = (length // self.config.block_size) * self.config.block_size
        status = [None]
        order = None
        started = time.time()
        if self.rank == 0:
            try:
                source_last_update = self.last_update[offset:offset + length]
                base_fraction = float(np.mean(source_last_update < 0))
                updated = source_last_update[source_last_update >= 0]
                staleness_mean = (
                    float(np.mean(self.current_step - updated)) if updated.size else None
                )
                print(
                    "[HOBIT_SOURCE_EPOCH_BEGIN] "
                    f"source={source_id}:{partition['name']} epoch={epoch} "
                    f"cases={length} blocks={expected_length // self.config.block_size} "
                    f"seed={plan_seed} base_cache_fraction={base_fraction:.6f} "
                    f"updated_staleness_mean={staleness_mean}",
                    flush=True,
                )

                def progress(done, total, diagnostics):
                    if done == 1 or done == total or done % 16 == 0:
                        print(
                            "[HOBIT_SOURCE_EPOCH_PROGRESS] "
                            f"source={source_id}:{partition['name']} epoch={epoch} "
                            f"blocks={done}/{total} candidate_union="
                            f"{diagnostics['candidate_union_size']} "
                            f"effective_k={diagnostics['candidate_k_effective']} "
                            f"hardness_mean={diagnostics['hardness_mean']:.8g}",
                            flush=True,
                        )

                previous_tf32 = None
                if self.device.type == "cuda":
                    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                    torch.backends.cuda.matmul.allow_tf32 = False
                try:
                    local_order, records, summary = construct_hobit_source_epoch(
                        self.query_cache[offset:offset + length],
                        self.target_cache[offset:offset + length],
                        config=self.config,
                        seed=plan_seed,
                        device=self.device,
                        avoid_first=avoid_first,
                        progress_callback=progress,
                    )
                finally:
                    if previous_tf32 is not None:
                        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
                order = local_order + offset
                summary.update({
                    "event": "hobit_source_epoch",
                    "source_id": source_id,
                    "source": partition["name"],
                    "source_epoch": epoch,
                    "seed": plan_seed,
                    "base_cache_fraction": base_fraction,
                    "updated_staleness_mean": staleness_mean,
                    "elapsed_seconds": time.time() - started,
                    "first_block": records[0],
                    "last_block": records[-1],
                })
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(summary, sort_keys=True) + "\n")
                print(
                    "[HOBIT_SOURCE_EPOCH_OK] "
                    f"source={source_id}:{partition['name']} epoch={epoch} "
                    f"covered={summary['covered_cases']}/{length} "
                    f"unique={summary['unique_cases']} dropped_tail={summary['dropped_tail']} "
                    f"candidate_expansions={summary['candidate_expansions']} "
                    f"sha256={summary['order_sha256'][:20]} "
                    f"elapsed={summary['elapsed_seconds']:.2f}s",
                    flush=True,
                )
            except Exception as exc:
                status[0] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(status, src=0, device=self.device)
        if status[0] is not None:
            raise RuntimeError(
                f"HOBIT failed to plan source {source_id} epoch {epoch}: {status[0]}"
            )
        received = self._broadcast_array(order, expected_length)
        if received.size != expected_length:
            raise RuntimeError("HOBIT broadcast an incomplete source epoch")
        if np.unique(received).size != received.size:
            raise RuntimeError("HOBIT source epoch contains duplicate global IDs")
        self.source_orders[source_id] = received
        self.source_cursors[source_id] = 0
        self.source_epochs[source_id] += 1
        if self.rank == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def next_source_block(
        self, source_id: int, avoid_first_global: np.ndarray | None = None
    ) -> np.ndarray:
        order = self.source_orders.get(source_id)
        cursor = self.source_cursors[source_id]
        if order is None or cursor + self.config.block_size > len(order):
            self._plan_source_epoch(source_id, avoid_first_global)
            order = self.source_orders[source_id]
            cursor = 0
        block = order[cursor:cursor + self.config.block_size]
        self.source_cursors[source_id] = cursor + self.config.block_size
        if np.unique(block).size != self.config.block_size:
            raise RuntimeError("HOBIT emitted a duplicate inside a source block")
        return block

    def record_expected_batch(self, step: int, global_ids: np.ndarray) -> None:
        global_ids = np.asarray(global_ids, dtype=np.int64)
        if np.unique(global_ids).size != global_ids.size:
            raise RuntimeError(
                f"HOBIT global batch {step} contains duplicate training cases"
            )
        self.expected_batches[int(step)] = {
            "count": int(global_ids.size),
            "sorted_sha256": hashlib.sha256(
                np.sort(global_ids).astype(np.int64, copy=False).tobytes()
            ).hexdigest(),
        }
        while len(self.expected_batches) > 8:
            del self.expected_batches[min(self.expected_batches)]

    @torch.no_grad()
    def update_cache(
        self,
        local_ids: list[int],
        query_reps: torch.Tensor,
        target_reps: torch.Tensor,
        step: int,
    ) -> None:
        if len(local_ids) != query_reps.size(0) or query_reps.shape != target_reps.shape:
            raise RuntimeError(
                "HOBIT cache update IDs/representations mismatch: "
                f"ids={len(local_ids)} query={tuple(query_reps.shape)} "
                f"target={tuple(target_reps.shape)}"
            )
        ids = torch.as_tensor(local_ids, device=query_reps.device, dtype=torch.int64)
        if dist.is_initialized():
            gathered_ids = [torch.empty_like(ids) for _ in range(self.world)]
            gathered_query = [torch.empty_like(query_reps) for _ in range(self.world)]
            gathered_target = [torch.empty_like(target_reps) for _ in range(self.world)]
            dist.all_gather(gathered_ids, ids.contiguous())
            dist.all_gather(gathered_query, query_reps.contiguous())
            dist.all_gather(gathered_target, target_reps.contiguous())
            ids = torch.cat(gathered_ids)
            query_reps = torch.cat(gathered_query)
            target_reps = torch.cat(gathered_target)
        ids_np = ids.cpu().numpy().astype(np.int64, copy=False)
        if np.unique(ids_np).size != ids_np.size:
            raise RuntimeError(f"HOBIT step {step} contains duplicate global IDs")
        if np.any(ids_np < 0) or np.any(ids_np >= int(self.manifest["num_cases"])):
            raise IndexError("HOBIT cache update contains an out-of-range global ID")
        if self.rank == 0:
            update_status = [None]
            try:
                query_np = query_reps.float().cpu().numpy().astype(np.float16)
                target_np = target_reps.float().cpu().numpy().astype(np.float16)
                self.query_cache[ids_np] = query_np
                self.target_cache[ids_np] = target_np
                self.last_update[ids_np] = int(step)
                self.update_count += ids_np.size
                if step == 1 or step % 100 == 0:
                    self.query_cache.flush()
                    self.target_cache.flush()
                    print(
                        "[HOBIT_CACHE_UPDATE] "
                        f"step={step} updated={ids_np.size} total_updates={self.update_count} "
                        f"unique_current_batch={np.unique(ids_np).size} "
                        f"query_norm_mean={np.linalg.norm(query_np.astype(np.float32), axis=1).mean():.8g} "
                        f"target_norm_mean={np.linalg.norm(target_np.astype(np.float32), axis=1).mean():.8g}",
                        flush=True,
                    )
            except Exception as exc:
                update_status[0] = f"{type(exc).__name__}: {exc}"
        else:
            update_status = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(update_status, src=0, device=query_reps.device)
        if update_status[0] is not None:
            raise RuntimeError(f"HOBIT rolling cache update failed: {update_status[0]}")
        self.current_step = int(step)


def get_hobit_manager() -> HOBITManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = HOBITManager()
    return _MANAGER


class HOBITSampler(torch.utils.data.Sampler[int]):
    """HOBIT-C source-aware replacement for VLM2VecInBatchSampler."""

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
        self.ds_lens = list(map(int, ds_lens))
        self.num_steps = int(num_steps)
        self.data_source = data_source
        self.seed = int(seed)
        self.manager = get_hobit_manager()
        self.source_block_size = int(source_block_size)
        weights = np.asarray(probabilities, dtype=np.float64)
        if (
            weights.ndim != 1
            or len(weights) != len(self.ds_lens)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or float(weights.sum()) <= 0
        ):
            raise ValueError(f"Invalid HOBIT source probabilities: {probabilities}")
        self.probabilities = weights / weights.sum()
        expected_lengths = [int(p["length"]) for p in self.manager.partitions]
        errors = []
        if self.ds_lens != expected_lengths:
            errors.append("dataset lengths differ from the HOBIT cache")
        if self.source_block_size != self.manager.config.block_size:
            errors.append("source block size differs from HOBIT block_size")
        if self.total_batch_size % self.source_block_size:
            errors.append("HOBIT source block does not divide the global batch")
        if self.total_batch_size % self.manager.world:
            errors.append("HOBIT global batch is not divisible by world size")
        if self.seed != int(self.manager.manifest["seed"]):
            errors.append("sampler seed differs from the HOBIT cache")
        if errors:
            raise RuntimeError("HOBIT sampler contract failed: " + "; ".join(errors))
        self._num_samples = self.total_batch_size * self.num_steps

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        blocks_per_batch = self.total_batch_size // self.source_block_size
        source_ids = rng.choice(
            len(self.ds_lens),
            size=self.num_steps * blocks_per_batch,
            p=self.probabilities,
        )
        print(
            "[HOBIT_SOURCE_SCHEDULE] "
            f"rank={self.manager.rank}/{self.manager.world} seed={self.seed} "
            f"source_blocks={len(source_ids)} probabilities={self.probabilities.tolist()} "
            f"first_sources={source_ids[:16].tolist()} "
            f"sha256={hashlib.sha256(source_ids.astype(np.int64).tobytes()).hexdigest()[:20]}",
            flush=True,
        )
        for step in range(1, self.num_steps + 1):
            selected_sources = source_ids[
                (step - 1) * blocks_per_batch:step * blocks_per_batch
            ]
            blocks = []
            selected_in_this_batch: dict[int, np.ndarray] = {}
            for source_id_value in selected_sources:
                source_id = int(source_id_value)
                block = self.manager.next_source_block(
                    source_id,
                    selected_in_this_batch.get(source_id),
                )
                blocks.append(block)
                selected_in_this_batch[source_id] = np.concatenate(
                    (selected_in_this_batch.get(source_id, np.empty(0, dtype=np.int64)), block)
                )
            global_ids = np.concatenate(blocks).astype(np.int64, copy=False)
            self.manager.record_expected_batch(step, global_ids)
            if step == 1 or step % 100 == 0:
                print(
                    "[HOBIT_GLOBAL_BATCH] "
                    f"rank={self.manager.rank}/{self.manager.world} step={step} "
                    f"sources={selected_sources.tolist()} "
                    f"global_batch={len(global_ids)} unique={np.unique(global_ids).size} "
                    f"sha256={hashlib.sha256(global_ids.tobytes()).hexdigest()[:20]}",
                    flush=True,
                )
            yield from rank_major_global_batch(global_ids, self.manager.world).tolist()


class HOBITCollator(MultimodalDataCollator):
    def __call__(self, examples):
        manager = get_hobit_manager()
        ids = [
            int(manager.offsets[example["global_dataset_name"]]) + int(example["idx"])
            for example in examples
        ]
        queries, positives = super().__call__(examples)
        queries[_IDS_KEY] = ids
        return queries, positives


class HOBITTrainer(GradCacheLateProcessTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hobit_manager = get_hobit_manager()
        self._hobit_pending_ids = None
        self._hobit_layout_verified = False
        manifest = self.hobit_manager.manifest
        errors = []
        if self.data_args.negative_strategy != "vlm2vec_inbatch":
            errors.append("negative_strategy must remain vlm2vec_inbatch")
        if self.args.embedding_gradient_strategy != "info_nce":
            errors.append("embedding gradient strategy must remain info_nce")
        supported_backbones = {"qwen2_vl", "qwen3_5_vl"}
        if self.model_args.model_backbone not in supported_backbones:
            errors.append(
                "HOBIT supports only qwen2_vl and qwen3_5_vl backbones"
            )
        if abs(float(self.model.temperature) - self.hobit_manager.config.temperature) > 1e-12:
            errors.append("InfoNCE and HOBIT batching temperatures must match")
        if os.path.realpath(manifest["model_name"]) != os.path.realpath(
            self.model_args.model_name
        ):
            errors.append("initial cache backbone differs from the training backbone")
        cached_backbone = manifest.get("model_backbone")
        if cached_backbone is None:
            # Preserve compatibility with the already-built Qwen2 HOBIT cache,
            # whose v1 manifest predates the explicit backbone field. New
            # Qwen3.5 caches must always carry the field so cross-backbone
            # cache reuse fails closed.
            if self.model_args.model_backbone != "qwen2_vl":
                errors.append("initial cache does not identify its model backbone")
        elif cached_backbone != self.model_args.model_backbone:
            errors.append(
                "initial cache model backbone differs from training: "
                f"cache={cached_backbone!r} "
                f"training={self.model_args.model_backbone!r}"
            )
        config_sha = hashlib.sha256(Path(self.data_args.dataset_config).read_bytes()).hexdigest()
        if manifest.get("dataset_config_sha256") != config_sha:
            errors.append("dataset configuration differs from the HOBIT cache")
        if manifest.get("pooling") != self.model.pooling:
            errors.append("pooling differs from the HOBIT cache")
        if bool(manifest.get("normalize")) != bool(self.model.normalize):
            errors.append("normalization differs from the HOBIT cache")
        model_config = self.model.config
        text_config = getattr(model_config, "text_config", None)
        model_hidden_size = getattr(
            text_config,
            "hidden_size",
            getattr(model_config, "hidden_size", None),
        )
        if model_hidden_size is None:
            errors.append("could not determine the training backbone hidden size")
        elif int(manifest.get("dimension", -1)) != int(model_hidden_size):
            errors.append(
                "initial cache embedding dimension differs from the training "
                f"backbone: cache={manifest.get('dimension')!r} "
                f"training={model_hidden_size!r}"
            )
        if int(manifest.get("world_size", -1)) != self.hobit_manager.world:
            errors.append("world size differs from the HOBIT cache")
        if self.model_args.model_backbone == "qwen2_vl":
            if manifest.get("qwen2_model_impl") != self.model_args.qwen2_model_impl:
                errors.append("Qwen2 model implementation differs from the HOBIT cache")
            if manifest.get("qwen2_b3_attention_impl") != self.model_args.qwen2_b3_attention_impl:
                errors.append("Qwen2 attention implementation differs from the HOBIT cache")
            if bool(manifest.get("qwen2_b3_legacy_position_ids")) != bool(
                self.model_args.qwen2_b3_legacy_position_ids
            ):
                errors.append("Qwen2 position-ID mode differs from the HOBIT cache")
        if not bool(manifest.get("resize_use_processor")):
            errors.append("HOBIT cache did not use processor-controlled resizing")
        if int(manifest.get("resize_min_pixels", -1)) != int(
            self.data_args.resize_min_pixels
        ):
            errors.append("minimum image pixels differ from the HOBIT cache")
        if int(manifest.get("resize_max_pixels", -1)) != int(
            self.data_args.resize_max_pixels
        ):
            errors.append("maximum image pixels differ from the HOBIT cache")
        if bool(manifest.get("init_lora_model")):
            errors.append("HOBIT cache was incorrectly built from a teacher adapter")
        if len(self.train_dataset) != int(manifest.get("num_cases", -1)):
            errors.append("training dataset length differs from the HOBIT cache")
        if self.args.resume_from != "none":
            errors.append(
                "resume is disabled until rolling HOBIT-C cache state is checkpointed"
            )
        if errors:
            raise RuntimeError("HOBIT training contract failed: " + "; ".join(errors))
        if self.hobit_manager.rank == 0:
            config = {
                "method": "HOBIT-C",
                "backbone": self.model_args.model_backbone,
                "model": os.path.realpath(self.model_args.model_name),
                "batch_construction": "source_aware_seed_plus_submodular_greedy",
                "global_batch_size": int(self.args.per_device_train_batch_size)
                * self.hobit_manager.world,
                "source_block_size": self.hobit_manager.config.block_size,
                "source_blocks_per_global_batch": (
                    int(self.args.per_device_train_batch_size)
                    * self.hobit_manager.world
                    // self.hobit_manager.config.block_size
                ),
                "seed_size": self.hobit_manager.config.seed_size,
                "candidate_k": self.hobit_manager.config.candidate_k,
                "alpha": self.hobit_manager.config.alpha,
                "batch_temperature": self.hobit_manager.config.temperature,
                "info_nce_temperature": self.model.temperature,
                "candidate_retrieval": "per_seed_query_target_topk",
                "hardness": "q_i_dot_d_j-alpha*d_i_dot_d_j",
                "cache": "initial_student_then_rolling_training_forward",
                "loss": "unchanged_cross_device_info_nce",
            }
            encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
            print(
                "[HOBIT_REPRO_CONFIG] "
                f"sha256={hashlib.sha256(encoded.encode()).hexdigest()} values={encoded}",
                flush=True,
            )

    def _on_reps_captured(self, all_reps):
        if self._hobit_pending_ids is None:
            raise RuntimeError("HOBIT captured representations without matching case IDs")
        query_reps, target_reps = all_reps
        self.hobit_manager.update_cache(
            self._hobit_pending_ids,
            query_reps.detach(),
            target_reps.detach(),
            int(self.state.global_step) + 1,
        )
        super()._on_reps_captured(all_reps)

    def training_step(self, model, inputs, *args, **kwargs):
        queries, _ = inputs
        local_ids = list(map(int, queries.pop(_IDS_KEY)))
        if len(local_ids) != int(self.args.per_device_train_batch_size):
            raise RuntimeError(
                "HOBIT collator produced an unexpected local batch: "
                f"{len(local_ids)} != {self.args.per_device_train_batch_size}"
            )
        if not self._hobit_layout_verified:
            local_tensor = torch.tensor(local_ids, dtype=torch.int64, device=self.args.device)
            if dist.is_initialized():
                gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, local_tensor.contiguous())
                actual = torch.cat(gathered).cpu().numpy()
            else:
                actual = local_tensor.cpu().numpy()
            step = int(self.state.global_step) + 1
            expected = self.hobit_manager.expected_batches.get(step)
            actual_sha = hashlib.sha256(
                np.sort(actual).astype(np.int64, copy=False).tobytes()
            ).hexdigest()
            valid = (
                expected is not None
                and len(actual) == expected["count"]
                and actual_sha == expected["sorted_sha256"]
            )
            valid_tensor = torch.tensor(int(valid), device=self.args.device)
            if dist.is_initialized():
                dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
            if not bool(valid_tensor.item()):
                raise RuntimeError(
                    "DataLoader/DDP sharding changed the HOBIT global batch IDs: "
                    f"step={step} actual_count={len(actual)} expected={expected} "
                    f"actual_sha256={actual_sha}"
                )
            if self.hobit_manager.rank == 0:
                print(
                    "[HOBIT_DDP_LAYOUT_OK] "
                    f"step={step} global_count={len(actual)} unique={np.unique(actual).size} "
                    f"sorted_sha256={actual_sha[:20]}",
                    flush=True,
                )
            self._hobit_layout_verified = True
        self._hobit_pending_ids = local_ids
        try:
            return super().training_step(model, inputs, *args, **kwargs)
        finally:
            self._hobit_pending_ids = None
