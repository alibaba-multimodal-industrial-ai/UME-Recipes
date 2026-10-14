"""Isolated online sampler/trainer for per-dataset dynamic hard batches."""

import json
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from src.training.data.collator import MultimodalDataCollator
from src.training.trainer import GradCacheLateProcessTrainer


_IDS_KEY = "_dynamic_hard_batch_corpus_ids"
_MANAGER = None


class DynamicHardBatchManager:
    def __init__(self):
        self.cache_dir = Path(os.environ["DHBG_RUNTIME_CACHE_DIR"])
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        if not self.manifest.get("complete") or self.manifest.get("version") != 1:
            raise RuntimeError("dynamic hard-batch cache is incomplete or incompatible")
        self.query = np.load(self.cache_dir / "query_embeddings.npy", mmap_mode="r+")
        self.positive = np.load(self.cache_dir / "positive_embeddings.npy", mmap_mode="r+")
        expected = (int(self.manifest["num_cases"]), int(self.manifest["dimension"]))
        if self.query.shape != expected or self.positive.shape != expected:
            raise RuntimeError(f"dynamic hard-batch cache shape mismatch: {self.query.shape}/{self.positive.shape} != {expected}")
        self.updated_at = np.zeros(expected[0], dtype=np.int32)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.candidate_multiplier = int(os.environ.get("DHBG_CANDIDATE_MULTIPLIER", self.manifest["candidate_multiplier"]))
        self.search_chunk = int(os.environ.get("DHBG_SEARCH_CHUNK", "16384"))
        self.log_path = Path(os.environ.get("DHBG_LOG_PATH", "dynamic_hard_batch_steps.jsonl"))

    @property
    def schedule(self):
        return self.manifest["dataset_schedule"]

    def _construct_rank0(self, step, dataset_id, seed_local, batch_size):
        partition = self.manifest["dataset_partitions"][dataset_id]
        offset, length = int(partition["offset"]), int(partition["length"])
        seed = offset + int(seed_local)
        device = torch.device("cuda", torch.cuda.current_device())
        q_seed = torch.from_numpy(np.asarray(self.query[seed], dtype=np.float32)).to(device)
        p_seed = torch.from_numpy(np.asarray(self.positive[seed], dtype=np.float32)).to(device)
        k = min(self.candidate_multiplier * batch_size, length - 1)
        best_scores = torch.empty(0, device=device)
        best_ids = torch.empty(0, dtype=torch.long, device=device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        retrieval_started = started
        with torch.no_grad():
            for local_start in range(0, length, self.search_chunk):
                local_end = min(local_start + self.search_chunk, length)
                q = torch.from_numpy(np.asarray(self.query[offset + local_start:offset + local_end])).to(device)
                p = torch.from_numpy(np.asarray(self.positive[offset + local_start:offset + local_end])).to(device)
                scores = (p.float() @ q_seed) + (q.float() @ p_seed)
                ids = torch.arange(local_start, local_end, device=device)
                scores[ids == seed_local] = -torch.inf
                best_scores = torch.cat((best_scores, scores))
                best_ids = torch.cat((best_ids, ids))
                if best_scores.numel() > 2 * k:
                    best_scores, slots = torch.topk(best_scores, k)
                    best_ids = best_ids[slots]
            best_scores, slots = torch.topk(best_scores, k)
            candidates = best_ids[slots]
            q_candidates = torch.from_numpy(np.asarray(self.query[offset + candidates.cpu().numpy()])).to(device).float()
            p_candidates = torch.from_numpy(np.asarray(self.positive[offset + candidates.cpu().numpy()])).to(device).float()
            torch.cuda.synchronize(device)
            retrieval_seconds = time.perf_counter() - retrieval_started

            # Keep the entire greedy loop on GPU. Converting argmax to a Python
            # integer here would force 1023 device synchronizations per step.
            greedy_started = time.perf_counter()
            gain = best_scores.float().clone()
            available = torch.ones(k, dtype=torch.bool, device=device)
            selected_tensor = torch.empty(batch_size, dtype=torch.long, device=device)
            selected_tensor[0] = int(seed_local)
            for selected_position in range(1, batch_size):
                slot = torch.argmax(gain.masked_fill(~available, -torch.inf))
                selected_tensor[selected_position] = candidates[slot]
                available[slot] = False
                gain.add_(q_candidates[slot] @ p_candidates.T)
                gain.add_(q_candidates @ p_candidates[slot])
            torch.cuda.synchronize(device)
            greedy_seconds = time.perf_counter() - greedy_started

            diagnostics_started = time.perf_counter()
            q_selected = torch.from_numpy(np.asarray(self.query[offset + selected_tensor.cpu().numpy()])).to(device).float()
            p_selected = torch.from_numpy(np.asarray(self.positive[offset + selected_tensor.cpu().numpy()])).to(device).float()
            cross = q_selected @ p_selected.T
            offdiag = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            offdiag_values = cross[offdiag]
            nearest_wrong = cross.masked_fill(~offdiag, -torch.inf).max(dim=1).values
            candidate_summary = [float(best_scores.mean()), float(torch.quantile(best_scores, .5)), float(torch.quantile(best_scores, .9))]
            batch_summary = [float(offdiag_values.mean()), float(torch.quantile(offdiag_values, .5)), float(torch.quantile(offdiag_values, .9)), float(nearest_wrong.mean())]
            selected_local = selected_tensor.cpu().numpy().astype(np.int64, copy=False)
            torch.cuda.synchronize(device)
            diagnostics_seconds = time.perf_counter() - diagnostics_started
        global_ids = selected_local + offset
        elapsed = time.perf_counter() - started
        staleness = self.updated_at[global_ids]
        record = {
            "step": int(step), "dataset": partition["name"], "seed_local": int(seed_local),
            "candidate_k": int(k), "candidate_hardness_mean_p50_p90": candidate_summary,
            "batch_cross_similarity_mean_p50_p90_nearest_wrong_mean": batch_summary,
            "selected_cache_staleness_mean_max": [float((step - staleness).mean()), int((step - staleness).max())],
            "retrieval_seconds": retrieval_seconds,
            "greedy_seconds": greedy_seconds,
            "diagnostics_seconds": diagnostics_seconds,
            "construction_seconds": elapsed,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if os.environ.get("DHBG_DEBUG", os.environ.get("B3_DEBUG", "1")) == "1":
            print("[B3_DEBUG][DYNAMIC_HARD_BATCH] " + " ".join(f"{k}={v}" for k, v in record.items()), flush=True)
        return global_ids.tolist()

    def construct(self, step, dataset_id, seed_local, batch_size):
        result = [None]
        if self.rank == 0:
            result[0] = self._construct_rank0(step, dataset_id, seed_local, batch_size)
        if dist.is_initialized():
            dist.broadcast_object_list(result, src=0)
        return result[0]

    def update(self, step, global_ids, query_reps, positive_reps):
        if self.rank == 0:
            ids = np.asarray(global_ids, dtype=np.int64)
            self.query[ids] = query_reps.detach().float().cpu().numpy().astype(np.float16)
            self.positive[ids] = positive_reps.detach().float().cpu().numpy().astype(np.float16)
            self.updated_at[ids] = int(step)
            self.query.flush(); self.positive.flush()
        if dist.is_initialized():
            dist.barrier()


def get_manager():
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = DynamicHardBatchManager()
    return _MANAGER


class DynamicHardBatchSampler(torch.utils.data.Sampler[int]):
    """Drop-in replacement used only by the dedicated entry point."""
    def __init__(self, total_batch_size, ds_lens, probabilities, source_block_size, num_steps, data_source, seed=42, **kwargs):
        self.total_batch_size = int(total_batch_size)
        self.num_steps = int(num_steps)
        self.data_source = data_source
        self.manager = get_manager()
        if list(map(int, ds_lens)) != [int(x) for x in self.manager.manifest["dataset_lengths"].values()]:
            raise RuntimeError("runtime dataset lengths differ from dynamic hard-batch cache")
        if self.num_steps != int(self.manager.manifest["max_steps"]):
            raise RuntimeError("max_steps differs from dynamic hard-batch cache")
        if self.total_batch_size != int(self.manager.manifest["global_batch_size"]):
            raise RuntimeError("global batch size differs from dynamic hard-batch cache")

    def __len__(self):
        return self.total_batch_size * self.num_steps

    def __iter__(self):
        positions = [0] * len(self.manager.manifest["seeds"])
        for step, dataset_id in enumerate(self.manager.schedule, start=1):
            dataset_id = int(dataset_id)
            pos = positions[dataset_id]
            seeds = self.manager.manifest["seeds"][dataset_id]
            seed = int(seeds[pos % len(seeds)])
            positions[dataset_id] += 1
            yield from self.manager.construct(step, dataset_id, seed, self.total_batch_size)


class DynamicHardBatchCollator(MultimodalDataCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offsets = get_manager().manifest["dataset_offsets"]

    def __call__(self, examples):
        ids = [self.offsets[e["global_dataset_name"]] + int(e["idx"]) for e in examples]
        queries, positives = super().__call__(examples)
        queries[_IDS_KEY] = ids
        return queries, positives


class DynamicHardBatchTrainer(GradCacheLateProcessTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_args.negative_strategy != "vlm2vec_inbatch" or self.args.embedding_gradient_strategy != "info_nce":
            raise ValueError("dynamic hard batches require ordinary vlm2vec_inbatch InfoNCE")
        self.dynamic_manager = get_manager()
        self._dynamic_local_ids = None
        manifest = self.dynamic_manager.manifest
        errors = []
        config_sha = hashlib.sha256(Path(self.data_args.dataset_config).read_bytes()).hexdigest()
        if manifest.get("dataset_config_sha256") != config_sha:
            errors.append("dataset config SHA256 differs")
        if os.path.realpath(manifest.get("model_name", "")) != os.path.realpath(self.model_args.model_name):
            errors.append("base model differs")
        if manifest.get("pooling") != self.model.pooling:
            errors.append("pooling differs")
        if bool(manifest.get("normalize")) != bool(self.model.normalize):
            errors.append("normalization differs")
        if int(manifest.get("seed", -1)) != int(self.args.seed):
            errors.append("seed differs")
        if int(manifest.get("world_size", -1)) != self.dynamic_manager.world:
            errors.append("world size differs")
        if len(self.train_dataset) != int(manifest.get("num_cases", -1)):
            errors.append("training corpus length differs")
        if errors:
            raise RuntimeError("dynamic hard-batch cache is incompatible: " + "; ".join(errors))

    def training_step(self, model, inputs, *args, **kwargs):
        queries, _ = inputs
        self._dynamic_local_ids = list(map(int, queries.pop(_IDS_KEY)))
        return super().training_step(model, inputs, *args, **kwargs)

    def _on_reps_captured(self, all_reps):
        super()._on_reps_captured(all_reps)
        if self._dynamic_local_ids is None:
            raise RuntimeError("dynamic hard-batch IDs were not set")
        query, positive = (rep.detach() for rep in all_reps)
        if dist.is_initialized():
            gathered_q = [torch.empty_like(query) for _ in range(dist.get_world_size())]
            gathered_p = [torch.empty_like(positive) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_q, query.contiguous())
            dist.all_gather(gathered_p, positive.contiguous())
            gathered_ids = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_ids, self._dynamic_local_ids)
            query, positive = torch.cat(gathered_q), torch.cat(gathered_p)
            ids = [item for rank_ids in gathered_ids for item in rank_ids]
        else:
            ids = self._dynamic_local_ids
        self.dynamic_manager.update(self.state.global_step + 1, ids, query, positive)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        config = {
            "method": "per_dataset_dynamic_hard_batch_greedy",
            "loss": "unchanged_cross_device_info_nce",
            "candidate_multiplier": self.dynamic_manager.candidate_multiplier,
            "cache_manifest": os.path.realpath(self.dynamic_manager.cache_dir / "manifest.json"),
        }
        (Path(output_dir) / "dynamic_hard_batch_config.json").write_text(json.dumps(config, indent=2) + "\n")
