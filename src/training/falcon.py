"""Isolated FALCON batch construction for Qwen2-VL MMEB training.

FALCON's actor and quantile-chain sampler follow the official implementation.
The agreed MMEB adaptation is deliberately narrow: each source owns its actor
and deterministic rotating probe stream, and the unavailable MLM reward is
replaced by the before/after improvement of a teacher-filtered probe InfoNCE
loss.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributions import Beta

from src.training.data.collator import (
    MultimodalDataCollator,
    process_vlm_inputs_fns,
)
from src.training.falcon_core import (
    FALCONConfig,
    SimbaPolicyQuantile,
    construct_quantile_chain,
    falcon_similarity_state,
    safe_probe_info_nce,
    safe_probe_mask_from_csr,
)
from src.training.hobit_core import rank_major_global_batch
from src.training.trainer import GradCacheLateProcessTrainer


_IDS_KEY = "_falcon_global_case_ids"
_MANAGER = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FALCONManager:
    """Own per-source actors, search spaces and rolling student features."""

    def __init__(self):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device(
            "cuda", int(os.environ.get("LOCAL_RANK", "0"))
        ) if torch.cuda.is_available() else torch.device("cpu")
        self.student_dir = Path(os.environ["FALCON_STUDENT_RUNTIME_CACHE_DIR"])
        self.probe_graph_dir = Path(os.environ["FALCON_PROBE_GRAPH_RUNTIME_DIR"])
        self.student_manifest = json.loads(
            (self.student_dir / "manifest.json").read_text()
        )
        self.graph_manifest = json.loads(
            (self.probe_graph_dir / "manifest.json").read_text()
        )
        if not self.student_manifest.get("complete"):
            raise RuntimeError("FALCON initial student cache is incomplete")
        if self.student_manifest.get("method") != "falcon_initial_student_embeddings":
            raise RuntimeError("FALCON student cache has the wrong owner")
        if not self.graph_manifest.get("complete"):
            raise RuntimeError("FALCON probe graph cache is incomplete")
        if self.graph_manifest.get("method") != "b3_safe_smooth_rank_full_dataset_batch_greedy":
            raise RuntimeError("FALCON probe cache is not the verified B3-safe graph")

        self.config = FALCONConfig(
            source_block_size=int(os.environ.get("FALCON_SOURCE_BLOCK_SIZE", "512")),
            search_space_size=int(os.environ.get("FALCON_SEARCH_SPACE_SIZE", "5632")),
            num_downsample=int(os.environ.get("FALCON_NUM_DOWNSAMPLE", "100")),
            policy_hidden_dim=int(os.environ.get("FALCON_POLICY_HIDDEN_DIM", "256")),
            policy_residual_blocks=int(os.environ.get("FALCON_POLICY_RESIDUAL_BLOCKS", "2")),
            policy_learning_rate=float(os.environ.get("FALCON_POLICY_LR", "1e-4")),
            policy_update_frequency=int(os.environ.get("FALCON_POLICY_UPDATE_FREQ", "10")),
            probe_global_size=int(os.environ.get("FALCON_PROBE_GLOBAL_SIZE", "64")),
            temperature=float(os.environ.get("FALCON_TEMPERATURE", "0.02")),
        )
        self.config.validate(world_size=self.world)
        self.partitions = self.student_manifest["dataset_partitions"]
        self.offsets = self.student_manifest["dataset_offsets"]
        graph_partitions = self.graph_manifest["dataset_partitions"]
        if len(graph_partitions) != len(self.partitions):
            raise RuntimeError("FALCON student/probe cache source count differs")
        for student, graph in zip(self.partitions, graph_partitions):
            if (
                student["name"] != graph["name"]
                or int(student["offset"]) != int(graph["offset"])
                or int(student["length"]) != int(graph["length"])
            ):
                raise RuntimeError(
                    "FALCON student/probe cache partition mismatch: "
                    f"{student} != {graph}"
                )
        self.graph_partitions = graph_partitions
        self.name_to_source = {
            partition["name"]: int(partition["dataset_id"])
            for partition in self.partitions
        }

        self.query_cache = self.target_cache = self.last_update = None
        init_status = [None]
        if self.rank == 0:
            try:
                self.query_cache = np.load(
                    self.student_dir / "query_embeddings.npy", mmap_mode="r+"
                )
                self.target_cache = np.load(
                    self.student_dir / "positive_embeddings.npy", mmap_mode="r+"
                )
                expected = (
                    int(self.student_manifest["num_cases"]),
                    int(self.student_manifest["dimension"]),
                )
                if (
                    self.query_cache.shape != expected
                    or self.target_cache.shape != expected
                ):
                    raise RuntimeError(
                        "FALCON student cache shape mismatch: "
                        f"{self.query_cache.shape}/{self.target_cache.shape} != {expected}"
                    )
                self.last_update = np.full(expected[0], -1, dtype=np.int32)
            except Exception as exc:
                init_status[0] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(init_status, src=0, device=self.device)
        if init_status[0] is not None:
            raise RuntimeError(
                f"FALCON rolling student cache initialization failed: {init_status[0]}"
            )

        self.graphs = []
        graph_error = None
        try:
            for partition in self.graph_partitions:
                graph = {
                    "indptr": np.load(
                        self.probe_graph_dir / partition["indptr_file"], mmap_mode="r"
                    ),
                    "indices": np.load(
                        self.probe_graph_dir / partition["indices_file"], mmap_mode="r"
                    ),
                    "delta": np.load(
                        self.probe_graph_dir / partition["delta_file"], mmap_mode="r"
                    ),
                    "rank_start": int(partition["rank_start"]),
                }
                length = int(partition["length"])
                if graph["indptr"].shape != (length + 1,):
                    raise RuntimeError(
                        f"FALCON graph indptr shape mismatch for {partition['name']}"
                    )
                if graph["indices"].shape != graph["delta"].shape:
                    raise RuntimeError(
                        f"FALCON graph index/value shape mismatch for {partition['name']}"
                    )
                if int(graph["indptr"][-1]) != len(graph["indices"]):
                    raise RuntimeError(
                        f"FALCON graph indptr terminal mismatch for {partition['name']}"
                    )
                if graph["rank_start"] not in {30, 70}:
                    raise RuntimeError(
                        f"FALCON graph has a non-B3 safety prefix for {partition['name']}: "
                        f"{graph['rank_start']}"
                    )
                self.graphs.append(graph)
        except Exception as exc:
            graph_error = exc
        graph_message = (
            None
            if graph_error is None
            else f"rank={self.rank} {type(graph_error).__name__}: {graph_error}"
        )
        if dist.is_initialized():
            graph_messages = [None for _ in range(self.world)]
            dist.all_gather_object(graph_messages, graph_message)
        else:
            graph_messages = [graph_message]
        graph_failures = [value for value in graph_messages if value is not None]
        if graph_failures:
            raise RuntimeError(
                "FALCON synchronized probe-graph initialization failure: "
                + " | ".join(graph_failures)
            )

        self.policies = []
        self.policy_optimizers = []
        base_seed = int(self.student_manifest["seed"])
        for source_id in range(len(self.partitions)):
            devices = [self.device.index] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(base_seed + 104729 * source_id)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(base_seed + 104729 * source_id)
                policy = SimbaPolicyQuantile(
                    input_dim=self.config.num_downsample,
                    hidden_dim=self.config.policy_hidden_dim,
                    num_residual_blocks=self.config.policy_residual_blocks,
                    output_dim=1,
                ).to(self.device)
            self.policies.append(policy)
            self.policy_optimizers.append(
                torch.optim.AdamW(
                    policy.parameters(), lr=self.config.policy_learning_rate
                )
            )
        self._verify_policy_initialization()

        self.source_orders: dict[int, np.ndarray] = {}
        self.source_cursors = [0] * len(self.partitions)
        self.source_epochs = [0] * len(self.partitions)
        self.probe_orders: dict[int, np.ndarray] = {}
        self.probe_cursors = [0] * len(self.partitions)
        self.probe_epochs = [0] * len(self.partitions)
        self.search_contexts: dict[int, dict] = {}
        self.cached_parameters: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.pending_actions: dict[int, dict[int, list[dict]]] = {}
        self.expected_batches: dict[int, dict] = {}
        self.current_step = 0
        self.update_count = 0
        self.log_path = Path(
            os.environ.get(
                "FALCON_DIAGNOSTICS_PATH", "falcon_diagnostics.jsonl"
            )
        )
        if self.rank == 0:
            print(
                "[FALCON_SOURCE_CONTRACT_OK] "
                f"sources={len(self.partitions)} "
                f"names={[p['name'] for p in self.partitions]} "
                f"lengths={[int(p['length']) for p in self.partitions]} "
                f"probe_unsafe_prefixes={[g['rank_start'] for g in self.graphs]}",
                flush=True,
            )

    def start_diagnostics(self) -> None:
        """Create run-owned diagnostics only after every trainer check passes."""

        if self.rank != 0:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_record({
            "event": "falcon_runtime_start",
            "config": self.config.__dict__,
            "sources": [p["name"] for p in self.partitions],
            "source_rank_starts": [g["rank_start"] for g in self.graphs],
            "policy_scope": "independent_actor_per_mmeb_source",
            "reward": "same_source_rotating_probe_safe_info_nce_improvement",
        })

    def _verify_policy_initialization(self) -> None:
        for source_id, policy in enumerate(self.policies):
            flat = torch.cat(
                [parameter.detach().float().reshape(-1).cpu() for parameter in policy.parameters()]
            )
            checksum = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
            checksums = [None for _ in range(self.world)]
            if dist.is_initialized():
                dist.all_gather_object(checksums, checksum)
            else:
                checksums = [checksum]
            if len(set(checksums)) != 1:
                raise RuntimeError(
                    f"FALCON policy {source_id} initialized differently across ranks"
                )
        if self.rank == 0:
            print(
                "[FALCON_POLICY_INIT_OK] "
                f"sources={len(self.policies)} architecture=100-256-residualx2-2 "
                "beta_parameterization=softplus_plus_1e-5 synchronized=True",
                flush=True,
            )

    def _write_record(self, record: dict) -> None:
        if self.rank != 0:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _raise_synchronized_rank0_error(self, stage: str, rank0_error) -> None:
        status = [
            None
            if rank0_error is None
            else f"{type(rank0_error).__name__}: {rank0_error}"
        ]
        if dist.is_initialized():
            dist.broadcast_object_list(status, src=0, device=self.device)
        if status[0] is not None:
            raise RuntimeError(
                f"FALCON synchronized rank-0 failure at {stage}: {status[0]}"
            )

    def is_policy_update_step(self, step: int) -> bool:
        # Official code uses zero-based i % 10 == 0, hence steps 1, 11, 21, ...
        return (int(step) - 1) % self.config.policy_update_frequency == 0

    def _next_search_ids(self, source_id: int, avoid_global=None) -> np.ndarray:
        partition = self.partitions[source_id]
        offset, length = int(partition["offset"]), int(partition["length"])
        required = self.config.search_space_size
        if length < required:
            raise ValueError(
                f"FALCON source {partition['name']} has {length} rows, fewer than "
                f"search space {required}"
            )
        avoid_local = set()
        if avoid_global is not None:
            avoid_local = {int(value) - offset for value in avoid_global}
        selected = []
        selected_set = set()
        while len(selected) < required:
            order = self.source_orders.get(source_id)
            cursor = self.source_cursors[source_id]
            if order is None or cursor >= len(order):
                epoch = self.source_epochs[source_id]
                rng = np.random.default_rng(
                    int(self.student_manifest["seed"]) + 1_000_003 * source_id + epoch
                )
                order = rng.permutation(length).astype(np.int64, copy=False)
                self.source_orders[source_id] = order
                self.source_cursors[source_id] = 0
                self.source_epochs[source_id] += 1
                cursor = 0
            candidate = int(order[cursor])
            self.source_cursors[source_id] = cursor + 1
            if candidate in selected_set or candidate in avoid_local:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
        return np.asarray(selected, dtype=np.int64) + offset

    def _broadcast_tensor(self, tensor, shape, dtype, src=0):
        if not dist.is_initialized():
            return tensor
        if self.rank != src:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
        else:
            # Distribution samples (notably Beta.sample()) can be strided
            # views. NCCL collectives require dense contiguous storage even
            # when dtype/device already match and Tensor.to() is therefore a
            # no-op.
            tensor = tensor.to(device=self.device, dtype=dtype).contiguous()
        dist.broadcast(tensor, src=src)
        return tensor

    def _new_search(self, source_id: int, avoid_global=None) -> dict:
        search_ids = similarity = state = generation_tensor = None
        search_diagnostics = None
        rank0_error = None
        if self.rank == 0:
            try:
                search_started = time.perf_counter()
                search_ids = self._next_search_ids(source_id, avoid_global)
                print(
                    "[FALCON_SEARCH_SPACE_BEGIN] "
                    f"source={source_id}:{self.partitions[source_id]['name']} "
                    f"size={self.config.search_space_size} "
                    "operation=student_symmetric_similarity_plus_100_quantiles",
                    flush=True,
                )
                query = torch.from_numpy(
                    np.asarray(self.query_cache[search_ids], dtype=np.float32)
                ).to(self.device)
                target = torch.from_numpy(
                    np.asarray(self.target_cache[search_ids], dtype=np.float32)
                ).to(self.device)
                similarity, state = falcon_similarity_state(
                    query, target, self.config.num_downsample
                )
                diagonal = torch.eye(len(similarity), dtype=torch.bool)
                off_diagonal = similarity[~diagonal]
                ages = self.last_update[search_ids]
                search_diagnostics = {
                    "similarity_mean": float(off_diagonal.mean()),
                    "similarity_std": float(off_diagonal.std()),
                    "similarity_min": float(off_diagonal.min()),
                    "similarity_max": float(off_diagonal.max()),
                    "student_cache_fresh_fraction": float(np.mean(ages >= 0)),
                    "student_cache_last_update_min": int(ages.min()),
                    "student_cache_last_update_max": int(ages.max()),
                    "build_seconds": float(time.perf_counter() - search_started),
                }
                generation_tensor = torch.tensor(
                    [self.source_epochs[source_id]], dtype=torch.int64
                )
                del query, target
            except Exception as exc:
                rank0_error = exc
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        self._raise_synchronized_rank0_error("search_space", rank0_error)
        search_ids_tensor = self._broadcast_tensor(
            None if search_ids is None else torch.from_numpy(search_ids),
            (self.config.search_space_size,),
            torch.int64,
        )
        state = self._broadcast_tensor(
            state,
            (self.config.search_space_size, self.config.num_downsample),
            torch.float32,
        )
        generation_tensor = self._broadcast_tensor(
            generation_tensor,
            (1,),
            torch.int64,
        )
        generation = int(generation_tensor.item())
        # Only rank 0 advances the physical source cursor.  Keep the logical
        # generation identical everywhere because it participates in the
        # deterministic action seed and is persisted in policy checkpoints.
        self.source_epochs[source_id] = generation
        context = {
            "global_ids": search_ids_tensor.cpu().numpy(),
            "similarity": similarity,
            "state": state,
            "remaining": torch.arange(self.config.search_space_size),
            "generation": generation,
        }
        self.search_contexts[source_id] = context
        if self.rank == 0:
            print(
                "[FALCON_SEARCH_SPACE] "
                f"source={source_id}:{self.partitions[source_id]['name']} "
                f"generation={context['generation']} size={self.config.search_space_size} "
                f"rank_start={self.graphs[source_id]['rank_start']} "
                f"sim_mean={search_diagnostics['similarity_mean']:.8g} "
                f"sim_std={search_diagnostics['similarity_std']:.8g} "
                f"sim_min={search_diagnostics['similarity_min']:.8g} "
                f"sim_max={search_diagnostics['similarity_max']:.8g} "
                f"cache_fresh={search_diagnostics['student_cache_fresh_fraction']:.8g} "
                f"cache_update_range={search_diagnostics['student_cache_last_update_min']}:"
                f"{search_diagnostics['student_cache_last_update_max']} "
                f"seconds={search_diagnostics['build_seconds']:.3f} "
                f"sha256={hashlib.sha256(search_ids.tobytes()).hexdigest()[:20]}",
                flush=True,
            )
        return context

    def _sample_quantiles(
        self, source_id: int, context: dict, step: int, block_slot: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        policy = self.policies[source_id]
        update_policy = self.is_policy_update_step(step)
        if update_policy or source_id not in self.cached_parameters:
            if update_policy:
                alpha, beta = policy(context["state"])
            else:
                with torch.no_grad():
                    alpha, beta = policy(context["state"])
            self.cached_parameters[source_id] = (alpha.detach(), beta.detach())
        else:
            alpha, beta = self.cached_parameters[source_id]
        distribution = Beta(alpha, beta)
        sampled = None
        rank0_error = None
        if self.rank == 0:
            devices = [self.device.index] if self.device.type == "cuda" else []
            sample_seed = (
                int(self.student_manifest["seed"])
                + 10_000_019 * int(step)
                + 1009 * source_id
                + 17 * int(block_slot)
                + int(context["generation"])
            )
            try:
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(sample_seed)
                    if self.device.type == "cuda":
                        torch.cuda.manual_seed(sample_seed)
                    sampled = distribution.sample()
            except Exception as exc:
                rank0_error = exc
        self._raise_synchronized_rank0_error("quantile_sampling", rank0_error)
        sampled = self._broadcast_tensor(
            sampled,
            (self.config.search_space_size,),
            torch.float32,
        )
        if (
            not torch.isfinite(sampled).all()
            or torch.any(sampled < 0)
            or torch.any(sampled > 1)
        ):
            raise FloatingPointError(
                f"FALCON actor {source_id} sampled an invalid quantile"
            )
        log_prob = None
        if update_policy:
            log_prob = distribution.log_prob(sampled)
            if not torch.isfinite(log_prob).all():
                raise FloatingPointError(
                    f"FALCON actor {source_id} produced non-finite log probabilities"
                )
        diagnostics = {
            "alpha_mean": float(alpha.detach().mean().cpu()),
            "beta_mean": float(beta.detach().mean().cpu()),
            "quantile_mean": None,
            "entropy_mean": None,
        }
        return sampled, log_prob, diagnostics

    def next_source_block(
        self,
        source_id: int,
        *,
        step: int,
        block_slot: int,
        avoid_global=None,
    ) -> np.ndarray:
        context = self.search_contexts.get(source_id)
        if context is None or len(context["remaining"]) < self.config.source_block_size:
            context = self._new_search(source_id, avoid_global)
        sampled, all_log_prob, diagnostics = self._sample_quantiles(
            source_id, context, step, block_slot
        )
        selected_positions = used_rows = None
        chain_similarity_mean = chain_similarity_max = float("nan")
        rank0_error = None
        if self.rank == 0:
            try:
                rng = np.random.default_rng(
                    int(self.student_manifest["seed"])
                    + 97_409 * int(step)
                    + 997 * source_id
                    + block_slot
                )
                start_position = int(rng.integers(len(context["remaining"])))
                selected_positions, used_rows = construct_quantile_chain(
                    context["similarity"],
                    context["remaining"],
                    sampled.cpu(),
                    block_size=self.config.source_block_size,
                    start_position=start_position,
                )
                transitions = context["similarity"][
                    used_rows[:-1], used_rows[1:]
                ]
                chain_similarity_mean = float(transitions.mean())
                chain_similarity_max = float(transitions.max())
            except Exception as exc:
                rank0_error = exc
        self._raise_synchronized_rank0_error("quantile_chain", rank0_error)
        selected_positions = self._broadcast_tensor(
            selected_positions,
            (self.config.source_block_size,),
            torch.int64,
        ).cpu()
        used_rows = self._broadcast_tensor(
            used_rows,
            (self.config.source_block_size,),
            torch.int64,
        ).cpu()
        selected_set = set(map(int, selected_positions.tolist()))
        context["remaining"] = torch.as_tensor(
            [
                int(value)
                for value in context["remaining"].tolist()
                if int(value) not in selected_set
            ],
            dtype=torch.long,
        )
        global_ids = context["global_ids"][selected_positions.numpy()]
        selected_quantiles = sampled.index_select(
            0, used_rows.to(device=sampled.device)
        )
        diagnostics["quantile_mean"] = float(selected_quantiles.detach().mean().cpu())
        action = {
            "block_slot": int(block_slot),
            "used_rows": used_rows.numpy(),
            "quantile_mean": diagnostics["quantile_mean"],
            "alpha_mean": diagnostics["alpha_mean"],
            "beta_mean": diagnostics["beta_mean"],
            "entropy_mean": diagnostics["entropy_mean"],
            "chain_similarity_mean": chain_similarity_mean,
            "chain_similarity_max": chain_similarity_max,
            "log_prob_sum": None,
        }
        if all_log_prob is not None:
            action["log_prob_sum"] = all_log_prob.index_select(
                0, used_rows.to(device=all_log_prob.device)
            ).sum()
            entropy = Beta(
                *self.cached_parameters[source_id]
            ).entropy().index_select(
                0, used_rows.to(device=self.device)
            )
            diagnostics["entropy_mean"] = float(entropy.detach().mean().cpu())
            action["entropy_mean"] = diagnostics["entropy_mean"]
        self.pending_actions.setdefault(int(step), {}).setdefault(
            source_id, []
        ).append(action)
        if self.rank == 0:
            print(
                "[FALCON_SOURCE_BLOCK] "
                f"step={step} slot={block_slot} "
                f"source={source_id}:{self.partitions[source_id]['name']} "
                f"policy_update={self.is_policy_update_step(step)} "
                f"q_mean={diagnostics['quantile_mean']:.8g} "
                f"alpha_mean={diagnostics['alpha_mean']:.8g} "
                f"beta_mean={diagnostics['beta_mean']:.8g} "
                f"entropy_mean={diagnostics['entropy_mean']} "
                f"chain_sim_mean={chain_similarity_mean:.8g} "
                f"chain_sim_max={chain_similarity_max:.8g} "
                f"remaining={len(context['remaining'])} "
                f"sha256={hashlib.sha256(global_ids.tobytes()).hexdigest()[:20]}",
                flush=True,
            )
        return global_ids.astype(np.int64, copy=False)

    def record_expected_batch(self, step: int, global_ids: np.ndarray) -> None:
        global_ids = np.asarray(global_ids, dtype=np.int64)
        if np.unique(global_ids).size != len(global_ids):
            raise RuntimeError(f"FALCON global batch {step} contains duplicate cases")
        self.expected_batches[int(step)] = {
            "count": len(global_ids),
            "sorted_sha256": hashlib.sha256(
                np.sort(global_ids).tobytes()
            ).hexdigest(),
        }
        while len(self.expected_batches) > 8:
            del self.expected_batches[min(self.expected_batches)]

    def bind_training_batch(self, step: int, local_ids: list[int]) -> list[int]:
        local = torch.as_tensor(local_ids, device=self.device, dtype=torch.int64)
        if dist.is_initialized():
            gathered = [torch.empty_like(local) for _ in range(self.world)]
            dist.all_gather(gathered, local)
            global_ids = torch.cat(gathered).cpu().numpy()
        else:
            global_ids = local.cpu().numpy()
        expected = self.expected_batches.get(int(step))
        actual = {
            "count": len(global_ids),
            "sorted_sha256": hashlib.sha256(
                np.sort(global_ids).astype(np.int64, copy=False).tobytes()
            ).hexdigest(),
        }
        if expected != actual:
            raise RuntimeError(
                "FALCON DataLoader/DDP sharding changed the global batch: "
                f"expected={expected} actual={actual}"
            )
        return global_ids.tolist()

    @torch.no_grad()
    def update_student_cache(
        self,
        local_ids: list[int],
        query_reps: torch.Tensor,
        target_reps: torch.Tensor,
        step: int,
    ) -> None:
        ids = torch.as_tensor(local_ids, device=query_reps.device, dtype=torch.int64)
        if len(ids) != len(query_reps) or query_reps.shape != target_reps.shape:
            raise RuntimeError("FALCON cache update IDs/representations mismatch")
        if dist.is_initialized():
            gathered_ids = [torch.empty_like(ids) for _ in range(self.world)]
            gathered_q = [torch.empty_like(query_reps) for _ in range(self.world)]
            gathered_p = [torch.empty_like(target_reps) for _ in range(self.world)]
            dist.all_gather(gathered_ids, ids.contiguous())
            dist.all_gather(gathered_q, query_reps.contiguous())
            dist.all_gather(gathered_p, target_reps.contiguous())
            ids = torch.cat(gathered_ids)
            query_reps = torch.cat(gathered_q)
            target_reps = torch.cat(gathered_p)
        ids_np = ids.cpu().numpy().astype(np.int64, copy=False)
        if np.unique(ids_np).size != ids_np.size:
            raise RuntimeError(f"FALCON step {step} contains duplicate global IDs")
        if np.any(ids_np < 0) or np.any(
            ids_np >= int(self.student_manifest["num_cases"])
        ):
            raise IndexError("FALCON cache update contains an out-of-range global ID")
        if self.rank == 0:
            update_status = [None]
            try:
                query_np = query_reps.float().cpu().numpy().astype(np.float16)
                target_np = target_reps.float().cpu().numpy().astype(np.float16)
                self.query_cache[ids_np] = query_np
                self.target_cache[ids_np] = target_np
                self.last_update[ids_np] = int(step)
                self.update_count += len(ids_np)
                if step == 1 or step % 100 == 0:
                    self.query_cache.flush()
                    self.target_cache.flush()
                    print(
                        "[FALCON_STUDENT_CACHE_UPDATE] "
                        f"step={step} cases={len(ids_np)} total={self.update_count} "
                        f"q_norm={np.linalg.norm(query_np.astype(np.float32), axis=1).mean():.8g} "
                        f"p_norm={np.linalg.norm(target_np.astype(np.float32), axis=1).mean():.8g}",
                        flush=True,
                    )
            except Exception as exc:
                update_status[0] = f"{type(exc).__name__}: {exc}"
        else:
            update_status = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(update_status, src=0, device=query_reps.device)
        if update_status[0] is not None:
            raise RuntimeError(
                f"FALCON rolling student cache update failed: {update_status[0]}"
            )
        self.current_step = int(step)

    def probe_ids_and_mask(
        self, source_id: int, step: int, current_global_ids: list[int]
    ) -> tuple[np.ndarray, torch.Tensor]:
        partition = self.partitions[source_id]
        offset, length = int(partition["offset"]), int(partition["length"])
        forbidden = {
            int(value) - offset
            for value in current_global_ids
            if offset <= int(value) < offset + length
        }
        if length - len(forbidden) < self.config.probe_global_size:
            raise RuntimeError(f"FALCON source {partition['name']} is too small for probe")
        probe_local = mask = None
        rank0_error = None
        if self.rank == 0:
            try:
                chosen = []
                chosen_set = set()
                while len(chosen) < self.config.probe_global_size:
                    order = self.probe_orders.get(source_id)
                    cursor = self.probe_cursors[source_id]
                    if order is None or cursor >= len(order):
                        epoch = self.probe_epochs[source_id]
                        rng = np.random.default_rng(
                            int(self.student_manifest["seed"])
                            + 32_452_843 * source_id
                            + epoch
                        )
                        order = rng.permutation(length).astype(np.int64, copy=False)
                        self.probe_orders[source_id] = order
                        self.probe_cursors[source_id] = 0
                        self.probe_epochs[source_id] += 1
                        cursor = 0
                    candidate = int(order[cursor])
                    self.probe_cursors[source_id] = cursor + 1
                    if candidate in forbidden or candidate in chosen_set:
                        continue
                    chosen.append(candidate)
                    chosen_set.add(candidate)
                probe_local = np.asarray(chosen, dtype=np.int64)
                graph = self.graphs[source_id]
                mask = safe_probe_mask_from_csr(
                    probe_local,
                    graph["indptr"],
                    graph["indices"],
                    graph["delta"],
                ).to(torch.uint8)
            except Exception as exc:
                rank0_error = exc
        self._raise_synchronized_rank0_error("probe_selection", rank0_error)
        probe_local_tensor = self._broadcast_tensor(
            None if probe_local is None else torch.from_numpy(probe_local),
            (self.config.probe_global_size,),
            torch.int64,
        )
        mask = self._broadcast_tensor(
            mask,
            (self.config.probe_global_size, self.config.probe_global_size),
            torch.uint8,
        ).bool().cpu()
        if self.rank == 0:
            safe_counts = mask.sum(dim=1) - 1
            print(
                "[FALCON_PROBE_BATCH] "
                f"step={step} source={source_id}:{partition['name']} "
                f"probe_epoch={self.probe_epochs[source_id]} "
                f"global_probe={len(probe_local)} "
                f"safe_negative_min={int(safe_counts.min())} "
                f"safe_negative_mean={float(safe_counts.float().mean()):.8g} "
                f"safe_negative_max={int(safe_counts.max())} "
                f"masked_negative_fraction="
                f"{float((~mask).sum() / max(1, len(mask) * (len(mask) - 1))):.8g} "
                f"sha256={hashlib.sha256(probe_local.tobytes()).hexdigest()[:20]}",
                flush=True,
            )
        return probe_local_tensor.cpu().numpy() + offset, mask

    def update_policies(self, step: int, rewards: dict[int, dict]) -> None:
        actions_by_source = self.pending_actions.pop(int(step), {})
        if not self.is_policy_update_step(step):
            if rewards:
                raise RuntimeError("FALCON received rewards on a non-policy-update step")
            return
        if set(actions_by_source) != set(rewards):
            raise RuntimeError(
                "FALCON reward/action source mismatch: "
                f"actions={sorted(actions_by_source)} rewards={sorted(rewards)}"
            )
        for source_id, actions in actions_by_source.items():
            log_probs = [action["log_prob_sum"] for action in actions]
            if any(value is None for value in log_probs):
                raise RuntimeError("FALCON update action lost its policy graph")
            reward = rewards[source_id]["reward"].detach()
            if not torch.isfinite(reward):
                raise FloatingPointError(
                    f"FALCON source {source_id} produced a non-finite probe reward"
                )
            optimizer = self.policy_optimizers[source_id]
            optimizer.zero_grad(set_to_none=True)
            actor_loss = -(torch.stack(log_probs).sum() * reward).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError(
                    f"FALCON actor {source_id} produced a non-finite loss"
                )
            actor_loss.backward()
            if dist.is_initialized():
                for parameter in self.policies[source_id].parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad.div_(self.world)
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in self.policies[source_id].parameters()
            ):
                raise FloatingPointError(
                    f"FALCON actor {source_id} produced non-finite gradients"
                )
            grad_norm = torch.linalg.vector_norm(
                torch.stack([
                    parameter.grad.detach().float().norm()
                    for parameter in self.policies[source_id].parameters()
                    if parameter.grad is not None
                ])
            )
            optimizer.step()
            if any(
                not torch.isfinite(parameter).all()
                for parameter in self.policies[source_id].parameters()
            ):
                raise FloatingPointError(
                    f"FALCON actor {source_id} became non-finite after AdamW"
                )
            self.cached_parameters.pop(source_id, None)
            record = {
                "event": "falcon_policy_update",
                "step": int(step),
                "source_id": int(source_id),
                "source": self.partitions[source_id]["name"],
                "blocks": len(actions),
                "reward": float(reward.cpu()),
                "probe_loss_before": rewards[source_id]["before"],
                "probe_loss_after": rewards[source_id]["after"],
                "actor_loss": float(actor_loss.detach().cpu()),
                "quantile_mean": float(np.mean([a["quantile_mean"] for a in actions])),
                "alpha_mean": float(np.mean([a["alpha_mean"] for a in actions])),
                "beta_mean": float(np.mean([a["beta_mean"] for a in actions])),
                "entropy_mean": float(np.mean([a["entropy_mean"] for a in actions])),
                "log_probability_sum": float(
                    torch.stack(log_probs).sum().detach().cpu()
                ),
                "policy_gradient_norm": float(grad_norm.detach().cpu()),
                "shared_reward_across_blocks": len(actions) > 1,
                "chain_similarity_mean": float(
                    np.mean([a["chain_similarity_mean"] for a in actions])
                ),
                "chain_similarity_max": float(
                    np.max([a["chain_similarity_max"] for a in actions])
                ),
                **rewards[source_id]["diagnostics"],
            }
            self._write_record(record)
            if self.rank == 0:
                print(
                    "[FALCON_POLICY_UPDATE] "
                    f"step={step} source={source_id}:{record['source']} "
                    f"blocks={len(actions)} reward={record['reward']:.8g} "
                    f"probe_before={record['probe_loss_before']:.8g} "
                    f"probe_after={record['probe_loss_after']:.8g} "
                    f"q_mean={record['quantile_mean']:.8g} "
                    f"entropy_mean={record['entropy_mean']:.8g} "
                    f"grad_norm={record['policy_gradient_norm']:.8g} "
                    f"shared_reward={record['shared_reward_across_blocks']} "
                    f"actor_loss={record['actor_loss']:.8g}",
                    flush=True,
                )

    def discard_nonupdate_actions(self, step: int) -> None:
        """Release detached sampler diagnostics on ordinary VLM update steps."""

        if self.is_policy_update_step(step):
            raise RuntimeError("Cannot discard FALCON actions on a policy-update step")
        actions = self.pending_actions.pop(int(step), None)
        if not actions:
            raise RuntimeError(f"FALCON step {step} has no sampler actions to discard")

    def save_policy_state(self, output_dir: str) -> None:
        if self.rank != 0:
            return
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({
            "method": "FALCON-MMEB-probe-adaptation",
            "config": self.config.__dict__,
            "policies": [policy.state_dict() for policy in self.policies],
            "policy_optimizers": [optimizer.state_dict() for optimizer in self.policy_optimizers],
            "source_epochs": self.source_epochs,
            "source_cursors": self.source_cursors,
            "probe_epochs": self.probe_epochs,
            "probe_cursors": self.probe_cursors,
            "current_step": self.current_step,
        }, output / "falcon_policy_state.pt")
        config = {
            "method": "FALCON-MMEB-probe-adaptation",
            "official_components": [
                "SimbaPolicyQuantile_100x256_residual2",
                "symmetric_i2t_plus_t2i_similarity",
                "softmax_quantile_state",
                "beta_quantile_sampling",
                "previous_selected_row_chain",
                "policy_update_every_10_steps",
                "reinforce_log_probability_sum",
            ],
            "necessary_adaptations": {
                "policy_scope": "one independent official actor per MMEB source",
                "reward": "same-source rotating safe-probe InfoNCE before-after improvement",
                "mlm_unavailable": True,
                "source_blocks_per_global_batch": 2,
            },
            "main_training_loss": "unchanged_global_cross_device_info_nce",
            "config": self.config.__dict__,
            "student_cache_manifest_sha256": _sha256_file(
                self.student_dir / "manifest.json"
            ),
            "probe_graph_manifest_sha256": _sha256_file(
                self.probe_graph_dir / "manifest.json"
            ),
        }
        (output / "falcon_training_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def get_falcon_manager() -> FALCONManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = FALCONManager()
    return _MANAGER


class FALCONSampler(torch.utils.data.Sampler[int]):
    """Two-source-block sampler with per-source official FALCON actors."""

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
        self.manager = get_falcon_manager()
        self.source_block_size = int(source_block_size)
        weights = np.asarray(probabilities, dtype=np.float64)
        if len(weights) != len(self.ds_lens) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(f"Invalid FALCON source probabilities: {probabilities}")
        self.probabilities = weights / weights.sum()
        expected_lengths = [int(p["length"]) for p in self.manager.partitions]
        errors = []
        if self.ds_lens != expected_lengths:
            errors.append("dataset lengths differ from FALCON cache")
        too_small = [
            self.manager.partitions[index]["name"]
            for index, length in enumerate(expected_lengths)
            if length < self.manager.config.search_space_size
        ]
        if too_small:
            errors.append(
                "sources smaller than FALCON search space: " + ",".join(too_small)
            )
        if self.source_block_size != self.manager.config.source_block_size:
            errors.append("source block differs from FALCON configuration")
        if self.total_batch_size != 2 * self.source_block_size:
            errors.append("FALCON MMEB run requires exactly two source blocks")
        if self.seed != int(self.manager.student_manifest["seed"]):
            errors.append("sampler seed differs from FALCON cache")
        if errors:
            raise RuntimeError("FALCON sampler contract failed: " + "; ".join(errors))
        self._num_samples = self.total_batch_size * self.num_steps

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        source_ids = rng.choice(
            len(self.ds_lens),
            size=self.num_steps * 2,
            p=self.probabilities,
        )
        if self.manager.rank == 0:
            print(
                "[FALCON_SOURCE_SCHEDULE] "
                f"steps={self.num_steps} blocks={len(source_ids)} "
                f"first_sources={source_ids[:16].tolist()} "
                f"sha256={hashlib.sha256(source_ids.astype(np.int64).tobytes()).hexdigest()[:20]}",
                flush=True,
            )
        for step in range(1, self.num_steps + 1):
            selected_sources = source_ids[(step - 1) * 2:step * 2]
            blocks = []
            selected_by_source: dict[int, np.ndarray] = {}
            for slot, source_value in enumerate(selected_sources):
                source_id = int(source_value)
                block = self.manager.next_source_block(
                    source_id,
                    step=step,
                    block_slot=slot,
                    avoid_global=selected_by_source.get(source_id),
                )
                blocks.append(block)
                selected_by_source[source_id] = np.concatenate((
                    selected_by_source.get(source_id, np.empty(0, dtype=np.int64)),
                    block,
                ))
            global_ids = np.concatenate(blocks).astype(np.int64, copy=False)
            self.manager.record_expected_batch(step, global_ids)
            if self.manager.rank == 0 and (step == 1 or step % 100 == 0):
                print(
                    "[FALCON_GLOBAL_BATCH] "
                    f"step={step} sources={selected_sources.tolist()} "
                    f"global_batch={len(global_ids)} unique={np.unique(global_ids).size} "
                    f"sha256={hashlib.sha256(global_ids.tobytes()).hexdigest()[:20]}",
                    flush=True,
                )
            yield from rank_major_global_batch(global_ids, self.manager.world).tolist()


class FALCONCollator(MultimodalDataCollator):
    def __call__(self, examples):
        manager = get_falcon_manager()
        ids = [
            int(manager.offsets[example["global_dataset_name"]]) + int(example["idx"])
            for example in examples
        ]
        queries, positives = super().__call__(examples)
        queries[_IDS_KEY] = ids
        return queries, positives


class FALCONTrainer(GradCacheLateProcessTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.falcon_manager = get_falcon_manager()
        self._falcon_pending_ids = None
        self._falcon_global_ids = None
        self._falcon_probe_batches = {}
        self._falcon_probe_before = {}
        errors = []
        if self.data_args.negative_strategy != "vlm2vec_inbatch":
            errors.append("negative_strategy must remain vlm2vec_inbatch")
        if self.args.embedding_gradient_strategy != "info_nce":
            errors.append("embedding gradient strategy must remain info_nce")
        if self.model_args.model_backbone != "qwen2_vl":
            errors.append("FALCON entry point is isolated to Qwen2-VL")
        if self.args.gradient_accumulation_steps != 1:
            errors.append("FALCON requires gradient_accumulation_steps=1")
        global_batch = int(self.args.per_device_train_batch_size) * self.falcon_manager.world
        if global_batch != 2 * self.falcon_manager.config.source_block_size:
            errors.append("global batch must contain exactly two FALCON source blocks")
        if abs(float(self.model.temperature) - self.falcon_manager.config.temperature) > 1e-12:
            errors.append("probe and training InfoNCE temperatures differ")
        if self.args.resume_from != "none":
            errors.append("resume is disabled until FALCON search/probe streams are checkpointed")
        student_manifest = self.falcon_manager.student_manifest
        graph_manifest = self.falcon_manager.graph_manifest
        dataset_config_sha256 = hashlib.sha256(
            Path(self.data_args.dataset_config).read_bytes()
        ).hexdigest()
        if student_manifest.get("dataset_config_sha256") != dataset_config_sha256:
            errors.append("initial student cache dataset configuration differs from training")
        if graph_manifest.get("dataset_config_sha256") != dataset_config_sha256:
            errors.append("safe-probe graph dataset configuration differs from training")
        if os.path.realpath(self.model_args.model_name) != student_manifest.get("model_name"):
            errors.append("initial student cache checkpoint differs from training checkpoint")
        if student_manifest.get("pooling") != self.model_args.pooling:
            errors.append("initial student cache pooling differs from training")
        if bool(student_manifest.get("normalize")) != bool(self.model_args.normalize):
            errors.append("initial student cache normalization differs from training")
        if student_manifest.get("qwen2_model_impl") != "b3":
            errors.append("initial student cache was not encoded by the B3 Qwen2 implementation")
        if student_manifest.get("qwen2_b3_attention_impl") != "flash_attention_2":
            errors.append("initial student cache attention implementation differs")
        if bool(student_manifest.get("qwen2_b3_legacy_position_ids")):
            errors.append("initial student cache used legacy Qwen2 position IDs")
        if int(student_manifest.get("resize_min_pixels", -1)) != int(
            self.data_args.resize_min_pixels
        ) or int(student_manifest.get("resize_max_pixels", -1)) != int(
            self.data_args.resize_max_pixels
        ):
            errors.append("initial student cache image pixel bounds differ from training")
        if int(student_manifest.get("seed", -1)) != int(self.args.seed):
            errors.append("initial student cache seed differs from training")
        if int(student_manifest.get("world_size", -1)) != self.falcon_manager.world:
            errors.append("initial student cache world size differs from training")
        if not bool(graph_manifest.get("teacher_init_lora_model")):
            errors.append("safe-probe graph was not built with a trained teacher adapter")
        if graph_manifest.get("student_model_name") != os.path.realpath(
            self.model_args.model_name
        ):
            errors.append("safe-probe graph targets a different Qwen2 student checkpoint")
        teacher_preprocessing = graph_manifest.get("teacher_preprocessing", {})
        if int(teacher_preprocessing.get("resize_min_pixels", -1)) != int(
            self.data_args.resize_min_pixels
        ) or int(teacher_preprocessing.get("resize_max_pixels", -1)) != int(
            self.data_args.resize_max_pixels
        ):
            errors.append("safe-probe teacher vectors used different image pixel bounds")
        if int(graph_manifest.get("rank_depth", -1)) != 256:
            errors.append("safe-probe graph rank depth differs from the controlled value 256")
        if abs(float(graph_manifest.get("rank_alpha", -1)) - 0.03) > 1e-12:
            errors.append("safe-probe graph rank alpha differs from 0.03")
        if int(graph_manifest.get("seed", -1)) != int(self.args.seed):
            errors.append("safe-probe graph seed differs from training")
        if errors:
            raise RuntimeError("FALCON training contract failed: " + "; ".join(errors))
        self.falcon_manager.start_diagnostics()
        if self.falcon_manager.rank == 0:
            print(
                "[FALCON_REPRO_CONFIG] "
                "main_loss=unchanged_global_cross_device_info_nce "
                "global_batch=1024 source_blocks=2x512 "
                "actor=official_100x256_residual2_beta policy_update_frequency=10 "
                "reward=per_source_rotating_safe_probe_info_nce_delta",
                flush=True,
            )

    def _synchronize_probe_error(self, stage: str, local_error) -> None:
        message = (
            None
            if local_error is None
            else f"rank={self.falcon_manager.rank} {type(local_error).__name__}: {local_error}"
        )
        if dist.is_initialized():
            messages = [None for _ in range(self.falcon_manager.world)]
            dist.all_gather_object(messages, message)
        else:
            messages = [message]
        failures = [value for value in messages if value is not None]
        if failures:
            raise RuntimeError(
                f"FALCON synchronized probe failure at {stage}: "
                + " | ".join(failures)
            )

    def training_step(self, model, inputs, *args, **kwargs):
        queries, _ = inputs
        ids = queries.pop(_IDS_KEY)
        step = int(self.state.global_step) + 1
        self._falcon_pending_ids = list(map(int, ids))
        self._falcon_global_ids = self.falcon_manager.bind_training_batch(step, ids)
        return super().training_step(model, inputs, *args, **kwargs)

    def _on_reps_captured(self, all_reps):
        if self._falcon_pending_ids is None:
            raise RuntimeError("FALCON captured representations without case IDs")
        query_reps, target_reps = all_reps
        self.falcon_manager.update_student_cache(
            self._falcon_pending_ids,
            query_reps.detach(),
            target_reps.detach(),
            int(self.state.global_step) + 1,
        )
        super()._on_reps_captured(all_reps)

    def _collate_probe(self, global_ids: np.ndarray):
        local_size = self.falcon_manager.config.probe_global_size // self.falcon_manager.world
        start = self.falcon_manager.rank * local_size
        local_ids = global_ids[start:start + local_size]
        examples = [self.train_dataset[int(index)] for index in local_ids]
        collator = self.data_collator
        query_inputs = collator._get_batch_inputs(examples, "query_text", "query_image")
        target_inputs = collator._get_batch_inputs(examples, "pos_text", "pos_image")
        process_fn = process_vlm_inputs_fns[self.args.model_backbone]
        queries = process_fn(
            query_inputs,
            processor=collator.processor,
            max_length=self.data_args.max_len,
        )
        targets = process_fn(
            target_inputs,
            processor=collator.processor,
            max_length=self.data_args.max_len,
        )
        return self._prepare_inputs((queries, targets))

    @torch.no_grad()
    def _probe_loss(self, model, batch, keep_mask, probe_seed):
        unwrapped = self.accelerator.unwrap_model(model)
        queries, targets = batch
        device_type = next(unwrapped.parameters()).device.type
        autocast_enabled = device_type == "cuda"
        devices = [next(unwrapped.parameters()).device.index] if device_type == "cuda" else []
        # Official FALCON measures reward with the VLM still in train mode.
        # Fork and reseed RNG so before/after see the same LoRA-dropout masks
        # without perturbing the main training random stream.
        local_error = None
        local_query = local_target = None
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(probe_seed))
                if device_type == "cuda":
                    torch.cuda.manual_seed(int(probe_seed))
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    local_query = unwrapped.encode_input(queries)
                    local_target = unwrapped.encode_input(targets)
        except Exception as exc:
            local_error = exc
            if device_type == "cuda":
                torch.cuda.empty_cache()
        self._synchronize_probe_error("forward", local_error)
        if dist.is_initialized():
            gathered_query = [
                torch.empty_like(local_query)
                for _ in range(self.falcon_manager.world)
            ]
            gathered_target = [
                torch.empty_like(local_target)
                for _ in range(self.falcon_manager.world)
            ]
            dist.all_gather(gathered_query, local_query.contiguous())
            dist.all_gather(gathered_target, local_target.contiguous())
            query = torch.cat(gathered_query)
            target = torch.cat(gathered_target)
        else:
            query, target = local_query, local_target
        loss, diagnostics = safe_probe_info_nce(
            query,
            target,
            keep_mask.to(query.device),
            temperature=self.falcon_manager.config.temperature,
        )
        return loss.detach(), diagnostics

    def _before_optimizer_step(self, model):
        super()._before_optimizer_step(model)
        step = int(self.state.global_step) + 1
        if not self.falcon_manager.is_policy_update_step(step):
            self.falcon_manager.discard_nonupdate_actions(step)
            return
        actions = self.falcon_manager.pending_actions.get(step, {})
        if not actions:
            raise RuntimeError(f"FALCON step {step} has no pending policy actions")
        self._falcon_probe_batches = {}
        self._falcon_probe_before = {}
        for source_id in sorted(actions):
            probe_ids, keep_mask = self.falcon_manager.probe_ids_and_mask(
                source_id, step, self._falcon_global_ids
            )
            batch = None
            local_error = None
            try:
                batch = self._collate_probe(probe_ids)
            except Exception as exc:
                local_error = exc
            self._synchronize_probe_error("collation", local_error)
            probe_seed = (
                int(self.falcon_manager.student_manifest["seed"])
                + 49_979_687 * step
                + 67_867_967 * source_id
            )
            loss, diagnostics = self._probe_loss(
                model, batch, keep_mask, probe_seed
            )
            self._falcon_probe_batches[source_id] = (
                batch,
                keep_mask,
                probe_seed,
            )
            self._falcon_probe_before[source_id] = (
                loss,
                diagnostics,
                hashlib.sha256(probe_ids.tobytes()).hexdigest()[:20],
            )

    def _after_optimizer_step(self, model, optimizer_was_run):
        super()._after_optimizer_step(model, optimizer_was_run)
        step = int(self.state.global_step) + 1
        if not self.falcon_manager.is_policy_update_step(step):
            return
        if not optimizer_was_run:
            self.falcon_manager.pending_actions.pop(step, None)
            self._falcon_probe_batches = {}
            self._falcon_probe_before = {}
            return
        rewards = {}
        for source_id in sorted(self._falcon_probe_batches):
            batch, keep_mask, probe_seed = self._falcon_probe_batches[source_id]
            after, after_diagnostics = self._probe_loss(
                model, batch, keep_mask, probe_seed
            )
            before, before_diagnostics, probe_sha = self._falcon_probe_before[source_id]
            if before_diagnostics != after_diagnostics:
                # Similarities are expected to change, but mask-derived counts are not.
                fixed_keys = (
                    "safe_negative_min",
                    "safe_negative_mean",
                    "safe_negative_max",
                    "masked_negative_fraction",
                )
                if any(
                    before_diagnostics[key] != after_diagnostics[key]
                    for key in fixed_keys
                ):
                    raise RuntimeError("FALCON probe mask diagnostics changed across update")
            rewards[source_id] = {
                "reward": before - after,
                "before": float(before.cpu()),
                "after": float(after.cpu()),
                "diagnostics": {
                    **{
                        key: before_diagnostics[key]
                        for key in (
                            "safe_negative_min",
                            "safe_negative_mean",
                            "safe_negative_max",
                            "masked_negative_fraction",
                        )
                    },
                    "positive_similarity_before": before_diagnostics[
                        "positive_similarity_mean"
                    ],
                    "positive_similarity_after": after_diagnostics[
                        "positive_similarity_mean"
                    ],
                    "probe_sha256": probe_sha,
                    "probe_seed": int(probe_seed),
                },
            }
        self.falcon_manager.update_policies(step, rewards)
        self._falcon_probe_batches = {}
        self._falcon_probe_before = {}

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        self.falcon_manager.save_policy_state(output_dir)
