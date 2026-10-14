"""Core HOBIT batch construction, independent of the training runtime."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class HOBITConfig:
    block_size: int = 512
    seed_size: int = 32
    candidate_k: int = 128
    alpha: float = 1.0
    temperature: float = 0.02

    def validate(self) -> None:
        if self.block_size <= 1:
            raise ValueError("HOBIT block_size must be greater than one")
        if not 0 < self.seed_size < self.block_size:
            raise ValueError(
                "HOBIT seed_size must lie strictly inside the block: "
                f"seed={self.seed_size}, block={self.block_size}"
            )
        if self.candidate_k <= 0:
            raise ValueError("HOBIT candidate_k must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("HOBIT alpha must be finite and positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("HOBIT temperature must be finite and positive")


def _as_unit_float_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(array), device=device, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"HOBIT embeddings must be a nonempty matrix: {tensor.shape}")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("HOBIT embedding cache contains non-finite values")
    norms = tensor.norm(dim=1)
    maximum_error = float((norms - 1.0).abs().max().item())
    # FP16 cache roundoff is normally below 1e-3. A broad 1% boundary catches
    # an unnormalised or corrupt cache without changing the vectors HOBIT sees.
    if maximum_error > 0.01:
        raise ValueError(
            "HOBIT requires L2-normalized cache embeddings; maximum norm "
            f"error is {maximum_error:.8g}"
        )
    return tensor


def _candidate_union(
    query_seed: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    candidate_k: int,
    required: int,
) -> tuple[torch.Tensor, int, int]:
    """Return the union of per-seed query-to-target top-k candidates.

    Algorithm 2 permits the candidate pool to become empty before the batch is
    full. The training runtime requires fixed-size source blocks, so k is
    doubled only when overlap leaves fewer than ``required`` unique cases. At
    the limit k=|U| the union is exactly U and no random fallback is needed.
    """

    available_ids = torch.nonzero(available, as_tuple=False).flatten()
    if available_ids.numel() < required:
        raise ValueError(
            f"HOBIT has only {available_ids.numel()} available cases for {required} slots"
        )
    similarities = query_seed @ target.index_select(0, available_ids).T
    effective_k = min(int(candidate_k), int(available_ids.numel()))
    expansions = 0
    while True:
        top_positions = torch.topk(
            similarities, k=effective_k, dim=1, largest=True, sorted=False
        ).indices
        candidate_ids = torch.unique(available_ids[top_positions.reshape(-1)])
        if candidate_ids.numel() >= required:
            return candidate_ids, effective_k, expansions
        if effective_k == available_ids.numel():
            raise RuntimeError(
                "HOBIT full candidate pool is unexpectedly smaller than the batch remainder"
            )
        effective_k = min(int(available_ids.numel()), effective_k * 2)
        expansions += 1


@torch.no_grad()
def construct_hobit_batch(
    query: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    *,
    rng: np.random.Generator,
    config: HOBITConfig,
) -> tuple[torch.Tensor, dict]:
    """Construct one paper-faithful seed-plus-greedy HOBIT source block."""

    config.validate()
    available_ids = torch.nonzero(available, as_tuple=False).flatten()
    if available_ids.numel() < config.block_size:
        raise ValueError(
            f"HOBIT needs {config.block_size} available cases, found {available_ids.numel()}"
        )
    available_cpu = available_ids.cpu().numpy()
    seed_np = rng.choice(available_cpu, size=config.seed_size, replace=False)
    seed_ids = torch.as_tensor(seed_np, device=query.device, dtype=torch.long)
    remaining = available.clone()
    remaining[seed_ids] = False

    query_seed = query.index_select(0, seed_ids)
    target_seed = target.index_select(0, seed_ids)
    required = config.block_size - config.seed_size
    candidate_ids, effective_k, expansions = _candidate_union(
        query_seed,
        target,
        remaining,
        config.candidate_k,
        required,
    )
    candidate_target = target.index_select(0, candidate_ids)

    # Eq. 6: w_ij = q_i^T d_j - alpha d_i^T d_j.
    candidate_q_similarity = query_seed @ candidate_target.T
    candidate_positive_similarity = target_seed @ candidate_target.T
    candidate_hardness = (
        candidate_q_similarity - config.alpha * candidate_positive_similarity
    )
    seed_q_similarity = query_seed @ target_seed.T
    seed_positive_similarity = target_seed @ target_seed.T
    seed_hardness = seed_q_similarity - config.alpha * seed_positive_similarity

    # Eq. 16 in log space. This is mathematically identical to maintaining
    # sum_j exp(w_ij/tau_h), but remains finite at tau_h=0.02.
    scaled_candidate = candidate_hardness / config.temperature
    log_denominator = torch.logsumexp(
        seed_hardness / config.temperature, dim=1
    )
    selected_positions = torch.empty(required, device=query.device, dtype=torch.long)
    marginal_gains = torch.empty(required, device=query.device, dtype=torch.float32)
    selectable = torch.ones(candidate_ids.numel(), device=query.device, dtype=torch.bool)
    for position in range(required):
        gains = config.temperature * torch.nn.functional.softplus(
            scaled_candidate - log_denominator[:, None]
        ).sum(dim=0)
        gains.masked_fill_(~selectable, -torch.inf)
        selected_position = int(torch.argmax(gains).item())
        selected_gain = gains[selected_position]
        if not torch.isfinite(selected_gain):
            raise FloatingPointError(
                f"HOBIT has no finite candidate at greedy position {position}"
            )
        selected_positions[position] = selected_position
        marginal_gains[position] = selected_gain
        selectable[selected_position] = False
        log_denominator = torch.logaddexp(
            log_denominator, scaled_candidate[:, selected_position]
        )

    selected_ids = candidate_ids.index_select(0, selected_positions)
    block = torch.cat((seed_ids, selected_ids))
    if torch.unique(block).numel() != config.block_size:
        raise RuntimeError("HOBIT selected duplicate cases inside one source block")
    if not available.index_select(0, block).all():
        raise RuntimeError("HOBIT selected a case outside the remaining epoch inventory")

    chosen_q_similarity = query_seed @ target.index_select(0, selected_ids).T
    chosen_positive_similarity = target_seed @ target.index_select(0, selected_ids).T
    chosen_hardness = chosen_q_similarity - config.alpha * chosen_positive_similarity
    diagnostics = {
        "seed_size": int(config.seed_size),
        "candidate_k_requested": int(config.candidate_k),
        "candidate_k_effective": int(effective_k),
        "candidate_expansions": int(expansions),
        "candidate_union_size": int(candidate_ids.numel()),
        "selected_size": int(selected_ids.numel()),
        "candidate_query_similarity_mean": float(
            candidate_q_similarity.mean().item()
        ),
        "candidate_positive_similarity_mean": float(
            candidate_positive_similarity.mean().item()
        ),
        "candidate_hardness_mean": float(candidate_hardness.mean().item()),
        "query_negative_similarity_mean": float(chosen_q_similarity.mean().item()),
        "query_negative_similarity_max": float(chosen_q_similarity.max().item()),
        "positive_negative_similarity_mean": float(
            chosen_positive_similarity.mean().item()
        ),
        "positive_negative_similarity_max": float(
            chosen_positive_similarity.max().item()
        ),
        "hardness_mean": float(chosen_hardness.mean().item()),
        "hardness_max": float(chosen_hardness.max().item()),
        "marginal_gain_first": float(marginal_gains[0].item()),
        "marginal_gain_last": float(marginal_gains[-1].item()),
        "block_sha256": hashlib.sha256(
            block.cpu().numpy().astype(np.int64, copy=False).tobytes()
        ).hexdigest(),
    }
    return block, diagnostics


@torch.no_grad()
def construct_hobit_source_epoch(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    *,
    config: HOBITConfig,
    seed: int,
    device: torch.device | str = "cpu",
    avoid_first: Iterable[int] | None = None,
    progress_callback: Callable[[int, int, dict], None] | None = None,
) -> tuple[np.ndarray, list[dict], dict]:
    """Order all complete source blocks for one HOBIT/HOBIT-C source epoch."""

    config.validate()
    if query_embeddings.shape != target_embeddings.shape:
        raise ValueError(
            "HOBIT query/target cache shape mismatch: "
            f"{query_embeddings.shape} != {target_embeddings.shape}"
        )
    device = torch.device(device)
    query = _as_unit_float_tensor(query_embeddings, device)
    target = _as_unit_float_tensor(target_embeddings, device)
    num_cases = int(query.shape[0])
    num_blocks = num_cases // config.block_size
    if num_blocks == 0:
        raise ValueError(
            f"HOBIT source has {num_cases} cases, fewer than block size {config.block_size}"
        )
    available = torch.ones(num_cases, device=device, dtype=torch.bool)
    avoid_first_array = np.asarray(
        list(avoid_first) if avoid_first is not None else [], dtype=np.int64
    )
    if avoid_first_array.size:
        if (
            np.any(avoid_first_array < 0)
            or np.any(avoid_first_array >= num_cases)
            or np.unique(avoid_first_array).size != avoid_first_array.size
        ):
            raise ValueError("HOBIT avoid_first contains invalid or duplicate IDs")
        available[
            torch.as_tensor(avoid_first_array, device=device, dtype=torch.long)
        ] = False
        if int(available.sum().item()) < config.block_size:
            raise ValueError("HOBIT avoid_first leaves too few cases for a full block")
    rng = np.random.default_rng(int(seed))
    blocks = []
    diagnostics = []
    for block_index in range(num_blocks):
        block, block_diagnostics = construct_hobit_batch(
            query,
            target,
            available,
            rng=rng,
            config=config,
        )
        available[block] = False
        if block_index == 0 and avoid_first_array.size:
            # The exclusion only prevents a source-epoch boundary from placing
            # the same case twice in one global batch. Restore those cases for
            # the remaining blocks so the new epoch retains full coverage.
            available[
                torch.as_tensor(avoid_first_array, device=device, dtype=torch.long)
            ] = True
        block_diagnostics["block_index"] = block_index
        blocks.append(block.cpu().numpy().astype(np.int64, copy=False))
        diagnostics.append(block_diagnostics)
        if progress_callback is not None:
            progress_callback(block_index + 1, num_blocks, block_diagnostics)
    order = np.concatenate(blocks)
    summary = {
        "num_cases": num_cases,
        "num_blocks": num_blocks,
        "covered_cases": int(order.size),
        "dropped_tail": int(num_cases - order.size),
        "unique_cases": int(np.unique(order).size),
        "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "candidate_expansions": int(
            sum(record["candidate_expansions"] for record in diagnostics)
        ),
        "first_block_boundary_exclusions": int(avoid_first_array.size),
    }
    if summary["unique_cases"] != summary["covered_cases"]:
        raise RuntimeError("HOBIT source epoch contains duplicate training cases")
    return order, diagnostics, summary


def rank_major_global_batch(indices: Iterable[int], world_size: int) -> np.ndarray:
    """Transpose a logical global batch for Accelerate's batch sharding."""

    indices = np.asarray(list(indices), dtype=np.int64)
    world_size = int(world_size)
    if world_size <= 0 or indices.size % world_size:
        raise ValueError(
            f"Cannot rank-shard {indices.size} HOBIT indices over {world_size} ranks"
        )
    return np.concatenate([indices[rank::world_size] for rank in range(world_size)])
