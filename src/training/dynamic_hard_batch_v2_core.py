"""Pure helpers for rank-weighted full-dataset batch-greedy sampling."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path

import numpy as np


def teacher_adapter_fingerprint(model_path):
    """Hash the exact PEFT adapter configuration and weight artifacts."""
    root = Path(model_path)
    config_path = root / "adapter_config.json"
    weight_paths = sorted(root.glob("adapter_model*.safetensors"))
    if not weight_paths:
        weight_paths = sorted(root.glob("adapter_model*.bin"))
    if not config_path.is_file() or not weight_paths:
        raise RuntimeError(
            f"trained teacher adapter artifacts are missing under {root}"
        )

    def digest(path):
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()

    return {
        "adapter_config_sha256": digest(config_path),
        "adapter_weight_sha256": {
            path.name: digest(path) for path in weight_paths
        },
        "adapter_weight_bytes": {
            path.name: path.stat().st_size for path in weight_paths
        },
    }


def b3_rank_start(subset_name):
    """Return the exact B3 safety boundary encoded in an HNPS subset name."""
    subset_name = str(subset_name)
    if "_70.170_" in subset_name:
        return 70
    if "_30.130_" in subset_name or "_30P_10_5_" in subset_name:
        return 30
    raise ValueError(
        "B3 HNPS subset does not encode a recognized 30–130/70–170 policy: "
        f"{subset_name}"
    )


def rank_weight(rank, start_rank, alpha):
    """B3-safe smooth rank weight; ranks are zero-based as in array slices."""
    if int(start_rank) < 0 or float(alpha) <= 0:
        raise ValueError("start_rank must be nonnegative and alpha must be positive")
    rank = np.asarray(rank)
    return np.where(
        rank < int(start_rank),
        0.0,
        np.exp(-float(alpha) * (rank - int(start_rank))),
    )


def build_rank_delta_csr(top_ids, start_rank, alpha):
    """Build the symmetric sparse score correction used by full-set greedy.

    ``top_ids[i, r]`` is target ``j`` at zero-based rank ``r`` for query i.
    Ranks beyond the stored depth use the common tail weight w(depth).  That
    common all-pairs term cancels from every greedy argmax, so only corrections
    from the sparse known-rank union are stored.  If either known direction is
    in the unsafe prefix, the pair's true score is zero and its correction is
    ``-tail_weight``.
    """
    try:
        import scipy.sparse as sp
    except ImportError as exc:
        raise RuntimeError("dynamic hard-batch v2 graph building requires scipy") from exc

    top_ids = np.asarray(top_ids)
    if top_ids.ndim != 2:
        raise ValueError("top_ids must have shape [num_cases, rank_depth]")
    n, depth = top_ids.shape
    if n < 2 or not 0 < start_rank < depth or float(alpha) <= 0:
        raise ValueError(f"invalid n/start/depth: {n}/{start_rank}/{depth}")
    if np.any(top_ids < 0) or np.any(top_ids >= n):
        raise ValueError("top_ids contains an out-of-dataset index")
    # torch.topk guarantees unique column indices. Keep the expensive per-row
    # validation for small synthetic/test inputs only; doing it for one million
    # production rows would dominate preprocessing time.
    if n <= 4096 and any(np.unique(row).size != depth for row in top_ids):
        raise ValueError("each ranking row must contain unique target IDs")

    rows = np.repeat(np.arange(n, dtype=np.int32), depth)
    cols = top_ids.reshape(-1).astype(np.int32, copy=False)
    ranks = np.tile(np.arange(depth, dtype=np.int16), n)
    nonself = rows != cols
    rows, cols, ranks = rows[nonself], cols[nonself], ranks[nonself]

    tail = float(math.exp(-float(alpha) * (depth - int(start_rank))))
    known_weight = np.exp(
        -float(alpha) * np.maximum(ranks.astype(np.int32) - int(start_rank), 0)
    ).astype(np.float32)
    correction = known_weight - tail
    directed = sp.csr_matrix((correction, (rows, cols)), shape=(n, n))
    delta = (directed + directed.T) * np.float32(0.5)

    unsafe_mask = ranks < int(start_rank)
    unsafe = sp.csr_matrix(
        (
            np.ones(int(unsafe_mask.sum()), dtype=np.float32),
            (rows[unsafe_mask], cols[unsafe_mask]),
        ),
        shape=(n, n),
    )
    unsafe.data.fill(1.0)
    unsafe = unsafe.maximum(unsafe.T)
    # Replace the existing correction at unsafe positions with -tail, without
    # densifying an N x N matrix.
    delta = delta - delta.multiply(unsafe) + unsafe * np.float32(-tail)
    delta.setdiag(0)
    delta.eliminate_zeros()
    delta.sum_duplicates()
    delta.sort_indices()
    symmetry_error = delta - delta.T
    if symmetry_error.nnz and float(np.max(np.abs(symmetry_error.data))) > 1e-6:
        raise RuntimeError("rank-weight graph is not symmetric")
    return delta.tocsr(), tail


def full_dataset_greedy(
    indptr, indices, data, seed, batch_size, *, return_diagnostics=False
):
    """Hard-prefix-safe max-sum greedy over every dataset case.

    Negative CSR entries encode the symmetric unsafe prefix: either endpoint
    ranks the other inside its B3 0:30/0:70 exclusion window.  Such an edge is
    a hard feasibility constraint, not merely a zero contribution.  Once a
    case is selected, every unsafe neighbor is removed from the preferred
    candidate pool. Gain is maximized over feasible cases whenever at least one
    exists. Only when that pool is empty does this position fall back to the
    original v2 gain argmax over every unselected case.
    """
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    data = np.asarray(data, dtype=np.float32)
    n = len(indptr) - 1
    seed = int(seed)
    batch_size = int(batch_size)
    if not 0 <= seed < n:
        raise IndexError(seed)
    if not 1 <= batch_size <= n:
        raise ValueError(f"batch_size={batch_size} is invalid for dataset size {n}")
    if len(indices) != len(data) or int(indptr[-1]) != len(indices):
        raise ValueError("invalid CSR arrays")

    if not np.isfinite(data).all():
        raise ValueError("CSR graph contains a nonfinite score correction")

    gain = np.zeros(n, dtype=np.float32)
    selected = np.empty(batch_size, dtype=np.int64)
    marginal = np.empty(batch_size, dtype=np.float32)
    chosen = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    feasible_before_selection = []
    fallback_positions = []

    def add_row(row):
        start, end = int(indptr[row]), int(indptr[row + 1])
        row_indices = indices[start:end]
        row_data = data[start:end]
        unsafe = row_data < 0
        # Keep the exact original-v2 gain for fallback decisions. For a strict
        # feasible candidate none of its accumulated edges is unsafe, so this
        # is also exactly its hard-prefix-safe gain.
        np.add.at(gain, row_indices, row_data)
        forbidden[row_indices[unsafe]] = True

    selected[0], marginal[0], chosen[seed] = seed, np.nan, True
    add_row(seed)
    if forbidden[seed]:
        raise RuntimeError("the symmetric unsafe graph contains a self conflict")
    gain[chosen] = -np.inf
    for position in range(1, batch_size):
        feasible = ~(chosen | forbidden)
        feasible_count = int(feasible.sum())
        feasible_before_selection.append(feasible_count)
        if feasible_count:
            decision_gain = np.where(feasible, gain, -np.inf)
            item = int(np.argmax(decision_gain))
        else:
            # No case can satisfy the hard prefix against every selected case.
            # Fall back for this position only, using the exact old-v2 gain.
            fallback_positions.append(position)
            decision_gain = np.where(~chosen, gain, -np.inf)
            item = int(np.argmax(decision_gain))
        if chosen[item] or not np.isfinite(decision_gain[item]):
            raise RuntimeError(
                "dynamic hard-batch v2 cannot select a finite unchosen candidate"
            )
        selected[position] = item
        marginal[position] = decision_gain[item]
        chosen[item] = True
        add_row(item)
        gain[chosen] = -np.inf
    if np.unique(selected).size != batch_size:
        raise RuntimeError("full-dataset greedy selected a duplicate")
    diagnostics = {
        "hard_prefix_exclusion_requested": True,
        "hard_prefix_exclusion_satisfied": not fallback_positions,
        "fallback_policy": "only_when_no_strict_candidate_then_original_v2_gain",
        "fallback_selection_count": int(len(fallback_positions)),
        "fallback_first_position": (
            int(fallback_positions[0]) if fallback_positions else None
        ),
        "fallback_positions_first16": [int(x) for x in fallback_positions[:16]],
        "feasible_before_selection_min": int(min(feasible_before_selection, default=0)),
        "feasible_before_last_selection": int(
            feasible_before_selection[-1] if feasible_before_selection else 0
        ),
        "forbidden_after_batch": int(forbidden.sum()),
        "remaining_feasible_after_batch": int((~(chosen | forbidden)).sum()),
    }
    if return_diagnostics:
        return selected, marginal, diagnostics
    return selected, marginal


def make_b3_proportional_schedule(batch_counts, max_steps, seed):
    """Shuffle B3's complete per-dataset batch inventory, pass by pass."""
    counts = np.asarray(batch_counts, dtype=np.int64)
    if counts.ndim != 1 or len(counts) == 0 or np.any(counts < 0) or counts.sum() == 0:
        raise ValueError("batch_counts must be a nonempty nonnegative vector")
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    inventory = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    rng = np.random.default_rng(int(seed))
    parts = []
    remaining = max_steps
    while remaining:
        shuffled = rng.permutation(inventory)
        take = min(remaining, len(shuffled))
        parts.append(shuffled[:take])
        remaining -= take
    return np.concatenate(parts).astype(np.int32, copy=False)
