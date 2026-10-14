"""Dependency-light math core for online false-negative-risk handling."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def compact_nonself_neighbors(neighbor_ids, query_ids, top_k):
    """Keep the first K valid non-self IDs, tolerating unused FAISS slots."""
    neighbor_ids = np.asarray(neighbor_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    if neighbor_ids.ndim != 2 or query_ids.shape != (neighbor_ids.shape[0],):
        raise ValueError("neighbor IDs and query IDs have incompatible shapes")
    valid = (neighbor_ids >= 0) & (neighbor_ids != query_ids[:, None])
    valid_counts = valid.sum(axis=1)
    if np.any(valid_counts < int(top_k)):
        raise RuntimeError(
            "FAISS returned too few valid non-self neighbors: "
            f"minimum={int(valid_counts.min())}, required={int(top_k)}"
        )
    order = np.argsort(~valid, axis=1, kind="stable")
    compact = np.take_along_axis(neighbor_ids, order, axis=1)[:, : int(top_k)]
    if np.any(compact < 0) or np.any(compact == query_ids[:, None]):
        raise RuntimeError("neighbor compaction retained an invalid/self ID")
    return compact.astype(np.int32, copy=False)


def exact_jaccard_matrix(left_neighbors, right_neighbors):
    """Exact all-pairs Jaccard for two batches of fixed-size ID sets."""
    left = np.asarray(left_neighbors, dtype=np.int64)
    right = np.asarray(right_neighbors, dtype=np.int64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("neighbor tables must be rank-2")
    if left.shape[1] == 0 or right.shape[1] == 0:
        raise ValueError("neighbor sets cannot be empty")
    inverted = {}
    left_sets = []
    for row, neighbors in enumerate(left):
        unique_neighbors = set(map(int, neighbors))
        left_sets.append(unique_neighbors)
        for neighbor in unique_neighbors:
            inverted.setdefault(neighbor, []).append(row)
    right_sets = [set(map(int, neighbors)) for neighbors in right]
    overlap = np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    for right_row, neighbors in enumerate(right_sets):
        for neighbor in neighbors:
            left_rows = inverted.get(neighbor)
            if left_rows:
                overlap[left_rows, right_row] += 1.0
    left_size = np.asarray([len(values) for values in left_sets], dtype=np.float32)
    right_size = np.asarray([len(values) for values in right_sets], dtype=np.float32)
    union = left_size[:, None] + right_size[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def mutual_neighbor_matrix(anchor_ids, candidate_ids, anchor_neighbors, candidate_neighbors):
    """Return m_ij for directional anchor i -> candidate j."""
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    anchor_neighbors = np.asarray(anchor_neighbors, dtype=np.int64)
    candidate_neighbors = np.asarray(candidate_neighbors, dtype=np.int64)
    batch = len(anchor_ids)
    if len(candidate_ids) != batch:
        raise ValueError("one candidate case ID is required per anchor")
    candidate_positions = {}
    for position, case_id in enumerate(candidate_ids.tolist()):
        candidate_positions.setdefault(case_id, []).append(position)
    anchor_positions = {}
    for position, case_id in enumerate(anchor_ids.tolist()):
        anchor_positions.setdefault(case_id, []).append(position)
    candidate_in_anchor = np.zeros((batch, batch), dtype=bool)
    for anchor_row, neighbors in enumerate(anchor_neighbors):
        for neighbor in neighbors:
            for candidate_col in candidate_positions.get(int(neighbor), ()):
                candidate_in_anchor[anchor_row, candidate_col] = True
    anchor_in_candidate = np.zeros((batch, batch), dtype=bool)
    for candidate_col, neighbors in enumerate(candidate_neighbors):
        for neighbor in neighbors:
            for anchor_row in anchor_positions.get(int(neighbor), ()):
                anchor_in_candidate[anchor_row, candidate_col] = True
    return candidate_in_anchor & anchor_in_candidate


def false_negative_risk_energy(
    positive_target_reps,
    anchor_ids,
    candidate_ids,
    anchor_neighbors,
    candidate_neighbors,
    *,
    mutual_coefficient=0.5,
    anchor_dataset_ids=None,
    candidate_dataset_ids=None,
):
    """Compute detached E_ij = a_ij^2 + Jaccard_ij^2 + gamma*m_ij."""
    reps = torch.as_tensor(positive_target_reps, dtype=torch.float32).detach()
    if reps.ndim != 2 or reps.size(0) != len(anchor_ids):
        raise ValueError("positive_target_reps must contain one row per anchor")
    if not math.isfinite(float(mutual_coefficient)) or mutual_coefficient < 0:
        raise ValueError("mutual_coefficient must be finite and non-negative")
    similarity = (reps @ reps.T).clamp(min=0.0, max=1.0).cpu().numpy()
    jaccard = exact_jaccard_matrix(anchor_neighbors, candidate_neighbors)
    mutual = mutual_neighbor_matrix(
        anchor_ids, candidate_ids, anchor_neighbors, candidate_neighbors
    ).astype(np.float32)
    if (anchor_dataset_ids is None) != (candidate_dataset_ids is None):
        raise ValueError("both anchor and candidate dataset IDs must be provided")
    if anchor_dataset_ids is not None:
        anchor_dataset_ids = np.asarray(anchor_dataset_ids, dtype=np.int64)
        candidate_dataset_ids = np.asarray(candidate_dataset_ids, dtype=np.int64)
        if anchor_dataset_ids.shape != (len(anchor_ids),) or candidate_dataset_ids.shape != (
            len(candidate_ids),
        ):
            raise ValueError("dataset IDs must contain one value per case")
        same_dataset = anchor_dataset_ids[:, None] == candidate_dataset_ids[None, :]
        # Risk evidence is meaningful only inside one source dataset. Cross-source
        # in-batch negatives retain their original InfoNCE logits unchanged.
        similarity = np.where(same_dataset, similarity, 0.0)
        jaccard = np.where(same_dataset, jaccard, 0.0)
        mutual = np.where(same_dataset, mutual, 0.0)
    energy = similarity**2 + jaccard**2 + float(mutual_coefficient) * mutual
    np.fill_diagonal(energy, 0.0)
    return {
        "positive_candidate_similarity": similarity,
        "jaccard": jaccard,
        "mutual": mutual,
        "energy": energy.astype(np.float32, copy=False),
    }


def risk_adjusted_info_nce(query, target, energy, *, temperature=0.02, risk_lambda=1.0):
    """Subtract lambda*E from negative logits and leave positive logits intact."""
    if query.ndim != 2 or target.ndim != 2 or query.shape != target.shape:
        raise ValueError("the risk loss requires one aligned target per query")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(risk_lambda)) or risk_lambda < 0:
        raise ValueError("risk_lambda must be finite and non-negative")
    energy = torch.as_tensor(energy, device=query.device, dtype=torch.float32).detach()
    if energy.shape != (query.size(0), target.size(0)):
        raise ValueError("energy must match the global similarity matrix")
    if not torch.isfinite(energy).all() or torch.any(energy < 0):
        raise ValueError("energy must be finite and non-negative")
    rows = torch.arange(query.size(0), device=query.device)
    energy = energy.clone()
    energy[rows, rows] = 0.0
    logits = query.float() @ target.float().T / float(temperature)
    adjusted_logits = logits - float(risk_lambda) * energy
    loss = F.cross_entropy(adjusted_logits, rows)
    return loss, {
        "energy": energy,
        "weight": torch.exp(-float(risk_lambda) * energy),
        "plain_logits": logits.detach(),
        "adjusted_logits": adjusted_logits.detach(),
    }


def risk_deletion_probabilities(
    energy,
    *,
    risk_lambda=1.5,
    max_pair_drop_probability=0.95,
    max_drop_ratio=0.40,
):
    """Map detached FN-risk energy to aggressive, row-budgeted drop probabilities.

    Before the row cap, ``p_drop = 1 - exp(-lambda * energy)`` is the exact
    Bernoulli counterpart of the previous soft weight ``exp(-lambda*energy)``.
    Rows whose expected deletion count exceeds the configured fraction are
    uniformly rescaled, preserving their relative risk ordering.
    """
    energy = torch.as_tensor(energy, dtype=torch.float32).detach()
    if energy.ndim != 2 or energy.shape[0] != energy.shape[1]:
        raise ValueError("risk deletion requires a square energy matrix")
    if energy.shape[0] < 2:
        raise ValueError("risk deletion requires at least two global pairs")
    if not torch.isfinite(energy).all() or torch.any(energy < 0):
        raise ValueError("energy must be finite and non-negative")
    if not math.isfinite(float(risk_lambda)) or risk_lambda <= 0:
        raise ValueError("risk_lambda must be finite and positive")
    if not 0.0 < float(max_pair_drop_probability) < 1.0:
        raise ValueError("max_pair_drop_probability must be in (0, 1)")
    if not 0.0 < float(max_drop_ratio) < 1.0:
        raise ValueError("max_drop_ratio must be in (0, 1)")

    batch_size = energy.shape[0]
    rows = torch.arange(batch_size, device=energy.device)
    raw_probability = (
        -torch.expm1(-float(risk_lambda) * energy)
    ).clamp_(min=0.0, max=float(max_pair_drop_probability))
    raw_probability = raw_probability.clone()
    raw_probability[rows, rows] = 0.0

    max_drop_count = int(math.floor(float(max_drop_ratio) * (batch_size - 1)))
    if max_drop_count < 1:
        raise ValueError(
            "max_drop_ratio leaves no deletion budget for this batch size"
        )
    raw_expected = raw_probability.sum(dim=1)
    # Stay infinitesimally below the integer boundary so floating-point
    # cumulative sums cannot produce max_drop_count + 1 selected entries.
    capped_budget = max(0.0, float(max_drop_count) - 1e-4)
    scale = torch.minimum(
        torch.ones_like(raw_expected),
        torch.full_like(raw_expected, capped_budget)
        / raw_expected.clamp_min(torch.finfo(torch.float32).tiny),
    )
    probability = raw_probability * scale.unsqueeze(1)
    probability[rows, rows] = 0.0
    return probability.detach(), {
        "raw_probability": raw_probability.detach(),
        "raw_expected_drop_count": raw_expected.detach(),
        "expected_drop_count": probability.sum(dim=1).detach(),
        "budget_scale": scale.detach(),
        "max_drop_count": max_drop_count,
    }


def systematic_probability_mask(drop_probability, *, seed):
    """Low-variance dependent Bernoulli sampling with exact pair marginals.

    A systematic random grid selects floor/ceil(sum(p)) entries per row while
    retaining each pair's marginal deletion probability.  A deterministic
    cyclic shift avoids coupling the grid to global candidate order.
    """
    probability = torch.as_tensor(drop_probability, dtype=torch.float32).detach()
    if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
        raise ValueError("drop_probability must be a square matrix")
    if not torch.isfinite(probability).all() or torch.any(probability < 0) or torch.any(
        probability >= 1
    ):
        raise ValueError("drop probabilities must be finite and in [0, 1)")
    batch_size = probability.shape[0]
    rows = torch.arange(batch_size, device=probability.device)
    if torch.any(probability[rows, rows] != 0):
        raise ValueError("positive diagonal must have zero deletion probability")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    offsets_cpu = torch.randint(
        0, batch_size, (batch_size,), generator=generator, dtype=torch.long
    )
    uniforms_cpu = torch.rand(batch_size, generator=generator, dtype=torch.float64)
    offsets = offsets_cpu.to(probability.device)
    uniforms = uniforms_cpu.to(probability.device, dtype=torch.float64)
    positions = torch.arange(batch_size, device=probability.device)
    order = (positions.unsqueeze(0) + offsets.unsqueeze(1)) % batch_size
    ordered_probability = probability.gather(1, order).double()
    cumulative = ordered_probability.cumsum(dim=1)
    previous = cumulative - ordered_probability
    selected_ordered = torch.floor(cumulative - uniforms.unsqueeze(1)) > torch.floor(
        previous - uniforms.unsqueeze(1)
    )
    selected = torch.zeros_like(selected_ordered, dtype=torch.bool)
    selected.scatter_(1, order, selected_ordered)
    selected[rows, rows] = False
    return selected.detach()


def probabilistic_risk_info_nce(query, target, drop_mask, *, temperature=0.02):
    """InfoNCE after sampled false-negative removal; positives are immutable."""
    if query.ndim != 2 or target.ndim != 2 or query.shape != target.shape:
        raise ValueError("probabilistic risk loss requires one aligned target per query")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    drop_mask = torch.as_tensor(drop_mask, device=query.device, dtype=torch.bool).detach()
    if drop_mask.shape != (query.size(0), target.size(0)):
        raise ValueError("drop_mask must match the global similarity matrix")
    rows = torch.arange(query.size(0), device=query.device)
    if torch.any(drop_mask[rows, rows]):
        raise ValueError("probabilistic risk loss cannot delete positive pairs")
    retained_negative_count = (~drop_mask).sum(dim=1) - 1
    if torch.any(retained_negative_count <= 0):
        raise RuntimeError("probabilistic risk deletion removed every negative")

    logits = query.float() @ target.float().T / float(temperature)
    sampled_logits = logits.masked_fill(drop_mask, float("-inf"))
    loss = F.cross_entropy(sampled_logits, rows)
    plain_loss = F.cross_entropy(logits, rows)
    if not torch.isfinite(loss):
        raise FloatingPointError("probabilistic risk InfoNCE loss is non-finite")
    with torch.no_grad():
        plain_probability = torch.softmax(logits, dim=1)
        removed_probability_mass = (plain_probability * drop_mask).sum(dim=1)
    return loss, {
        "drop_mask": drop_mask,
        "plain_logits": logits.detach(),
        "sampled_logits": sampled_logits.detach(),
        "plain_loss": plain_loss.detach(),
        "removed_probability_mass": removed_probability_mass.detach(),
        "retained_negative_count": retained_negative_count.detach(),
    }
