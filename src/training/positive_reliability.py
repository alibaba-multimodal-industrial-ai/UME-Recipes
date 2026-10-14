"""Single-step positive-reliability diagnostics.

This module is intentionally independent from the active training loss.  It
measures how often in-batch negatives outrank each query's labeled positive
and maps that evidence to a proposed positive weight.
"""

import math
from typing import Dict

import torch
from torch import Tensor
import torch.nn.functional as F


def positive_reliability_from_similarity(
    similarity: Tensor,
    positive_indices: Tensor,
    *,
    temperature: float = 0.02,
    min_weight: float = 0.1,
    scale: float = 4.0,
) -> Dict[str, Tensor]:
    """Compute per-query violation statistics and a proposed loss weight.

    ``violation_magnitude`` is conditional on a negative actually exceeding
    the positive.  Consequently, ``ratio * magnitude`` equals the mean
    positive gap over *all* negatives and does not count violation frequency
    twice.

    The dimensionless anomaly score is::

        score = scale * violation_ratio * violation_magnitude / temperature

    and the proposed positive weight uses a hard floor::

        weight = max(exp(-score), min_weight)
    """
    if similarity.ndim != 2:
        raise ValueError(
            f"similarity must be rank 2, got shape={tuple(similarity.shape)}"
        )
    num_queries, num_targets = similarity.shape
    if num_targets < 2:
        raise ValueError("positive reliability requires at least one negative")
    if positive_indices.shape != (num_queries,):
        raise ValueError(
            "positive_indices must contain one index per query, got "
            f"shape={tuple(positive_indices.shape)} queries={num_queries}"
        )
    if positive_indices.dtype != torch.long:
        raise TypeError("positive_indices must have dtype torch.long")
    if torch.any(positive_indices < 0) or torch.any(positive_indices >= num_targets):
        raise ValueError("positive_indices contains an out-of-range target index")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(min_weight)) or not 0 <= min_weight < 1:
        raise ValueError("min_weight must be in [0, 1)")
    if not math.isfinite(float(scale)) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    if not torch.isfinite(similarity).all():
        raise FloatingPointError("similarity contains non-finite values")

    similarity_fp32 = similarity.float()
    rows = torch.arange(num_queries, device=similarity.device)
    positive_similarity = similarity_fp32[rows, positive_indices]
    negative_mask = torch.ones_like(similarity_fp32, dtype=torch.bool)
    negative_mask[rows, positive_indices] = False

    gaps = similarity_fp32 - positive_similarity.unsqueeze(1)
    positive_gaps = gaps.clamp_min(0.0).masked_fill(~negative_mask, 0.0)
    violations = (gaps > 0.0) & negative_mask
    violation_count = violations.sum(dim=1)
    negative_count = negative_mask.sum(dim=1)

    violation_ratio = violation_count.float() / negative_count.float()
    violation_magnitude = positive_gaps.sum(dim=1) / violation_count.clamp_min(1).float()
    score = (
        float(scale)
        * violation_ratio
        * violation_magnitude
        / float(temperature)
    )
    weight = torch.exp(-score).clamp_min(float(min_weight))

    return {
        "positive_similarity": positive_similarity,
        "violation_count": violation_count,
        "violation_ratio": violation_ratio,
        "violation_magnitude": violation_magnitude,
        "score": score,
        "weight": weight,
    }


def positive_reliability_weighted_info_nce(
    query: Tensor,
    target: Tensor,
    *,
    temperature: float = 0.02,
    min_weight: float = 0.1,
    scale: float = 4.0,
    positive_indices: Tensor = None,
):
    """Return normalized reliability-weighted InfoNCE and diagnostics.

    Reliability is deliberately computed from detached similarities.  The
    model therefore receives only the weighted InfoNCE gradient; it cannot
    improve the objective by manipulating the weight itself.
    """
    if query.ndim != 2 or target.ndim != 2:
        raise ValueError("query and target must both be rank-2 tensors")
    if query.size(1) != target.size(1):
        raise ValueError(
            "query and target embedding dimensions differ: "
            f"{query.size(1)} != {target.size(1)}"
        )
    if positive_indices is None:
        if query.size(0) != target.size(0):
            raise ValueError(
                "the probe requires exactly one aligned target per query, got "
                f"query={tuple(query.shape)} target={tuple(target.shape)}"
            )
        positive_indices = torch.arange(
            query.size(0), device=query.device, dtype=torch.long
        )

    # Accumulate dot products and cross entropy in fp32 even under bf16 AMP.
    similarity = torch.matmul(query.float(), target.float().transpose(0, 1))
    diagnostics = positive_reliability_from_similarity(
        similarity.detach(),
        positive_indices,
        temperature=temperature,
        min_weight=min_weight,
        scale=scale,
    )
    per_case_loss = F.cross_entropy(
        similarity / float(temperature),
        positive_indices,
        reduction="none",
    )
    weights = diagnostics["weight"].detach()
    loss = (weights * per_case_loss).sum() / weights.sum().clamp_min(1e-12)
    diagnostics["per_case_loss"] = per_case_loss.detach()
    return loss, diagnostics
