"""Numerically stable Explicit Gradient Amplification (EGA).

EGA changes the gradients of the ordinary one-positive InfoNCE objective; it
does not change the forward loss used for logging.  This module deliberately
contains no model, prompt, dataset, or distributed-training policy so the
method can be tested independently from the Qwen3.5 training stack.
"""

import math
from typing import Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def ega_gradients_and_loss(
    query: Tensor,
    target: Tensor,
    *,
    temperature: float = 0.02,
    alpha: float = 20.0,
    return_diagnostics: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
    """Return un-reduced EGA embedding gradients and standard InfoNCE loss.

    ``query[i]`` and ``target[i]`` must be the positive pair.  Every other
    target is an in-batch negative.  Gradient tensors are calculated in FP32
    for stability and are *not* divided by batch size; the caller owns the
    reduction convention (important for DDP gradient caching).

    The negative probability mass of ordinary InfoNCE is preserved per query.
    EGA only redistributes that mass toward harder negatives.
    """
    if query.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "EGA expects rank-2 query/target tensors, got "
            f"query={tuple(query.shape)} target={tuple(target.shape)}"
        )
    if query.shape != target.shape:
        raise ValueError(
            "EGA requires exactly one aligned target per query, got "
            f"query={tuple(query.shape)} target={tuple(target.shape)}"
        )
    batch_size = query.size(0)
    if batch_size < 2:
        raise ValueError("EGA requires at least two global pairs")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError(f"EGA temperature must be finite and positive, got {temperature}")
    if not math.isfinite(float(alpha)) or alpha < 0:
        raise ValueError(f"EGA alpha must be finite and non-negative, got {alpha}")
    if not torch.isfinite(query).all() or not torch.isfinite(target).all():
        raise FloatingPointError("EGA received non-finite representations")

    # EGA probability and gradient algebra is intentionally FP32 even when the
    # encoder runs in BF16.  Raw exp(alpha * delta) is avoided because it can
    # overflow before the paper's negative-mass normalization cancels it.
    query_fp32 = query.detach().float()
    target_fp32 = target.detach().float()
    similarity = query_fp32 @ target_fp32.transpose(0, 1)
    logits = similarity / float(temperature)
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    rows = torch.arange(batch_size, device=query.device)
    negative_mask = torch.ones_like(probs, dtype=torch.bool)
    negative_mask[rows, rows] = False

    if float(alpha) == 0.0:
        # This is both the mathematical identity and a useful exact control:
        # no extra normalization round-off is introduced at alpha=0.
        ega_probs = probs
    else:
        positive_similarity = similarity[rows, rows]
        log_hardness = float(alpha) * (
            similarity - positive_similarity.unsqueeze(1)
        )
        amplified_negative_log_weights = log_probs + log_hardness
        amplified_negative_log_weights = amplified_negative_log_weights.masked_fill(
            ~negative_mask, float("-inf")
        )

        original_negative_log_mass = torch.logsumexp(
            log_probs.masked_fill(~negative_mask, float("-inf")), dim=1
        )
        amplified_negative_log_mass = torch.logsumexp(
            amplified_negative_log_weights, dim=1
        )
        normalized_negative_probs = torch.exp(
            amplified_negative_log_weights
            - amplified_negative_log_mass.unsqueeze(1)
            + original_negative_log_mass.unsqueeze(1)
        )
        ega_probs = normalized_negative_probs.masked_fill(~negative_mask, 0.0)
        ega_probs[rows, rows] = probs[rows, rows]

    weights = ega_probs.clone()
    weights[rows, rows] -= 1.0
    # Paper Eq. (Step 4): sum_j W_ij * (t_j - t_i).  Avoid materializing the
    # [B, B, D] difference tensor while retaining the stated formula even in
    # finite precision (where a row sum may be a few ulps away from zero).
    grad_query = (
        weights @ target_fp32
        - weights.sum(dim=1, keepdim=True) * target_fp32
    ) / float(temperature)
    grad_target = (weights.transpose(0, 1) @ query_fp32) / float(temperature)
    loss = F.cross_entropy(logits, rows, reduction="mean")

    if not (
        torch.isfinite(grad_query).all()
        and torch.isfinite(grad_target).all()
        and torch.isfinite(loss)
    ):
        raise FloatingPointError("EGA produced non-finite loss or embedding gradients")

    diagnostics: Dict[str, float] = {}
    if return_diagnostics:
        with torch.no_grad():
            original_negative_mass = probs.masked_fill(~negative_mask, 0.0).sum(dim=1)
            ega_negative_mass = ega_probs.masked_fill(~negative_mask, 0.0).sum(dim=1)
            mass_error = (ega_negative_mass - original_negative_mass).abs().max()

            negative_prob = ega_probs.masked_fill(~negative_mask, 0.0)
            conditional_negative_prob = negative_prob / ega_negative_mass.clamp_min(
                torch.finfo(torch.float32).tiny
            ).unsqueeze(1)
            entropy = -(
                conditional_negative_prob
                * conditional_negative_prob.clamp_min(torch.finfo(torch.float32).tiny).log()
            ).sum(dim=1)

            standard_weights = probs.clone()
            standard_weights[rows, rows] -= 1.0
            standard_grad_query = (
                standard_weights @ target_fp32
                - standard_weights.sum(dim=1, keepdim=True) * target_fp32
            ) / float(temperature)
            standard_norm = standard_grad_query.norm(dim=1)
            ega_norm = grad_query.norm(dim=1)
            gradient_cosine = F.cosine_similarity(
                standard_grad_query, grad_query, dim=1, eps=1e-12
            )

            negative_similarity = similarity.masked_fill(~negative_mask, float("-inf"))
            hardest_similarity = negative_similarity.max(dim=1).values
            diagnostics = {
                "loss": float(loss.item()),
                "positive_similarity_mean": float(similarity.diag().mean().item()),
                "hardest_negative_similarity_mean": float(hardest_similarity.mean().item()),
                "negative_mass_mean": float(ega_negative_mass.mean().item()),
                "negative_mass_max_error": float(mass_error.item()),
                "weight_row_sum_max_error": float(
                    weights.sum(dim=1).abs().max().item()
                ),
                "effective_negatives_mean": float(entropy.exp().mean().item()),
                "top1_negative_share_mean": float(conditional_negative_prob.max(dim=1).values.mean().item()),
                "query_gradient_norm_standard_mean": float(standard_norm.mean().item()),
                "query_gradient_norm_ega_mean": float(ega_norm.mean().item()),
                "query_gradient_norm_ratio_mean": float(
                    (ega_norm / standard_norm.clamp_min(1e-12)).mean().item()
                ),
                "query_gradient_cosine_mean": float(gradient_cosine.mean().item()),
            }

    return grad_query, grad_target, loss.detach(), diagnostics


def b3_ega_gradients_and_loss(
    query: Tensor,
    target: Tensor,
    *,
    target_per_query: int,
    temperature: float = 0.02,
    alpha: float = 20.0,
    return_diagnostics: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
    """Apply EGA to B3's one-positive-plus-hard-negatives target layout.

    B3 stores each query's positive first, followed by its explicitly mined
    negatives.  For ``K=target_per_query`` the positive column for query ``i``
    is therefore ``i * K``.  All other columns, including targets belonging to
    other B3 cases, remain ordinary cross-device in-batch negatives.

    The returned embedding gradients are sums over global queries, rather
    than mean-reduced gradients.  GradCache applies the DDP/local-batch scale.
    This is the rectangular counterpart of :func:`ega_gradients_and_loss` and
    deliberately leaves that verified one-target-per-query path unchanged.
    """
    if query.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "B3+EGA expects rank-2 query/target tensors, got "
            f"query={tuple(query.shape)} target={tuple(target.shape)}"
        )
    if query.size(0) <= 0 or query.size(1) != target.size(1):
        raise ValueError(
            "B3+EGA query/target representation dimensions are incompatible: "
            f"query={tuple(query.shape)} target={tuple(target.shape)}"
        )
    if not isinstance(target_per_query, int) or target_per_query < 2:
        raise ValueError(
            "B3+EGA requires one positive and at least one explicit hard "
            f"negative per query, got target_per_query={target_per_query!r}"
        )
    expected_targets = query.size(0) * target_per_query
    if target.size(0) != expected_targets:
        raise ValueError(
            "B3+EGA target layout must be rectangular B x (B*K): "
            f"query={tuple(query.shape)} target={tuple(target.shape)} "
            f"target_per_query={target_per_query} expected_targets={expected_targets}"
        )
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError(
            f"B3+EGA temperature must be finite and positive, got {temperature}"
        )
    if not math.isfinite(float(alpha)) or alpha < 0:
        raise ValueError(f"B3+EGA alpha must be finite and non-negative, got {alpha}")
    if not torch.isfinite(query).all() or not torch.isfinite(target).all():
        raise FloatingPointError("B3+EGA received non-finite representations")

    query_fp32 = query.detach().float()
    target_fp32 = target.detach().float()
    similarity = query_fp32 @ target_fp32.transpose(0, 1)
    logits = similarity / float(temperature)
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    rows = torch.arange(query.size(0), device=query.device)
    positive_indices = rows * target_per_query
    negative_mask = torch.ones_like(probs, dtype=torch.bool)
    negative_mask[rows, positive_indices] = False

    if float(alpha) == 0.0:
        # Exact control: no re-normalization round-off at alpha=0, so this
        # path must reproduce B3's rectangular InfoNCE embedding gradients.
        ega_probs = probs
    else:
        positive_similarity = similarity[rows, positive_indices]
        log_hardness = float(alpha) * (
            similarity - positive_similarity.unsqueeze(1)
        )
        amplified_negative_log_weights = (log_probs + log_hardness).masked_fill(
            ~negative_mask, float("-inf")
        )
        original_negative_log_mass = torch.logsumexp(
            log_probs.masked_fill(~negative_mask, float("-inf")), dim=1
        )
        amplified_negative_log_mass = torch.logsumexp(
            amplified_negative_log_weights, dim=1
        )
        normalized_negative_probs = torch.exp(
            amplified_negative_log_weights
            - amplified_negative_log_mass.unsqueeze(1)
            + original_negative_log_mass.unsqueeze(1)
        )
        ega_probs = normalized_negative_probs.masked_fill(~negative_mask, 0.0)
        ega_probs[rows, positive_indices] = probs[rows, positive_indices]

    weights = ega_probs.clone()
    weights[rows, positive_indices] -= 1.0
    positive_targets = target_fp32[positive_indices]
    grad_query = (
        weights @ target_fp32
        - weights.sum(dim=1, keepdim=True) * positive_targets
    ) / float(temperature)
    grad_target = (weights.transpose(0, 1) @ query_fp32) / float(temperature)
    loss = F.cross_entropy(logits, positive_indices, reduction="mean")

    if not (
        torch.isfinite(grad_query).all()
        and torch.isfinite(grad_target).all()
        and torch.isfinite(loss)
    ):
        raise FloatingPointError(
            "B3+EGA produced non-finite loss or embedding gradients"
        )

    diagnostics: Dict[str, float] = {}
    if return_diagnostics:
        with torch.no_grad():
            original_negative_mass = probs.masked_fill(~negative_mask, 0.0).sum(dim=1)
            ega_negative_prob = ega_probs.masked_fill(~negative_mask, 0.0)
            ega_negative_mass = ega_negative_prob.sum(dim=1)
            mass_error = (ega_negative_mass - original_negative_mass).abs().max()
            conditional_negative_prob = ega_negative_prob / ega_negative_mass.clamp_min(
                torch.finfo(torch.float32).tiny
            ).unsqueeze(1)
            entropy = -(
                conditional_negative_prob
                * conditional_negative_prob.clamp_min(
                    torch.finfo(torch.float32).tiny
                ).log()
            ).sum(dim=1)

            target_owners = (
                torch.arange(target.size(0), device=query.device) // target_per_query
            )
            explicit_negative_mask = (
                target_owners.unsqueeze(0) == rows.unsqueeze(1)
            ) & negative_mask
            explicit_share = conditional_negative_prob.masked_fill(
                ~explicit_negative_mask, 0.0
            ).sum(dim=1)

            standard_weights = probs.clone()
            standard_weights[rows, positive_indices] -= 1.0
            standard_grad_query = (
                standard_weights @ target_fp32
                - standard_weights.sum(dim=1, keepdim=True) * positive_targets
            ) / float(temperature)
            standard_norm = standard_grad_query.norm(dim=1)
            ega_norm = grad_query.norm(dim=1)
            gradient_cosine = F.cosine_similarity(
                standard_grad_query, grad_query, dim=1, eps=1e-12
            )

            negative_similarity = similarity.masked_fill(
                ~negative_mask, float("-inf")
            )
            hardest_similarity, hardest_indices = negative_similarity.max(dim=1)
            hardest_is_explicit = explicit_negative_mask[
                rows, hardest_indices
            ].float()
            topk = min(5, target.size(0) - 1)
            topk_share = conditional_negative_prob.topk(topk, dim=1).values.sum(dim=1)
            positive_similarity = similarity[rows, positive_indices]

            diagnostics = {
                "loss": float(loss.item()),
                "positive_similarity_mean": float(positive_similarity.mean().item()),
                "hardest_negative_similarity_mean": float(hardest_similarity.mean().item()),
                "negative_beats_positive_fraction": float(
                    (hardest_similarity >= positive_similarity).float().mean().item()
                ),
                "negative_mass_mean": float(ega_negative_mass.mean().item()),
                "negative_mass_max_error": float(mass_error.item()),
                "weight_row_sum_max_error": float(
                    weights.sum(dim=1).abs().max().item()
                ),
                "effective_negatives_mean": float(entropy.exp().mean().item()),
                "top1_negative_share_mean": float(
                    conditional_negative_prob.max(dim=1).values.mean().item()
                ),
                "top5_negative_share_mean": float(topk_share.mean().item()),
                "explicit_b3_negative_share_mean": float(explicit_share.mean().item()),
                "ambient_inbatch_negative_share_mean": float(
                    (1.0 - explicit_share).mean().item()
                ),
                "hardest_is_explicit_b3_fraction": float(
                    hardest_is_explicit.mean().item()
                ),
                "query_gradient_norm_standard_mean": float(standard_norm.mean().item()),
                "query_gradient_norm_ega_mean": float(ega_norm.mean().item()),
                "query_gradient_norm_ratio_mean": float(
                    (ega_norm / standard_norm.clamp_min(1e-12)).mean().item()
                ),
                "query_gradient_cosine_mean": float(gradient_cosine.mean().item()),
            }

    return grad_query, grad_target, loss.detach(), diagnostics
