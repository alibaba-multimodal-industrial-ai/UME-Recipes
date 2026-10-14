"""Exact hardness-weighted contrastive objective from LLaVE.

The reward model used in the paper is synchronized with the policy model and
does not backpropagate.  Therefore its score can be implemented exactly as a
detached copy of the current cosine similarity; no second model or forward is
required.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def llave_loss_from_similarity(
    similarity: Tensor,
    target: Tensor,
    *,
    temperature: float = 0.02,
    alpha: float = 9.0,
    reduction: str = "mean",
    return_diagnostics: bool = False,
) -> Tuple[Tensor, Optional[Dict[str, float]]]:
    """Compute the official LLaVE objective from a query-target cosine matrix.

    The paper presents the query-to-target direction in Eq. (5); the authors'
    released ``WeightedClipLoss`` computes both query-to-target and
    target-to-query and averages them.  This function follows that official
    implementation.  In each direction, positive logits remain ``s/tau`` and
    only negatives get ``alpha * stop_gradient(s)``.

    The calculation is performed in FP32 and expressed as additive logits.
    This is algebraically identical to the official ``log(exp(...))`` form but
    uses PyTorch's stable cross-entropy implementation without materializing
    the exponential.
    """
    if similarity.ndim != 2:
        raise ValueError(
            f"LLaVE expects a rank-2 similarity matrix, got {tuple(similarity.shape)}"
        )
    if target.ndim != 1 or target.numel() != similarity.size(0):
        raise ValueError(
            "LLaVE labels must contain one positive index per query: "
            f"similarity={tuple(similarity.shape)} target={tuple(target.shape)}"
        )
    if similarity.size(1) <= 1:
        raise ValueError("LLaVE requires at least one in-batch negative per query")
    if similarity.size(0) != similarity.size(1):
        raise ValueError(
            "The official symmetric LLaVE objective requires one target per "
            f"query and a square matrix, got {tuple(similarity.shape)}"
        )
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError(f"LLaVE temperature must be finite and positive, got {temperature}")
    if not math.isfinite(float(alpha)) or alpha < 0:
        raise ValueError(f"LLaVE alpha must be finite and non-negative, got {alpha}")
    if target.numel() and (
        int(target.min().item()) < 0 or int(target.max().item()) >= similarity.size(1)
    ):
        raise ValueError(
            f"LLaVE positive labels are out of range for {similarity.size(1)} targets"
        )

    similarity_fp32 = similarity.float()
    query_indices = torch.arange(
        similarity_fp32.size(0), device=target.device, dtype=target.dtype
    )
    reverse_target = torch.empty_like(target)
    reverse_target.scatter_(0, target, query_indices)

    def directional_loss(scores: Tensor, labels: Tensor):
        negative_mask = torch.ones_like(scores, dtype=torch.bool)
        negative_mask.scatter_(1, labels[:, None], False)

        # Official WeightedClipLoss:
        #   policy logits = logit_scale * cosine = cosine / temperature
        #   reward logits = alpha * policy.detach() / logit_scale
        #                 = alpha * stop_gradient(cosine)
        reward = float(alpha) * scores.detach()
        reward = reward.masked_fill(~negative_mask, 0.0)
        weighted_logits = scores / float(temperature) + reward
        directional = F.cross_entropy(weighted_logits, labels, reduction=reduction)
        return directional, negative_mask, reward, weighted_logits

    query_loss, query_negative_mask, query_reward, query_logits = directional_loss(
        similarity_fp32, target
    )
    target_loss, target_negative_mask, target_reward, target_logits = directional_loss(
        similarity_fp32.transpose(0, 1), reverse_target
    )
    loss = (query_loss + target_loss) / 2.0

    diagnostics = None
    if return_diagnostics:
        with torch.no_grad():
            def directional_diagnostics(scores, labels, negative_mask, reward, logits):
                positives = scores.gather(1, labels[:, None]).squeeze(1)
                hardest = scores.masked_fill(
                    ~negative_mask, float("-inf")
                ).max(dim=1).values
                negative_logits = logits.masked_fill(
                    ~negative_mask, float("-inf")
                )
                probabilities = torch.softmax(negative_logits, dim=1)
                safe_probability = probabilities.clamp_min(
                    torch.finfo(torch.float32).tiny
                )
                entropy = -(
                    probabilities
                    * safe_probability.log()
                    * negative_mask.to(dtype=torch.float32)
                ).sum(dim=1)
                k = min(5, scores.size(1) - 1)
                topk_share = probabilities.topk(k=k, dim=1).values.sum(dim=1)
                negative_reward = reward.masked_select(negative_mask)
                positive_reward = reward.gather(1, labels[:, None]).squeeze(1)
                return {
                    "positive_similarity_mean": positives.mean(),
                    "hardest_negative_similarity_mean": hardest.mean(),
                    "negative_reward_mean": negative_reward.mean(),
                    "negative_reward_min": negative_reward.min(),
                    "negative_reward_max": negative_reward.max(),
                    "positive_reward_abs_max": positive_reward.abs().max(),
                    "effective_negatives_mean": entropy.exp().mean(),
                    "top1_negative_share_mean": probabilities.max(dim=1).values.mean(),
                    "top5_negative_share_mean": topk_share.mean(),
                }

            query_stats = directional_diagnostics(
                similarity_fp32,
                target,
                query_negative_mask,
                query_reward,
                query_logits,
            )
            target_stats = directional_diagnostics(
                similarity_fp32.transpose(0, 1),
                reverse_target,
                target_negative_mask,
                target_reward,
                target_logits,
            )
            diagnostics = {
                "loss": float(loss.detach().float().mean().item()),
                "query_to_target_loss": float(query_loss.detach().float().mean().item()),
                "target_to_query_loss": float(target_loss.detach().float().mean().item()),
            }
            for name in query_stats:
                query_value = query_stats[name]
                target_value = target_stats[name]
                diagnostics[name] = float(((query_value + target_value) / 2).item())
                diagnostics[f"query_to_target_{name}"] = float(query_value.item())
                diagnostics[f"target_to_query_{name}"] = float(target_value.item())

    return loss, diagnostics
