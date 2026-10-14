"""Paper-faithful FALCON primitives plus the agreed MMEB probe reward.

The policy network, similarity state and quantile-chain construction mirror
the official CVPR 2026 implementation.  The only objective substitution lives
in :func:`safe_probe_info_nce`: MMEB has no MLM objective, so an independent
same-source probe stream supplies the before/after scalar reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Exact residual MLP block used by the official FALCON actor."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear_up = nn.Linear(hidden_dim, hidden_dim * 4)
        self.activation = nn.ReLU()
        self.linear_down = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        inputs = self.norm1(inputs)
        inputs = self.linear_up(inputs)
        inputs = self.activation(inputs)
        inputs = self.linear_down(inputs)
        return residual + inputs


class SimbaPolicyQuantile(nn.Module):
    """Official FALCON policy: 100-D state -> per-row Beta parameters."""

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 256,
        num_residual_blocks: int = 2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear_logits_alpha = nn.Linear(hidden_dim, output_dim)
        self.linear_logits_beta = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_layer(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm1(hidden)
        alpha = F.softplus(self.linear_logits_alpha(hidden)) + 1e-5
        beta = F.softplus(self.linear_logits_beta(hidden)) + 1e-5
        return alpha.squeeze(-1), beta.squeeze(-1)


@dataclass(frozen=True)
class FALCONConfig:
    source_block_size: int = 512
    search_space_size: int = 5632
    num_downsample: int = 100
    policy_hidden_dim: int = 256
    policy_residual_blocks: int = 2
    policy_learning_rate: float = 1e-4
    policy_update_frequency: int = 10
    probe_global_size: int = 64
    temperature: float = 0.02

    def validate(self, *, world_size: int = 1) -> None:
        if self.source_block_size <= 1:
            raise ValueError("FALCON source block size must be greater than one")
        if self.search_space_size < self.source_block_size:
            raise ValueError("FALCON search space is smaller than its source block")
        if self.search_space_size % self.source_block_size:
            raise ValueError(
                "FALCON search space must be divisible by source block size: "
                f"{self.search_space_size} % {self.source_block_size}"
            )
        if self.num_downsample <= 1:
            raise ValueError("FALCON quantile state must contain at least two values")
        if self.policy_hidden_dim <= 0 or self.policy_residual_blocks < 0:
            raise ValueError("Invalid FALCON policy architecture")
        if not math.isfinite(self.policy_learning_rate) or self.policy_learning_rate <= 0:
            raise ValueError("FALCON policy learning rate must be finite and positive")
        if self.policy_update_frequency <= 0:
            raise ValueError("FALCON policy update frequency must be positive")
        if self.probe_global_size <= 1 or self.probe_global_size % world_size:
            raise ValueError(
                "FALCON global probe size must be >1 and divisible by world size"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("FALCON temperature must be finite and positive")


def quantile_downsample(similarity: torch.Tensor, count: int) -> torch.Tensor:
    """Match official ``quantile_downsample`` including descending flip."""

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(
            f"FALCON similarity must be square, got {tuple(similarity.shape)}"
        )
    if count <= 1:
        raise ValueError("FALCON quantile count must exceed one")
    quantiles = torch.linspace(
        0, 1, steps=int(count), device=similarity.device, dtype=similarity.dtype
    )
    return torch.quantile(similarity, quantiles, dim=1).transpose(0, 1).flip([1])


@torch.no_grad()
def falcon_similarity_state(
    query: torch.Tensor,
    target: torch.Tensor,
    num_downsample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official symmetric cross-modal similarity and actor input."""

    query = query.float()
    target = target.float()
    if query.shape != target.shape or query.ndim != 2:
        raise ValueError(
            f"FALCON query/target cache mismatch: {query.shape} != {target.shape}"
        )
    if not torch.isfinite(query).all() or not torch.isfinite(target).all():
        raise FloatingPointError("FALCON cache contains non-finite embeddings")
    cross = query @ target.T
    similarity = cross + cross.T
    state = F.softmax(
        quantile_downsample(similarity.cpu(), num_downsample), dim=-1
    )
    return similarity.cpu(), state


def construct_quantile_chain(
    similarity: torch.Tensor,
    remaining: torch.Tensor,
    quantile: torch.Tensor,
    *,
    block_size: int,
    start_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official no-replacement, previous-item-conditioned FALCON chain."""

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("FALCON chain requires a square similarity matrix")
    if quantile.ndim != 1 or len(quantile) != len(similarity):
        raise ValueError("FALCON requires one sampled quantile per search-space row")
    remaining = remaining.clone().long()
    if len(remaining) < block_size:
        raise ValueError(
            f"FALCON has {len(remaining)} remaining rows for block {block_size}"
        )
    if not 0 <= int(start_position) < len(remaining):
        raise IndexError(f"Invalid FALCON start position: {start_position}")

    searched = remaining[int(start_position)]
    selected = [searched]
    used_rows = [searched]
    remaining = remaining[remaining != searched]
    for _ in range(block_size - 1):
        q = float(quantile[searched].detach().cpu())
        if not math.isfinite(q) or q < 0 or q > 1:
            raise ValueError(f"Invalid FALCON quantile {q}")
        kth = int(round((len(remaining) - 1) * q)) + 1
        _, relative = torch.kthvalue(similarity[searched, remaining], kth)
        searched = remaining[int(relative)]
        selected.append(searched)
        used_rows.append(searched)
        remaining = remaining[remaining != searched]
    selected = torch.stack(selected).long()
    used_rows = torch.stack(used_rows).long()
    if torch.unique(selected).numel() != block_size:
        raise RuntimeError("FALCON constructed a source block with duplicates")
    return selected, used_rows


def safe_probe_mask_from_csr(
    local_ids: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    delta: np.ndarray,
) -> torch.Tensor:
    """Build the B3-prefix-safe probe denominator mask for selected rows."""

    local_ids = np.asarray(local_ids, dtype=np.int64)
    size = len(local_ids)
    mask = np.ones((size, size), dtype=bool)
    positions = {int(value): position for position, value in enumerate(local_ids)}
    for row_position, row_id in enumerate(local_ids):
        start, end = int(indptr[row_id]), int(indptr[row_id + 1])
        row_indices = indices[start:end]
        row_delta = delta[start:end]
        for neighbor in row_indices[row_delta < 0]:
            neighbor_position = positions.get(int(neighbor))
            if neighbor_position is not None:
                mask[row_position, neighbor_position] = False
    np.fill_diagonal(mask, True)
    return torch.from_numpy(mask)


def safe_probe_info_nce(
    query: torch.Tensor,
    target: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict]:
    """Query-to-target InfoNCE after masking teacher-risky off-diagonals."""

    if query.shape != target.shape or query.ndim != 2:
        raise ValueError("FALCON probe query/target embeddings must match")
    count = query.size(0)
    if keep_mask.shape != (count, count):
        raise ValueError(
            f"FALCON probe mask has shape {tuple(keep_mask.shape)}, expected {(count, count)}"
        )
    keep_mask = keep_mask.to(device=query.device, dtype=torch.bool)
    if not torch.diagonal(keep_mask).all():
        raise ValueError("FALCON probe mask removed an aligned positive")
    safe_negative_counts = keep_mask.sum(dim=1) - 1
    if torch.any(safe_negative_counts <= 0):
        raise ValueError("FALCON probe row has no safe negative")
    logits = (query.float() @ target.float().T) / float(temperature)
    logits = logits.masked_fill(~keep_mask, -torch.inf)
    labels = torch.arange(count, device=query.device)
    loss = F.cross_entropy(logits, labels)
    diagnostics = {
        "safe_negative_min": int(safe_negative_counts.min().item()),
        "safe_negative_mean": float(safe_negative_counts.float().mean().item()),
        "safe_negative_max": int(safe_negative_counts.max().item()),
        "masked_negative_fraction": float(
            ((~keep_mask).sum() / max(1, count * (count - 1))).item()
        ),
        "positive_similarity_mean": float(
            (query.float() * target.float()).sum(dim=1).mean().item()
        ),
    }
    return loss, diagnostics
