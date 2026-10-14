"""Dependency-light core for per-anchor GloFND filtering.

The threshold update follows the released GloFND implementation.  The only
adaptations are lazy, one-time mini-batch-quantile initialization for each
anchor and a degenerate-update guard, because this repository's fixed-step
weighted sampler does not expose every training row once in a conventional
epoch.  The guard changes a threshold only when the released Adam update would
otherwise remove every observed negative for an anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GloFNDHyperparameters:
    alpha: float = 5e-4
    threshold_lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"GloFND alpha must be in (0, 1), got {self.alpha}")
        if self.threshold_lr <= 0.0:
            raise ValueError(
                f"GloFND threshold learning rate must be positive, got {self.threshold_lr}"
            )
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError(
                f"GloFND Adam betas must be in [0, 1), got {(self.beta1, self.beta2)}"
            )
        if self.eps <= 0.0:
            raise ValueError(f"GloFND epsilon must be positive, got {self.eps}")


class PerAnchorGloFNDState:
    """CPU-resident per-anchor thresholds and Adam moments."""

    VERSION = 1

    def __init__(self, num_anchors: int, hyperparameters: GloFNDHyperparameters):
        if int(num_anchors) <= 0:
            raise ValueError(f"num_anchors must be positive, got {num_anchors}")
        hyperparameters.validate()
        self.num_anchors = int(num_anchors)
        self.hyperparameters = hyperparameters
        self.threshold = torch.ones(self.num_anchors, dtype=torch.float32)
        self.first_moment = torch.zeros(self.num_anchors, dtype=torch.float32)
        self.second_moment = torch.zeros(self.num_anchors, dtype=torch.float32)
        self.visit_count = torch.zeros(self.num_anchors, dtype=torch.int32)
        self.initialized = torch.zeros(self.num_anchors, dtype=torch.bool)
        self.initialized_total = 0

    @staticmethod
    def _validate_similarity_and_ids(
        similarity: torch.Tensor, anchor_ids: torch.Tensor
    ) -> torch.Tensor:
        if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
            raise ValueError(
                "GloFND requires a square one-positive-per-query similarity matrix, "
                f"got {tuple(similarity.shape)}"
            )
        if similarity.shape[0] < 2:
            raise ValueError("GloFND requires at least two pairs in a batch")
        anchor_ids = anchor_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if anchor_ids.numel() != similarity.shape[0]:
            raise ValueError(
                f"GloFND IDs/similarity mismatch: {anchor_ids.numel()} != "
                f"{similarity.shape[0]}"
            )
        if torch.unique(anchor_ids).numel() != anchor_ids.numel():
            raise ValueError("GloFND requires unique anchor IDs in each global batch")
        return anchor_ids

    @torch.no_grad()
    def update_and_get_negative_mask(
        self,
        similarity: torch.Tensor,
        anchor_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Update visited anchors once and return a detached keep-negative mask."""
        anchor_ids = self._validate_similarity_and_ids(similarity, anchor_ids)
        if anchor_ids.min().item() < 0 or anchor_ids.max().item() >= self.num_anchors:
            raise IndexError(
                "GloFND anchor ID outside allocated state: "
                f"min={anchor_ids.min().item()} max={anchor_ids.max().item()} "
                f"capacity={self.num_anchors}"
            )

        scores = similarity.detach().float()
        if not torch.isfinite(scores).all():
            raise FloatingPointError("GloFND received non-finite similarities")
        batch_size = scores.shape[0]
        negative_mask = ~torch.eye(batch_size, device=scores.device, dtype=torch.bool)
        negative_scores = scores.masked_select(negative_mask).view(batch_size, -1)
        batch_quantile = torch.quantile(
            negative_scores,
            1.0 - self.hyperparameters.alpha,
            dim=1,
        )

        was_initialized_cpu = self.initialized[anchor_ids]
        is_new_cpu = ~was_initialized_cpu
        is_old_cpu = was_initialized_cpu
        is_new = is_new_cpu.to(scores.device)
        is_old = is_old_cpu.to(scores.device)

        lda = self.threshold[anchor_ids].to(scores.device)
        first_moment = self.first_moment[anchor_ids].to(scores.device)
        second_moment = self.second_moment[anchor_ids].to(scores.device)
        visits = self.visit_count[anchor_ids].to(scores.device, dtype=torch.long)

        # Released GloFND optionally initializes lambda from the mini-batch
        # quantile during its initialization epoch.  Here it is done exactly
        # once, when each anchor is first observed under the cycling sampler.
        lda = torch.where(is_new, batch_quantile, lda)

        if is_old.any():
            exceed_fraction = (
                ((scores > lda[:, None]) & negative_mask).sum(dim=1).float()
                / float(batch_size - 1)
            )
            gradient = self.hyperparameters.alpha - exceed_fraction
            updated_m = (
                self.hyperparameters.beta1 * first_moment
                + (1.0 - self.hyperparameters.beta1) * gradient
            )
            updated_v = (
                self.hyperparameters.beta2 * second_moment
                + (1.0 - self.hyperparameters.beta2) * gradient.square()
            )
            # Count the initialization visit as the first observation, matching
            # the official epoch-indexed bias-correction semantics.
            updated_visits = visits + 1
            m_hat = updated_m / (
                1.0 - self.hyperparameters.beta1 ** updated_visits.float()
            )
            v_hat = updated_v / (
                1.0 - self.hyperparameters.beta2 ** updated_visits.float()
            )
            updated_lda = (
                lda
                - self.hyperparameters.threshold_lr
                * m_hat
                / (v_hat.sqrt() + self.hyperparameters.eps)
            ).clamp_(-1.0, 1.0)
            lda = torch.where(is_old, updated_lda, lda)
            first_moment = torch.where(is_old, updated_m, first_moment)
            second_moment = torch.where(is_old, updated_v, second_moment)

        proposed_lda = lda.clone()
        keep_negative = negative_mask & (scores <= lda[:, None])
        retained_count = keep_negative.sum(dim=1)
        # The paper assumes that N_i(lambda_i) is non-empty.  With the
        # fixed-step sampler used here, an anchor can be revisited in a batch
        # whose similarity range is narrower than one threshold-Adam step.
        # The proposed lambda can then jump below every observed negative.
        # Recalibrate only those degenerate rows to the same current-batch
        # quantile used for lazy initialization.  This retains meaningful
        # negatives and avoids silently turning the row into a zero-loss case.
        empty_filter_fallback = is_old & (retained_count == 0)
        if empty_filter_fallback.any():
            lda = torch.where(empty_filter_fallback, batch_quantile, lda)
            keep_negative = negative_mask & (scores <= lda[:, None])
            retained_count = keep_negative.sum(dim=1)

        filtered_mask = negative_mask & ~keep_negative
        filtered_count = filtered_mask.sum(dim=1)
        if torch.any(retained_count == 0):
            rows = torch.nonzero(retained_count == 0, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "GloFND quantile safety failed to retain a negative for global "
                f"rows {rows[:16]}"
            )

        new_visits = visits + 1
        self.threshold[anchor_ids] = lda.cpu()
        self.first_moment[anchor_ids] = first_moment.cpu()
        self.second_moment[anchor_ids] = second_moment.cpu()
        self.visit_count[anchor_ids] = new_visits.to(torch.int32).cpu()
        self.initialized[anchor_ids] = True
        new_count = int(is_new_cpu.sum().item())
        self.initialized_total += new_count

        diagnostics = {
            "threshold": lda.detach(),
            "proposed_threshold": proposed_lda.detach(),
            "batch_quantile": batch_quantile.detach(),
            "new_anchor": is_new.detach(),
            "empty_filter_fallback": empty_filter_fallback.detach(),
            "visit_count": new_visits.detach(),
            "retained_count": retained_count.detach(),
            "filtered_count": filtered_count.detach(),
            "filtered_mask": filtered_mask.detach(),
            "negative_mask": negative_mask.detach(),
        }
        return keep_negative.detach(), diagnostics

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "num_anchors": self.num_anchors,
            "hyperparameters": vars(self.hyperparameters),
            "threshold": self.threshold,
            "first_moment": self.first_moment,
            "second_moment": self.second_moment,
            "visit_count": self.visit_count,
            "initialized": self.initialized,
            "initialized_total": self.initialized_total,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("version", -1)) != self.VERSION:
            raise RuntimeError(
                f"Unsupported GloFND state version: {state.get('version')}"
            )
        expected_hparams = vars(self.hyperparameters)
        if state.get("hyperparameters") != expected_hparams:
            raise RuntimeError(
                "GloFND checkpoint hyperparameters differ: "
                f"checkpoint={state.get('hyperparameters')} current={expected_hparams}"
            )
        if int(state.get("num_anchors", -1)) != self.num_anchors:
            raise RuntimeError(
                "GloFND checkpoint capacity differs: "
                f"{state.get('num_anchors')} != {self.num_anchors}"
            )
        tensor_specs = {
            "threshold": torch.float32,
            "first_moment": torch.float32,
            "second_moment": torch.float32,
            "visit_count": torch.int32,
            "initialized": torch.bool,
        }
        for name, dtype in tensor_specs.items():
            value = state.get(name)
            if not torch.is_tensor(value) or value.shape != (self.num_anchors,):
                raise RuntimeError(
                    f"Invalid GloFND checkpoint tensor {name}: "
                    f"{None if value is None else tuple(value.shape)}"
                )
            setattr(self, name, value.detach().to(device="cpu", dtype=dtype).clone())
        for name in ("threshold", "first_moment", "second_moment"):
            if not torch.isfinite(getattr(self, name)).all():
                raise FloatingPointError(
                    f"GloFND checkpoint contains non-finite {name}"
                )
        # The released quantile initialization is not clamped. With BF16
        # normalized representations its first-visit value can therefore be
        # a few ulps outside the ideal cosine interval. Subsequent Adam updates
        # use the official [-1, 1] clamp. Keep a narrow load tolerance so such
        # a valid first-visit checkpoint remains resumable.
        if torch.any(self.threshold < -1.05) or torch.any(self.threshold > 1.05):
            raise RuntimeError(
                "GloFND checkpoint thresholds exceed the BF16 cosine tolerance"
            )
        if torch.any(self.visit_count < 0):
            raise RuntimeError("GloFND checkpoint has negative visit counts")
        if torch.any(self.second_moment < 0):
            raise RuntimeError("GloFND checkpoint has negative Adam second moments")
        if not torch.equal(self.initialized, self.visit_count > 0):
            raise RuntimeError("GloFND initialized flags and visit counts disagree")
        uninitialized = ~self.initialized
        if not torch.equal(
            self.threshold[uninitialized],
            torch.ones_like(self.threshold[uninitialized]),
        ):
            raise RuntimeError("GloFND uninitialized thresholds must equal one")
        if torch.count_nonzero(self.first_moment[uninitialized]).item() != 0 or torch.count_nonzero(
            self.second_moment[uninitialized]
        ).item() != 0:
            raise RuntimeError("GloFND uninitialized Adam moments must equal zero")
        initialized_total = int(self.initialized.sum().item())
        if initialized_total != int(state.get("initialized_total", -1)):
            raise RuntimeError(
                "GloFND initialized count mismatch: "
                f"tensor={initialized_total} metadata={state.get('initialized_total')}"
            )
        self.initialized_total = initialized_total


def glofnd_filtered_info_nce(
    query: torch.Tensor,
    target: torch.Tensor,
    anchor_ids: torch.Tensor,
    state: PerAnchorGloFNDState,
    temperature: float,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One-direction InfoNCE with GloFND's detached false-negative filter."""
    if query.shape != target.shape:
        raise ValueError(
            f"GloFND requires one target per query, got {query.shape} and {target.shape}"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    # Preserve the established in-batch InfoNCE arithmetic: form similarities
    # in the representation dtype.  The released GloFND code likewise casts
    # the already-computed similarities to FP32 only for threshold estimation.
    similarity_native = query @ target.T
    similarity = similarity_native.float()
    keep_negative, diagnostics = state.update_and_get_negative_mask(
        similarity, anchor_ids
    )
    batch_size = similarity.shape[0]
    positive_mask = torch.eye(batch_size, device=similarity.device, dtype=torch.bool)
    allowed = positive_mask | keep_negative
    logits = (similarity_native / float(temperature)).masked_fill(
        ~allowed, float("-inf")
    )
    labels = torch.arange(batch_size, device=similarity.device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    unfiltered_loss = F.cross_entropy(
        similarity_native / float(temperature), labels
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"GloFND InfoNCE loss is non-finite: {loss.item()}")
    with torch.no_grad():
        unfiltered_probability = torch.softmax(
            similarity / float(temperature), dim=1
        )
        removed_negative_probability_mass = (
            unfiltered_probability * diagnostics["filtered_mask"]
        ).sum(dim=1)
        unfiltered_accuracy = (similarity.argmax(dim=1) == labels).float().mean()
        filtered_accuracy = (logits.argmax(dim=1) == labels).float().mean()
    diagnostics.update(
        {
            "similarity": similarity.detach(),
            "positive_similarity": similarity.diagonal().detach(),
            "loss": loss.detach(),
            "unfiltered_loss": unfiltered_loss.detach(),
            "removed_negative_probability_mass": removed_negative_probability_mass,
            "unfiltered_accuracy": unfiltered_accuracy,
            "filtered_accuracy": filtered_accuracy,
        }
    )
    return loss, diagnostics
