"""Fast dependency-free checks of FALCON's paper-faithful primitives."""

import numpy as np
import torch
from torch.distributions import Beta

from src.training.falcon_core import (
    FALCONConfig,
    SimbaPolicyQuantile,
    construct_quantile_chain,
    falcon_similarity_state,
    safe_probe_info_nce,
    safe_probe_mask_from_csr,
)


def main():
    torch.manual_seed(42)
    config = FALCONConfig(
        source_block_size=8,
        search_space_size=24,
        probe_global_size=4,
    )
    config.validate(world_size=1)

    actor = SimbaPolicyQuantile()
    actor_parameter_count = sum(parameter.numel() for parameter in actor.parameters())
    if actor_parameter_count != 1_079_042:
        raise RuntimeError(
            "FALCON official actor inventory changed: "
            f"{actor_parameter_count} != 1079042"
        )
    query = torch.randn(24, 16)
    target = torch.randn(24, 16)
    query = torch.nn.functional.normalize(query, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)
    similarity, state = falcon_similarity_state(query, target, 100)
    alpha, beta = actor(state)
    quantile = Beta(alpha, beta).sample()
    selected, used_rows = construct_quantile_chain(
        similarity,
        torch.arange(24),
        quantile,
        block_size=8,
        start_position=3,
    )
    if state.shape != (24, 100) or selected.shape != (8,):
        raise RuntimeError("FALCON state/chain shape contract failed")
    if torch.unique(selected).numel() != 8 or not torch.equal(selected, used_rows):
        raise RuntimeError("FALCON chain is not unique or row-conditioned")
    if not torch.isfinite(alpha).all() or not torch.isfinite(beta).all():
        raise RuntimeError("FALCON actor produced non-finite Beta parameters")
    if torch.any(alpha <= 0) or torch.any(beta <= 0):
        raise RuntimeError("FALCON actor produced non-positive Beta parameters")
    log_probability = Beta(alpha, beta).log_prob(quantile)
    actor_loss = -(log_probability.index_select(0, used_rows).sum() * 0.125)
    actor_loss.backward()
    actor_grads = [
        parameter.grad for parameter in actor.parameters() if parameter.grad is not None
    ]
    if not actor_grads or any(not torch.isfinite(grad).all() for grad in actor_grads):
        raise RuntimeError("FALCON REINFORCE gradient contract failed")

    # Rows 0 and 1 are a symmetric teacher-risky pair; all remaining probe
    # negatives are safe.  The diagonal positive must never be removed.
    indptr = np.asarray([0, 1, 2, 2, 2], dtype=np.int64)
    indices = np.asarray([1, 0], dtype=np.int64)
    delta = np.asarray([-1.0, -1.0], dtype=np.float32)
    keep = safe_probe_mask_from_csr(
        np.arange(4, dtype=np.int64), indptr, indices, delta
    )
    if keep[0, 1] or keep[1, 0] or not torch.diagonal(keep).all():
        raise RuntimeError("FALCON B3-safe probe mask contract failed")
    probe_q = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    probe_t = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    loss, diagnostics = safe_probe_info_nce(
        probe_q, probe_t, keep, temperature=0.02
    )
    if not torch.isfinite(loss) or diagnostics["safe_negative_min"] != 2:
        raise RuntimeError("FALCON safe probe InfoNCE contract failed")

    print(
        "[FALCON_PREFLIGHT_OK] "
        "actor=official_100x256_residual2 beta=softplus_plus_1e-5 "
        "similarity=i2t_plus_t2i state=100_descending_quantiles_softmax "
        "chain=previous_row_conditioned_no_replacement "
        "policy_update_steps=1,11,21,... "
        "reward=per_source_same_probe_before_minus_after_safe_info_nce "
        f"actor_parameters={actor_parameter_count} reinforce_grad_finite=True "
        f"selected_unique={torch.unique(selected).numel()}/8 "
        f"probe_masked_fraction={diagnostics['masked_negative_fraction']:.8g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
