"""Fast synthetic verification of HOBIT's equations and inventory semantics."""

import numpy as np
import torch

from src.training.hobit_core import (
    HOBITConfig,
    construct_hobit_batch,
    construct_hobit_source_epoch,
    rank_major_global_batch,
)


def unit_vectors(rows, dimension, seed):
    values = np.random.default_rng(seed).normal(size=(rows, dimension)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values.astype(np.float16)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_np = unit_vectors(35, 16, 71)
    target_np = unit_vectors(35, 16, 72)
    query = torch.from_numpy(query_np.astype(np.float32)).to(device)
    target = torch.from_numpy(target_np.astype(np.float32)).to(device)
    config = HOBITConfig(
        block_size=8,
        seed_size=2,
        candidate_k=4,
        alpha=1.0,
        temperature=0.02,
    )
    first, diagnostics = construct_hobit_batch(
        query,
        target,
        torch.ones(35, dtype=torch.bool, device=device),
        rng=np.random.default_rng(42),
        config=config,
    )
    repeated, _ = construct_hobit_batch(
        query,
        target,
        torch.ones(35, dtype=torch.bool, device=device),
        rng=np.random.default_rng(42),
        config=config,
    )
    if not torch.equal(first, repeated) or torch.unique(first).numel() != 8:
        raise RuntimeError("HOBIT synthetic batch is not deterministic and unique")

    previous = np.arange(8, dtype=np.int64)
    order, _, summary = construct_hobit_source_epoch(
        query_np,
        target_np,
        config=config,
        seed=43,
        device=device,
        avoid_first=previous,
    )
    if (
        len(order) != 32
        or np.unique(order).size != 32
        or set(order[:8]).intersection(previous.tolist())
    ):
        raise RuntimeError("HOBIT source coverage/boundary verification failed")
    rank_major = rank_major_global_batch(np.arange(16), world_size=4)
    if rank_major.tolist() != [
        0, 4, 8, 12,
        1, 5, 9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15,
    ]:
        raise RuntimeError("HOBIT rank-major layout verification failed")
    print(
        "[HOBIT_PREFLIGHT_OK] "
        f"device={device} deterministic=True unique=True "
        f"candidate_k_effective={diagnostics['candidate_k_effective']} "
        f"source_covered={summary['covered_cases']}/{summary['num_cases']} "
        "boundary_duplicate=False rank_major=True",
        flush=True,
    )


if __name__ == "__main__":
    main()
