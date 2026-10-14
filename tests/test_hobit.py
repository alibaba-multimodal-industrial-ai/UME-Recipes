import numpy as np
import torch

from src.training.hobit_core import (
    HOBITConfig,
    construct_hobit_batch,
    construct_hobit_source_epoch,
    rank_major_global_batch,
)


def _unit_vectors(rows, dimension, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, dimension)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values.astype(np.float16)


def test_hobit_batch_is_deterministic_unique_and_uses_exact_size():
    query = torch.from_numpy(_unit_vectors(24, 12, seed=1).astype(np.float32))
    target = torch.from_numpy(_unit_vectors(24, 12, seed=2).astype(np.float32))
    available = torch.ones(24, dtype=torch.bool)
    config = HOBITConfig(
        block_size=8,
        seed_size=2,
        candidate_k=4,
        alpha=1.0,
        temperature=0.02,
    )
    first, first_diag = construct_hobit_batch(
        query,
        target,
        available,
        rng=np.random.default_rng(42),
        config=config,
    )
    second, second_diag = construct_hobit_batch(
        query,
        target,
        available,
        rng=np.random.default_rng(42),
        config=config,
    )
    assert torch.equal(first, second)
    assert len(first) == len(torch.unique(first)) == 8
    assert first_diag == second_diag
    assert first_diag["selected_size"] == 6
    assert np.isfinite(first_diag["marginal_gain_first"])
    assert first_diag["marginal_gain_first"] >= first_diag["marginal_gain_last"]


def test_hobit_expands_overlapping_candidate_union_instead_of_random_fallback():
    target = torch.from_numpy(_unit_vectors(20, 8, seed=3).astype(np.float32))
    query = target[0].repeat(20, 1)
    config = HOBITConfig(
        block_size=8,
        seed_size=2,
        candidate_k=1,
        alpha=1.0,
        temperature=0.02,
    )
    block, diagnostics = construct_hobit_batch(
        query,
        target,
        torch.ones(20, dtype=torch.bool),
        rng=np.random.default_rng(7),
        config=config,
    )
    assert len(torch.unique(block)) == 8
    assert diagnostics["candidate_expansions"] > 0
    assert diagnostics["candidate_union_size"] >= 6


def test_hobit_first_greedy_choice_matches_equation_16_brute_force():
    query = torch.from_numpy(_unit_vectors(18, 9, seed=31).astype(np.float32))
    target = torch.from_numpy(_unit_vectors(18, 9, seed=32).astype(np.float32))
    config = HOBITConfig(
        block_size=6,
        seed_size=2,
        candidate_k=16,
        alpha=1.0,
        temperature=0.02,
    )
    block, _ = construct_hobit_batch(
        query,
        target,
        torch.ones(18, dtype=torch.bool),
        rng=np.random.default_rng(13),
        config=config,
    )
    seeds = block[:2]
    selected_first = int(block[2])
    remaining = torch.tensor(
        [index for index in range(18) if index not in seeds.tolist()],
        dtype=torch.long,
    )
    q_seed = query[seeds]
    d_seed = target[seeds]
    seed_w = q_seed @ d_seed.T - config.alpha * (d_seed @ d_seed.T)
    denominator = torch.exp(seed_w / config.temperature).sum(dim=1)
    candidate_d = target[remaining]
    candidate_w = q_seed @ candidate_d.T - config.alpha * (d_seed @ candidate_d.T)
    brute_gain = config.temperature * torch.log1p(
        torch.exp(candidate_w / config.temperature) / denominator[:, None]
    ).sum(dim=0)
    assert selected_first == int(remaining[torch.argmax(brute_gain)])


def test_hobit_source_epoch_has_coverage_and_avoids_previous_boundary_block():
    query = _unit_vectors(35, 10, seed=4)
    target = _unit_vectors(35, 10, seed=5)
    config = HOBITConfig(
        block_size=8,
        seed_size=2,
        candidate_k=4,
        alpha=1.0,
        temperature=0.02,
    )
    previous = np.arange(8, dtype=np.int64)
    order, records, summary = construct_hobit_source_epoch(
        query,
        target,
        config=config,
        seed=11,
        avoid_first=previous,
    )
    assert len(order) == 32
    assert len(records) == 4
    assert summary["covered_cases"] == summary["unique_cases"] == 32
    assert summary["dropped_tail"] == 3
    assert not set(order[:8]).intersection(previous.tolist())
    # The boundary exclusion is temporary, so previous cases remain eligible
    # for later blocks of the new source epoch.
    assert set(previous).intersection(order[8:].tolist())


def test_rank_major_layout_reconstructs_per_rank_strides():
    logical = np.arange(16, dtype=np.int64)
    rank_major = rank_major_global_batch(logical, world_size=4)
    assert rank_major.tolist() == [
        0, 4, 8, 12,
        1, 5, 9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15,
    ]
