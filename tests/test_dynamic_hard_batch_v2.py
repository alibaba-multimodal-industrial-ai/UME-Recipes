import numpy as np
import pytest

from src.training.dynamic_hard_batch_v2_core import (
    b3_rank_start,
    build_rank_delta_csr,
    full_dataset_greedy,
    make_b3_proportional_schedule,
    rank_weight,
    teacher_adapter_fingerprint,
)


def test_b3_rank_start_mapping():
    assert b3_rank_start("OK-VQA_HNPS_Metis_bs16bi_70.170_qwen2b") == 70
    assert b3_rank_start("ImageNet_1K_HNPS_Metis_bs16bi_30P_10_5_qwen2b") == 30
    assert b3_rank_start("VisDial_HNPS_Metis_bs16bi_30.130_qwen2b") == 30
    with pytest.raises(ValueError):
        b3_rank_start("mistyped_HNPS_subset")


def test_teacher_adapter_fingerprint_changes_with_weights(tmp_path):
    (tmp_path / "adapter_config.json").write_text('{"r": 16}')
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"trained-adapter-a")
    first = teacher_adapter_fingerprint(tmp_path)
    assert first["adapter_weight_bytes"][weights.name] == len(b"trained-adapter-a")
    weights.write_bytes(b"trained-adapter-b")
    second = teacher_adapter_fingerprint(tmp_path)
    assert first["adapter_weight_sha256"] != second["adapter_weight_sha256"]


def test_rank_weight_matches_proposal():
    ranks = np.asarray([29, 30, 50, 80, 100, 130])
    values = rank_weight(ranks, start_rank=30, alpha=0.03)
    assert values[0] == 0
    np.testing.assert_allclose(
        values[1:],
        [1.0, 0.54881164, 0.22313016, 0.12245643, 0.04978707],
        rtol=1e-6,
    )


def test_sparse_delta_is_exact_for_clipped_tail_semantics():
    top_ids = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 4],
            [2, 3, 0, 4],
            [3, 2, 4, 0],
            [4, 3, 1, 2],
        ],
        dtype=np.int32,
    )
    start, alpha, depth = 1, 0.2, top_ids.shape[1]
    graph, tail = build_rank_delta_csr(top_ids, start, alpha)
    actual = graph.toarray()
    expected = np.zeros_like(actual)
    rank_maps = [dict(zip(row.tolist(), range(depth))) for row in top_ids]
    for left in range(len(top_ids)):
        for right in range(left + 1, len(top_ids)):
            lr = rank_maps[left].get(right, depth)
            rl = rank_maps[right].get(left, depth)
            if lr < start or rl < start:
                score = 0.0
            else:
                score = 0.5 * (
                    np.exp(-alpha * (lr - start))
                    + np.exp(-alpha * (rl - start))
                )
            expected[left, right] = expected[right, left] = score - tail
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_full_dataset_greedy_matches_dense_reference():
    top_ids = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 4],
            [2, 3, 0, 4],
            [3, 2, 4, 0],
            [4, 3, 1, 2],
        ],
        dtype=np.int32,
    )
    graph, _ = build_rank_delta_csr(top_ids, start_rank=1, alpha=0.2)
    selected, marginal = full_dataset_greedy(
        graph.indptr, graph.indices, graph.data, seed=0, batch_size=4
    )
    dense = graph.toarray()
    reference = [0]
    reference_marginal = []
    while len(reference) < 4:
        pair_values = dense[:, reference]
        feasible = ~(pair_values < 0).any(axis=1)
        gain = np.where(pair_values > 0, pair_values, 0).sum(axis=1)
        feasible[reference] = False
        gain[~feasible] = -np.inf
        item = int(np.argmax(gain))
        reference.append(item)
        reference_marginal.append(float(gain[item]))
    np.testing.assert_array_equal(selected, reference)
    np.testing.assert_allclose(marginal[1:], reference_marginal, atol=1e-7)


def test_full_dataset_greedy_hard_excludes_unsafe_candidate_despite_gain():
    scipy = pytest.importorskip("scipy.sparse")
    dense = np.asarray(
        [
            [0.0, 2.0, -0.1, 0.0],
            [2.0, 0.0, 9.0, 0.8],
            [-0.1, 9.0, 0.0, 10.0],
            [0.0, 0.8, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    graph = scipy.csr_matrix(dense)
    selected, _, diagnostics = full_dataset_greedy(
        graph.indptr,
        graph.indices,
        graph.data,
        seed=0,
        batch_size=3,
        return_diagnostics=True,
    )
    # Case 2 would have much larger aggregate gain after case 1, but its
    # unsafe relation with seed 0 makes it permanently ineligible.
    np.testing.assert_array_equal(selected, [0, 1, 3])
    assert diagnostics["hard_prefix_exclusion_satisfied"] is True
    assert diagnostics["fallback_selection_count"] == 0


def test_full_dataset_greedy_uses_old_gain_only_when_no_safe_candidate_exists():
    scipy = pytest.importorskip("scipy.sparse")
    dense = np.asarray(
        [
            [0.0, -0.1, -0.1],
            [-0.1, 0.0, 1.0],
            [-0.1, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    graph = scipy.csr_matrix(dense)
    selected, _, diagnostics = full_dataset_greedy(
        graph.indptr,
        graph.indices,
        graph.data,
        seed=0,
        batch_size=2,
        return_diagnostics=True,
    )
    np.testing.assert_array_equal(selected, [0, 1])
    assert diagnostics["hard_prefix_exclusion_satisfied"] is False
    assert diagnostics["fallback_selection_count"] == 1
    assert diagnostics["fallback_first_position"] == 1


def test_b3_schedule_preserves_complete_batch_inventory_each_pass():
    counts = np.asarray([2, 1, 3])
    schedule = make_b3_proportional_schedule(counts, max_steps=14, seed=42)
    assert len(schedule) == 14
    np.testing.assert_array_equal(np.bincount(schedule[:6], minlength=3), counts)
    np.testing.assert_array_equal(np.bincount(schedule[6:12], minlength=3), counts)
    np.testing.assert_array_equal(
        schedule, make_b3_proportional_schedule(counts, max_steps=14, seed=42)
    )


def test_invalid_controls_fail_before_preprocessing_or_sampling():
    with pytest.raises(ValueError):
        rank_weight(np.asarray([30]), start_rank=30, alpha=0)
    with pytest.raises(ValueError):
        make_b3_proportional_schedule([1, 2], max_steps=0, seed=42)


def test_b3_sized_seed_sets_cycle_once_per_complete_schedule_pass():
    batch_counts = [2, 1, 3]
    seeds = [list(range(count)) for count in batch_counts]
    schedule = make_b3_proportional_schedule(batch_counts, max_steps=12, seed=42)
    positions = [0] * len(seeds)
    consumed = [[] for _ in seeds]
    for dataset_id in schedule:
        dataset_id = int(dataset_id)
        position = positions[dataset_id]
        consumed[dataset_id].append(
            seeds[dataset_id][position % len(seeds[dataset_id])]
        )
        positions[dataset_id] += 1
    for dataset_id, count in enumerate(batch_counts):
        assert consumed[dataset_id] == list(range(count)) * 2


def test_reproduced_b3_complete_batch_inventory_is_996_steps():
    counts = [
        97, 47, 8, 7, 19, 8, 16, 38, 23, 27,
        68, 97, 25, 97, 97, 97, 97, 15, 16, 97,
    ]
    assert sum(counts) == 996
    schedule = make_b3_proportional_schedule(counts, max_steps=1992, seed=42)
    np.testing.assert_array_equal(
        np.bincount(schedule[:996], minlength=20), counts
    )
    np.testing.assert_array_equal(
        np.bincount(schedule[996:], minlength=20), counts
    )
