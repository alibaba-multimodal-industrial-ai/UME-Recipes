import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.training.online_false_negative_risk_core import (
    compact_nonself_neighbors,
    exact_jaccard_matrix,
    false_negative_risk_energy,
    mutual_neighbor_matrix,
    probabilistic_risk_info_nce,
    risk_adjusted_info_nce,
    risk_deletion_probabilities,
    systematic_probability_mask,
)


class OnlineFalseNegativeRiskTests(unittest.TestCase):
    def test_neighbor_compaction_accepts_unused_faiss_slot(self):
        found = np.array([[10, 21, 22, -1], [31, 30, 32, -1]])
        result = compact_nonself_neighbors(found, np.array([10, 30]), top_k=2)
        np.testing.assert_array_equal(result, np.array([[21, 22], [31, 32]]))

    def test_neighbor_compaction_rejects_actual_shortfall(self):
        found = np.array([[10, 21, -1, -1]])
        with self.assertRaisesRegex(RuntimeError, "minimum=1, required=2"):
            compact_nonself_neighbors(found, np.array([10]), top_k=2)

    def test_exact_jaccard(self):
        left = np.array([[1, 2, 3], [4, 5, 6]])
        right = np.array([[2, 3, 7], [4, 8, 9]])
        result = exact_jaccard_matrix(left, right)
        expected = np.array([[0.5, 0.0], [0.0, 0.2]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_directional_mutual_neighbors(self):
        ids = np.array([10, 20])
        anchor_neighbors = np.array([[20, 30], [30, 40]])
        candidate_neighbors = np.array([[30, 40], [10, 50]])
        result = mutual_neighbor_matrix(
            ids, ids, anchor_neighbors, candidate_neighbors
        )
        self.assertTrue(result[0, 1])
        self.assertFalse(result[1, 0])

    def test_energy_formula_and_positive_diagonal(self):
        reps = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
        ids = np.array([10, 20])
        anchor_neighbors = np.array([[20, 30], [10, 40]])
        candidate_neighbors = np.array([[20, 50], [10, 30]])
        result = false_negative_risk_energy(
            reps, ids, ids, anchor_neighbors, candidate_neighbors
        )
        # a=0.8, Jaccard({20,30},{10,30})=1/3, mutual=True.
        expected = 0.8**2 + (1 / 3) ** 2 + 0.5
        self.assertAlmostEqual(result["energy"][0, 1], expected, places=6)
        self.assertEqual(result["energy"][0, 0], 0.0)

    def test_cross_dataset_pairs_are_never_adjusted(self):
        reps = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.8, 0.6]])
        ids = np.array([10, 20, 30])
        neighbors = np.array(
            [[20, 30], [10, 30], [10, 20]], dtype=np.int64
        )
        dataset_ids = np.array([0, 1, 0])
        result = false_negative_risk_energy(
            reps,
            ids,
            ids,
            neighbors,
            neighbors,
            anchor_dataset_ids=dataset_ids,
            candidate_dataset_ids=dataset_ids,
        )
        self.assertEqual(result["energy"][0, 1], 0.0)
        self.assertEqual(result["positive_candidate_similarity"][0, 1], 0.0)
        self.assertGreater(result["energy"][0, 2], 0.0)

    def test_lambda_zero_matches_original_info_nce(self):
        query = F.normalize(torch.tensor([[1.0, 0.0], [0.5, 0.5]]), dim=1)
        target = F.normalize(torch.tensor([[0.9, 0.1], [0.4, 0.6]]), dim=1)
        energy = torch.tensor([[0.0, 1.2], [0.7, 0.0]])
        loss, _ = risk_adjusted_info_nce(
            query, target, energy, temperature=0.02, risk_lambda=0.0
        )
        expected = F.cross_entropy(query @ target.T / 0.02, torch.arange(2))
        self.assertTrue(torch.allclose(loss, expected))

    def test_only_negative_logits_are_adjusted(self):
        query = F.normalize(torch.tensor([[1.0, 0.0], [0.5, 0.5]]), dim=1)
        target = F.normalize(torch.tensor([[0.9, 0.1], [0.4, 0.6]]), dim=1)
        energy = torch.tensor([[9.0, 1.2], [0.7, 9.0]], requires_grad=True)
        _, result = risk_adjusted_info_nce(query, target, energy, risk_lambda=1.0)
        difference = result["plain_logits"] - result["adjusted_logits"]
        self.assertEqual(difference[0, 0].item(), 0.0)
        self.assertEqual(difference[1, 1].item(), 0.0)
        self.assertAlmostEqual(difference[0, 1].item(), 1.2, places=5)
        self.assertAlmostEqual(difference[1, 0].item(), 0.7, places=5)
        self.assertFalse(result["energy"].requires_grad)

    def test_aggressive_probability_mapping_and_row_cap(self):
        energy = torch.full((6, 6), 2.0)
        energy.fill_diagonal_(0.0)
        probability, diagnostics = risk_deletion_probabilities(
            energy,
            risk_lambda=1.5,
            max_pair_drop_probability=0.95,
            max_drop_ratio=0.40,
        )
        self.assertTrue(torch.equal(probability.diag(), torch.zeros(6)))
        self.assertEqual(diagnostics["max_drop_count"], 2)
        self.assertTrue(torch.all(diagnostics["budget_scale"] < 1.0))
        self.assertTrue(torch.all(diagnostics["expected_drop_count"] < 2.0))
        self.assertTrue(torch.all(diagnostics["expected_drop_count"] > 1.99))

    def test_zero_risk_is_never_deleted_and_pair_cap_is_respected(self):
        energy = torch.tensor(
            [
                [0.0, 0.0, 10.0, 0.2],
                [0.0, 0.0, 0.2, 10.0],
                [10.0, 0.2, 0.0, 0.0],
                [0.2, 10.0, 0.0, 0.0],
            ]
        )
        probability, diagnostics = risk_deletion_probabilities(
            energy,
            risk_lambda=1.5,
            max_pair_drop_probability=0.95,
            max_drop_ratio=0.90,
        )
        self.assertTrue(torch.all(probability[energy == 0] == 0))
        self.assertLessEqual(float(probability.max().item()), 0.95)
        self.assertLessEqual(float(diagnostics["raw_probability"].max().item()), 0.95)
        sampled = systematic_probability_mask(probability, seed=91)
        self.assertFalse(sampled[energy == 0].any())

    def test_systematic_mask_is_deterministic_and_low_variance(self):
        probability = torch.tensor(
            [
                [0.0, 0.1, 0.4, 0.7],
                [0.2, 0.0, 0.6, 0.9],
                [0.3, 0.3, 0.0, 0.3],
                [0.8, 0.2, 0.4, 0.0],
            ]
        )
        first = systematic_probability_mask(probability, seed=173)
        second = systematic_probability_mask(probability, seed=173)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(first.diag().any())
        expected = probability.sum(dim=1)
        counts = first.sum(dim=1).float()
        self.assertTrue(torch.all(counts >= expected.floor()))
        self.assertTrue(torch.all(counts <= expected.ceil()))

    def test_probability_deletion_loss_matches_explicit_mask(self):
        query = F.normalize(torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.1, 0.9]]), dim=1)
        target = F.normalize(torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]), dim=1)
        drop_mask = torch.tensor(
            [
                [False, True, False],
                [False, False, True],
                [True, False, False],
            ]
        )
        loss, diagnostics = probabilistic_risk_info_nce(
            query, target, drop_mask, temperature=0.02
        )
        logits = query @ target.T / 0.02
        expected = F.cross_entropy(
            logits.masked_fill(drop_mask, float("-inf")), torch.arange(3)
        )
        torch.testing.assert_close(loss, expected)
        self.assertTrue(torch.equal(diagnostics["drop_mask"], drop_mask))
        self.assertTrue(torch.all(diagnostics["retained_negative_count"] == 1))


if __name__ == "__main__":
    unittest.main()
