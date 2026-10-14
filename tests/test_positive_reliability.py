import unittest

import torch

from src.training.positive_reliability import (
    positive_reliability_from_similarity,
    positive_reliability_weighted_info_nce,
)


class PositiveReliabilityTests(unittest.TestCase):
    def test_no_violation_keeps_full_weight(self):
        similarity = torch.tensor([[0.8, 0.7, 0.2, -0.1]])
        result = positive_reliability_from_similarity(
            similarity,
            torch.tensor([0]),
        )
        self.assertEqual(result["violation_count"].item(), 0)
        self.assertEqual(result["violation_ratio"].item(), 0.0)
        self.assertEqual(result["violation_magnitude"].item(), 0.0)
        self.assertEqual(result["score"].item(), 0.0)
        self.assertEqual(result["weight"].item(), 1.0)

    def test_ratio_times_conditional_magnitude_has_expected_scale(self):
        # Positive is 0.5. Two of four negatives exceed it, by 0.1 and 0.3.
        # ratio=0.5, conditional magnitude=0.2, score=0.5*0.2/0.02=5.
        similarity = torch.tensor([[0.5, 0.6, 0.8, 0.4, 0.1]])
        result = positive_reliability_from_similarity(
            similarity,
            torch.tensor([0]),
            temperature=0.02,
            min_weight=0.1,
            scale=1.0,
        )
        self.assertAlmostEqual(result["violation_ratio"].item(), 0.5, places=6)
        self.assertAlmostEqual(
            result["violation_magnitude"].item(), 0.2, places=6
        )
        self.assertAlmostEqual(result["score"].item(), 5.0, places=5)
        expected = max(torch.exp(torch.tensor(-5.0)).item(), 0.1)
        self.assertAlmostEqual(result["weight"].item(), expected, places=6)

    def test_moderate_anomaly_uses_exponential_before_floor(self):
        # Two of four negatives exceed the positive, by 0.01 and 0.03.
        # With scale=1: score=0.5*0.02/0.02=0.5 and weight=exp(-0.5).
        similarity = torch.tensor([[0.5, 0.51, 0.53, 0.4, 0.1]])
        result = positive_reliability_from_similarity(
            similarity,
            torch.tensor([0]),
            temperature=0.02,
            min_weight=0.1,
            scale=1.0,
        )
        self.assertAlmostEqual(result["score"].item(), 0.5, places=5)
        self.assertAlmostEqual(
            result["weight"].item(),
            torch.exp(torch.tensor(-0.5)).item(),
            places=6,
        )

    def test_more_or_larger_violations_reduce_weight(self):
        similarity = torch.tensor(
            [
                [0.5, 0.51, 0.2, 0.1],
                [0.5, 0.60, 0.2, 0.1],
                [0.5, 0.60, 0.70, 0.1],
            ]
        )
        result = positive_reliability_from_similarity(
            similarity,
            torch.tensor([0, 0, 0]),
            scale=0.1,
        )
        weights = result["weight"]
        self.assertGreater(weights[0].item(), weights[1].item())
        self.assertGreater(weights[1].item(), weights[2].item())

    def test_extreme_mismatch_approaches_but_does_not_cross_floor(self):
        similarity = torch.tensor([[0.1, 0.8, 0.7, 0.6]])
        result = positive_reliability_from_similarity(
            similarity,
            torch.tensor([0]),
            min_weight=0.1,
        )
        weight = result["weight"].item()
        self.assertAlmostEqual(weight, 0.1, places=6)

    def test_all_candidate_magnitude_identity(self):
        similarity = torch.tensor([[0.5, 0.6, 0.8, 0.4, 0.1]])
        result = positive_reliability_from_similarity(
            similarity,
            torch.tensor([0]),
            temperature=0.02,
            scale=1.0,
        )
        all_negative_mean_positive_gap = (0.1 + 0.3) / 4
        self.assertAlmostEqual(
            result["score"].item(),
            all_negative_mean_positive_gap / 0.02,
            places=5,
        )

    def test_weighted_info_nce_is_normalized_by_weight_sum(self):
        query = torch.tensor([[1.0, 0.0], [0.8, 0.2]], requires_grad=True)
        target = torch.tensor([[0.6, 0.4], [1.0, 0.0]], requires_grad=True)
        loss, diagnostics = positive_reliability_weighted_info_nce(
            query, target, scale=1.0
        )
        similarity = query.float() @ target.float().T
        per_case = torch.nn.functional.cross_entropy(
            similarity / 0.02, torch.arange(2), reduction="none"
        )
        expected = (
            diagnostics["weight"] * per_case
        ).sum() / diagnostics["weight"].sum()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_reliability_weight_has_no_gradient_path(self):
        query = torch.tensor([[1.0, 0.0], [0.8, 0.2]], requires_grad=True)
        target = torch.tensor([[0.6, 0.4], [1.0, 0.0]], requires_grad=True)
        _, diagnostics = positive_reliability_weighted_info_nce(query, target)
        self.assertFalse(diagnostics["weight"].requires_grad)

    def test_weighted_info_nce_rejects_unaligned_batches(self):
        with self.assertRaisesRegex(ValueError, "exactly one aligned target"):
            positive_reliability_weighted_info_nce(
                torch.randn(2, 4), torch.randn(3, 4)
            )


if __name__ == "__main__":
    unittest.main()
