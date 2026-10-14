import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.training.glofnd_core import (
    GloFNDHyperparameters,
    PerAnchorGloFNDState,
    glofnd_filtered_info_nce,
)


class GloFNDCoreTest(unittest.TestCase):
    def setUp(self):
        self.hyperparameters = GloFNDHyperparameters(
            alpha=0.25,
            threshold_lr=0.05,
            beta1=0.9,
            beta2=0.98,
            eps=1e-8,
        )

    def test_first_visit_uses_batch_quantile_immediately(self):
        similarity = torch.tensor(
            [
                [0.9, 0.1, 0.3, 0.2],
                [0.4, 0.8, 0.2, 0.1],
                [0.0, 0.6, 0.7, 0.2],
                [0.2, 0.1, 0.5, 0.9],
            ],
            dtype=torch.float32,
        )
        ids = torch.tensor([7, 2, 9, 4])
        state = PerAnchorGloFNDState(12, self.hyperparameters)
        keep, diagnostics = state.update_and_get_negative_mask(similarity, ids)
        off_diagonal = ~torch.eye(4, dtype=torch.bool)
        negatives = similarity.masked_select(off_diagonal).view(4, 3)
        expected_threshold = torch.quantile(negatives, 0.75, dim=1)
        expected_keep = off_diagonal & (similarity <= expected_threshold[:, None])

        torch.testing.assert_close(diagnostics["threshold"], expected_threshold)
        self.assertTrue(torch.equal(keep, expected_keep))
        torch.testing.assert_close(state.threshold[ids], expected_threshold)
        self.assertTrue(torch.equal(state.first_moment[ids], torch.zeros(4)))
        self.assertTrue(torch.equal(state.second_moment[ids], torch.zeros(4)))
        self.assertTrue(torch.equal(state.visit_count[ids], torch.ones(4, dtype=torch.int32)))
        self.assertEqual(state.initialized_total, 4)

    def test_second_visit_matches_adam_subgradient_with_visit_clock(self):
        similarity = torch.tensor(
            [
                [0.9, 0.1, 0.3, 0.2],
                [0.4, 0.8, 0.2, 0.1],
                [0.0, 0.6, 0.7, 0.2],
                [0.2, 0.1, 0.5, 0.9],
            ],
            dtype=torch.float32,
        )
        ids = torch.tensor([0, 1, 2, 3])
        state = PerAnchorGloFNDState(4, self.hyperparameters)
        state.update_and_get_negative_mask(similarity, ids)
        old_threshold = state.threshold.clone()

        off_diagonal = ~torch.eye(4, dtype=torch.bool)
        exceed_fraction = (
            ((similarity > old_threshold[:, None]) & off_diagonal).sum(1).float() / 3
        )
        gradient = self.hyperparameters.alpha - exceed_fraction
        moment = (1 - self.hyperparameters.beta1) * gradient
        variance = (1 - self.hyperparameters.beta2) * gradient.square()
        expected = old_threshold - self.hyperparameters.threshold_lr * (
            moment / (1 - self.hyperparameters.beta1**2)
        ) / (
            (variance / (1 - self.hyperparameters.beta2**2)).sqrt()
            + self.hyperparameters.eps
        )
        expected.clamp_(-1, 1)

        _, diagnostics = state.update_and_get_negative_mask(similarity, ids)
        torch.testing.assert_close(diagnostics["threshold"], expected)
        torch.testing.assert_close(state.first_moment, moment)
        torch.testing.assert_close(state.second_moment, variance)
        self.assertTrue(torch.equal(state.visit_count, torch.full((4,), 2, dtype=torch.int32)))

    def test_adam_overshoot_recalibrates_only_empty_rows(self):
        hyperparameters = GloFNDHyperparameters(
            alpha=0.0005,
            threshold_lr=0.05,
            beta1=0.9,
            beta2=0.98,
            eps=1e-8,
        )
        state = PerAnchorGloFNDState(4, hyperparameters)
        state.threshold.fill_(0.12)
        state.visit_count.fill_(1)
        state.initialized.fill_(True)
        state.initialized_total = 4
        similarity = torch.tensor(
            [
                [0.9, 0.100, 0.105, 0.110],
                [0.100, 0.9, 0.105, 0.110],
                [0.100, 0.105, 0.9, 0.110],
                [0.100, 0.105, 0.110, 0.9],
            ],
            dtype=torch.float32,
        )

        keep, diagnostics = state.update_and_get_negative_mask(
            similarity, torch.arange(4)
        )

        self.assertTrue(torch.all(diagnostics["empty_filter_fallback"]))
        self.assertTrue(torch.all(diagnostics["retained_count"] > 0))
        self.assertTrue(torch.all(diagnostics["proposed_threshold"] < 0.100))
        torch.testing.assert_close(
            diagnostics["threshold"], diagnostics["batch_quantile"]
        )
        self.assertTrue(torch.equal(
            keep,
            (~torch.eye(4, dtype=torch.bool))
            & (similarity <= diagnostics["threshold"][:, None]),
        ))

    def test_loss_equals_explicit_masked_infonce(self):
        query = F.normalize(torch.randn(4, 8, generator=torch.Generator().manual_seed(3)), dim=-1).requires_grad_()
        target = F.normalize(torch.randn(4, 8, generator=torch.Generator().manual_seed(5)), dim=-1).requires_grad_()
        ids = torch.tensor([2, 4, 6, 8])
        state = PerAnchorGloFNDState(10, self.hyperparameters)
        loss, diagnostics = glofnd_filtered_info_nce(query, target, ids, state, 0.02)
        similarity = query.float() @ target.float().T
        allowed = torch.eye(4, dtype=torch.bool) | ~diagnostics["filtered_mask"]
        reference = F.cross_entropy(
            (similarity / 0.02).masked_fill(~allowed, float("-inf")),
            torch.arange(4),
        )
        torch.testing.assert_close(loss, reference)
        unfiltered_reference = F.cross_entropy(similarity / 0.02, torch.arange(4))
        torch.testing.assert_close(diagnostics["unfiltered_loss"], unfiltered_reference)
        loss.backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(target.grad).all())

    def test_state_round_trip_and_strict_hyperparameters(self):
        state = PerAnchorGloFNDState(5, self.hyperparameters)
        similarity = torch.eye(3) + 0.1
        state.update_and_get_negative_mask(similarity, torch.tensor([0, 2, 4]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(state.state_dict(), path)
            loaded = torch.load(path, weights_only=False)
        restored = PerAnchorGloFNDState(5, self.hyperparameters)
        restored.load_state_dict(loaded)
        for name in ("threshold", "first_moment", "second_moment", "visit_count", "initialized"):
            self.assertTrue(torch.equal(getattr(restored, name), getattr(state, name)))
        incompatible = PerAnchorGloFNDState(
            5,
            GloFNDHyperparameters(alpha=0.1, threshold_lr=0.05, beta1=0.9, beta2=0.98),
        )
        with self.assertRaisesRegex(RuntimeError, "hyperparameters differ"):
            incompatible.load_state_dict(loaded)
        corrupted = dict(loaded)
        corrupted["second_moment"] = loaded["second_moment"].clone()
        corrupted["second_moment"][0] = -1.0
        with self.assertRaisesRegex(RuntimeError, "negative Adam second moments"):
            PerAnchorGloFNDState(5, self.hyperparameters).load_state_dict(
                corrupted
            )

    def test_duplicate_ids_are_rejected(self):
        state = PerAnchorGloFNDState(4, self.hyperparameters)
        with self.assertRaisesRegex(ValueError, "unique anchor IDs"):
            state.update_and_get_negative_mask(torch.eye(3), torch.tensor([0, 0, 2]))

    def test_lazy_initialization_matches_unclamped_official_quantile(self):
        similarity = torch.tensor(
            [
                [1.0, 0.999, 1.003],
                [0.999, 1.0, 1.003],
                [1.004, 0.999, 1.0],
            ]
        )
        state = PerAnchorGloFNDState(3, self.hyperparameters)
        _, diagnostics = state.update_and_get_negative_mask(
            similarity, torch.arange(3)
        )
        self.assertGreater(float(diagnostics["batch_quantile"].max()), 1.0)
        torch.testing.assert_close(
            diagnostics["threshold"], diagnostics["batch_quantile"]
        )
        restored = PerAnchorGloFNDState(3, self.hyperparameters)
        restored.load_state_dict(state.state_dict())
        torch.testing.assert_close(restored.threshold, state.threshold)


if __name__ == "__main__":
    unittest.main()
