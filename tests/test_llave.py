import unittest

import torch
import torch.nn.functional as F

from src.training.grad_cache.grad_cache import GradCache
from src.training.llave import llave_loss_from_similarity
from src.training.loss import LLaVESimpleContrastiveLoss, SimpleContrastiveLoss


def official_weighted_clip_reference(similarity, labels, temperature, alpha):
    """Literal, independent translation of the released WeightedClipLoss."""
    reverse_labels = torch.empty_like(labels)
    reverse_labels.scatter_(0, labels, torch.arange(labels.numel()))
    logit_scale = 1.0 / temperature

    def one_direction(scores, direction_labels):
        logits = logit_scale * scores
        reward = torch.log(
            torch.exp(alpha * logits.detach() / logit_scale)
        )
        reward.scatter_(1, direction_labels[:, None], 0.0)
        return F.cross_entropy(logits + reward, direction_labels)

    return (
        one_direction(similarity, labels)
        + one_direction(similarity.T, reverse_labels)
    ) / 2


class LLaVELossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260903)
        self.temperature = 0.02
        self.alpha = 9.0
        self.query = F.normalize(torch.randn(7, 13), dim=1)
        self.target = F.normalize(torch.randn(7, 13), dim=1)
        self.labels = torch.arange(7)

    def test_matches_official_bidirectional_weighted_clip_loss(self):
        similarity = self.query @ self.target.T
        actual, _ = llave_loss_from_similarity(
            similarity,
            self.labels,
            temperature=self.temperature,
            alpha=self.alpha,
        )
        expected = official_weighted_clip_reference(
            similarity,
            self.labels,
            self.temperature,
            self.alpha,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)

    def test_alpha_zero_matches_symmetric_clip_loss_and_gradients(self):
        standard_q = self.query.clone().requires_grad_(True)
        standard_t = self.target.clone().requires_grad_(True)
        standard_similarity = standard_q @ standard_t.T
        standard_loss = (
            F.cross_entropy(standard_similarity / self.temperature, self.labels)
            + F.cross_entropy(standard_similarity.T / self.temperature, self.labels)
        ) / 2
        standard_loss.backward()

        llave_q = self.query.clone().requires_grad_(True)
        llave_t = self.target.clone().requires_grad_(True)
        llave_loss, _ = llave_loss_from_similarity(
            llave_q @ llave_t.T,
            self.labels,
            temperature=self.temperature,
            alpha=0.0,
        )
        llave_loss.backward()

        torch.testing.assert_close(llave_loss, standard_loss, rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(llave_q.grad, standard_q.grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(llave_t.grad, standard_t.grad, rtol=1e-6, atol=1e-7)

    def test_reward_is_stopped_and_positive_is_unweighted(self):
        actual_similarity = (
            self.query @ self.target.T
        ).detach().clone().requires_grad_(True)
        actual_loss, diagnostics = llave_loss_from_similarity(
            actual_similarity,
            self.labels,
            temperature=self.temperature,
            alpha=self.alpha,
            return_diagnostics=True,
        )
        actual_loss.backward()

        reference_similarity = actual_similarity.detach().clone().requires_grad_(True)
        reference_loss = official_weighted_clip_reference(
            reference_similarity,
            self.labels,
            self.temperature,
            self.alpha,
        )
        reference_loss.backward()

        wrong_similarity = actual_similarity.detach().clone().requires_grad_(True)
        positive_mask = torch.eye(
            wrong_similarity.size(0), dtype=torch.bool, device=wrong_similarity.device
        )
        wrong_query_reward = (self.alpha * wrong_similarity).masked_fill(
            positive_mask, 0.0
        )
        wrong_target_reward = (self.alpha * wrong_similarity.T).masked_fill(
            positive_mask, 0.0
        )
        wrong_loss = (
            F.cross_entropy(
                wrong_similarity / self.temperature + wrong_query_reward,
                self.labels,
            )
            + F.cross_entropy(
                wrong_similarity.T / self.temperature + wrong_target_reward,
                self.labels,
            )
        ) / 2
        wrong_loss.backward()

        torch.testing.assert_close(
            actual_similarity.grad,
            reference_similarity.grad,
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertGreater(
            (actual_similarity.grad - wrong_similarity.grad).abs().max().item(),
            1e-5,
        )
        self.assertEqual(diagnostics["positive_reward_abs_max"], 0.0)
        self.assertIn("query_to_target_loss", diagnostics)
        self.assertIn("target_to_query_loss", diagnostics)

    def test_hardness_weighting_concentrates_negative_probability(self):
        similarity = self.query @ self.target.T
        _, standard = llave_loss_from_similarity(
            similarity,
            self.labels,
            temperature=self.temperature,
            alpha=0.0,
            return_diagnostics=True,
        )
        _, weighted = llave_loss_from_similarity(
            similarity,
            self.labels,
            temperature=self.temperature,
            alpha=self.alpha,
            return_diagnostics=True,
        )
        self.assertGreater(
            weighted["top1_negative_share_mean"],
            standard["top1_negative_share_mean"],
        )
        self.assertLess(
            weighted["effective_negatives_mean"],
            standard["effective_negatives_mean"],
        )

    def test_grad_cache_uses_llave_autograd_path(self):
        loss_fn = LLaVESimpleContrastiveLoss(
            temperature=self.temperature,
            alpha=self.alpha,
        )
        loss_fn.expected_target_per_query = 1
        cache = GradCache(
            models=[torch.nn.Identity(), torch.nn.Identity()],
            chunk_sizes=[7, 7],
            loss_fn=loss_fn,
            dynamic_limit=1,
            embedding_gradient_strategy="llave",
        )
        gradients, loss = cache.build_cache(self.query, self.target)
        self.assertEqual(len(gradients), 2)
        self.assertTrue(torch.isfinite(gradients[0]).all())
        self.assertTrue(torch.isfinite(gradients[1]).all())
        self.assertTrue(torch.isfinite(loss))

    def test_existing_info_nce_class_is_unchanged(self):
        loss_fn = SimpleContrastiveLoss(temperature=self.temperature)
        actual = loss_fn(self.query, self.target, target=self.labels)
        expected = F.cross_entropy(
            (self.query @ self.target.T) / self.temperature,
            self.labels,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bf16_qwen2_width_is_finite(self):
        generator = torch.Generator().manual_seed(20260903)
        query = F.normalize(
            torch.randn(8, 1536, generator=generator), dim=1
        ).to(torch.bfloat16)
        target = F.normalize(
            torch.randn(8, 1536, generator=generator), dim=1
        ).to(torch.bfloat16)
        loss, diagnostics = llave_loss_from_similarity(
            query.float() @ target.float().T,
            torch.arange(8),
            temperature=0.02,
            alpha=9.0,
            return_diagnostics=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in diagnostics.values()))


if __name__ == "__main__":
    unittest.main()
