import unittest

import torch
import torch.nn.functional as F

from src.training.ega import ega_gradients_and_loss
from src.training.grad_cache.grad_cache import GradCache
from src.training.loss import SimpleContrastiveLoss


class EGAGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.temperature = 0.07
        self.query = F.normalize(torch.randn(7, 11), dim=1)
        self.target = F.normalize(torch.randn(7, 11), dim=1)

    def test_alpha_zero_matches_info_nce_autograd(self):
        query = self.query.clone().requires_grad_(True)
        target = self.target.clone().requires_grad_(True)
        labels = torch.arange(query.size(0))
        loss = F.cross_entropy(
            (query @ target.transpose(0, 1)) / self.temperature,
            labels,
        )
        loss.backward()

        grad_query, grad_target, ega_loss, diagnostics = ega_gradients_and_loss(
            self.query,
            self.target,
            temperature=self.temperature,
            alpha=0.0,
            return_diagnostics=True,
        )
        batch_size = self.query.size(0)
        torch.testing.assert_close(ega_loss, loss.detach(), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            grad_query / batch_size, query.grad, rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            grad_target / batch_size, target.grad, rtol=2e-5, atol=2e-6
        )
        self.assertAlmostEqual(
            diagnostics["query_gradient_norm_ratio_mean"], 1.0, places=6
        )
        self.assertAlmostEqual(
            diagnostics["query_gradient_cosine_mean"], 1.0, places=6
        )

    def test_negative_probability_mass_is_preserved(self):
        _, _, _, diagnostics = ega_gradients_and_loss(
            self.query,
            self.target,
            temperature=0.02,
            alpha=20.0,
            return_diagnostics=True,
        )
        self.assertLess(diagnostics["negative_mass_max_error"], 2e-6)
        self.assertLess(diagnostics["weight_row_sum_max_error"], 2e-6)

    def test_alpha_focuses_the_negative_distribution(self):
        _, _, _, standard = ega_gradients_and_loss(
            self.query,
            self.target,
            temperature=self.temperature,
            alpha=0.0,
            return_diagnostics=True,
        )
        _, _, _, amplified = ega_gradients_and_loss(
            self.query,
            self.target,
            temperature=self.temperature,
            alpha=20.0,
            return_diagnostics=True,
        )
        self.assertGreater(
            amplified["top1_negative_share_mean"],
            standard["top1_negative_share_mean"],
        )
        self.assertLess(
            amplified["effective_negatives_mean"],
            standard["effective_negatives_mean"],
        )

    def test_large_alpha_stays_finite(self):
        grad_query, grad_target, loss, diagnostics = ega_gradients_and_loss(
            self.query,
            self.target,
            temperature=0.02,
            alpha=1000.0,
            return_diagnostics=True,
        )
        self.assertTrue(torch.isfinite(grad_query).all())
        self.assertTrue(torch.isfinite(grad_target).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(diagnostics["negative_mass_max_error"], 2e-6)

    def test_qwen2_embedding_width_and_bf16_cache_are_finite(self):
        """Exercise Qwen2-VL's 1536D normalized representation contract."""
        generator = torch.Generator().manual_seed(20260816)
        query = F.normalize(
            torch.randn(8, 1536, generator=generator), dim=1
        ).to(torch.bfloat16)
        target = F.normalize(
            torch.randn(8, 1536, generator=generator), dim=1
        ).to(torch.bfloat16)
        grad_query, grad_target, loss, diagnostics = ega_gradients_and_loss(
            query,
            target,
            temperature=0.02,
            alpha=20.0,
            return_diagnostics=True,
        )
        self.assertEqual(grad_query.shape, query.shape)
        self.assertEqual(grad_target.shape, target.shape)
        self.assertEqual(grad_query.dtype, torch.float32)
        self.assertEqual(grad_target.dtype, torch.float32)
        self.assertTrue(torch.isfinite(grad_query).all())
        self.assertTrue(torch.isfinite(grad_target).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(diagnostics["negative_mass_max_error"], 2e-6)

    def test_grad_cache_alpha_zero_matches_existing_cache(self):
        loss_fn = SimpleContrastiveLoss(temperature=self.temperature)
        common = dict(
            models=[torch.nn.Identity(), torch.nn.Identity()],
            chunk_sizes=[8, 8],
            loss_fn=loss_fn,
            dynamic_limit=1,
        )
        standard_cache = GradCache(**common)
        ega_cache = GradCache(
            **common,
            embedding_gradient_strategy="ega",
            ega_alpha=0.0,
        )
        standard_grad, standard_loss = standard_cache.build_cache(
            self.query, self.target
        )
        ega_grad, ega_loss = ega_cache.build_cache(self.query, self.target)

        torch.testing.assert_close(ega_loss, standard_loss, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            ega_grad[0], standard_grad[0], rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            ega_grad[1], standard_grad[1], rtol=2e-5, atol=2e-6
        )

    def test_ddp_cache_scaling_matches_global_mean_update(self):
        """Simulate DDP's average of two rank-local replay gradients."""
        world_size = 2
        local_batch = 4
        input_dim = 5
        output_dim = 7
        query_input = torch.randn(world_size * local_batch, input_dim)
        target_input = torch.randn(world_size * local_batch, input_dim)
        initial_weight = torch.randn(input_dim, output_dim)

        global_weight = initial_weight.clone().requires_grad_(True)
        global_query = F.normalize(query_input @ global_weight, dim=1)
        global_target = F.normalize(target_input @ global_weight, dim=1)
        labels = torch.arange(world_size * local_batch)
        global_loss = F.cross_entropy(
            (global_query @ global_target.T) / self.temperature, labels
        )
        global_loss.backward()

        cached_query, cached_target, _, _ = ega_gradients_and_loss(
            global_query.detach(),
            global_target.detach(),
            temperature=self.temperature,
            alpha=0.0,
        )
        rank_parameter_grads = []
        for rank in range(world_size):
            start = rank * local_batch
            stop = start + local_batch
            rank_weight = initial_weight.clone().requires_grad_(True)
            rank_query = F.normalize(query_input[start:stop] @ rank_weight, dim=1)
            rank_target = F.normalize(target_input[start:stop] @ rank_weight, dim=1)
            surrogate = (
                rank_query * (cached_query[start:stop] / local_batch)
            ).sum() + (
                rank_target * (cached_target[start:stop] / local_batch)
            ).sum()
            surrogate.backward()
            rank_parameter_grads.append(rank_weight.grad)

        ddp_averaged_gradient = torch.stack(rank_parameter_grads).mean(dim=0)
        torch.testing.assert_close(
            ddp_averaged_gradient,
            global_weight.grad,
            rtol=3e-5,
            atol=3e-6,
        )


if __name__ == "__main__":
    unittest.main()
