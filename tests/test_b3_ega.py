import unittest

import torch
import torch.nn.functional as F

from src.training.ega import b3_ega_gradients_and_loss, ega_gradients_and_loss
from src.training.grad_cache.grad_cache import GradCache
from src.training.loss import SimpleContrastiveLoss


class B3EGAGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260904)
        self.batch_size = 7
        self.target_per_query = 6
        self.hidden_size = 13
        self.temperature = 0.07
        self.query = F.normalize(
            torch.randn(self.batch_size, self.hidden_size), dim=1
        )
        self.target = F.normalize(
            torch.randn(
                self.batch_size * self.target_per_query, self.hidden_size
            ),
            dim=1,
        )

    def test_alpha_zero_matches_rectangular_b3_info_nce_autograd(self):
        query = self.query.clone().requires_grad_(True)
        target = self.target.clone().requires_grad_(True)
        labels = torch.arange(self.batch_size) * self.target_per_query
        loss = F.cross_entropy(
            (query @ target.transpose(0, 1)) / self.temperature,
            labels,
        )
        loss.backward()

        grad_query, grad_target, b3_ega_loss, diagnostics = (
            b3_ega_gradients_and_loss(
                self.query,
                self.target,
                target_per_query=self.target_per_query,
                temperature=self.temperature,
                alpha=0.0,
                return_diagnostics=True,
            )
        )
        torch.testing.assert_close(
            b3_ega_loss, loss.detach(), rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            grad_query / self.batch_size, query.grad, rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            grad_target / self.batch_size, target.grad, rtol=2e-5, atol=2e-6
        )
        self.assertAlmostEqual(
            diagnostics["query_gradient_norm_ratio_mean"], 1.0, places=6
        )
        self.assertAlmostEqual(
            diagnostics["query_gradient_cosine_mean"], 1.0, places=6
        )

    def test_alpha_twenty_preserves_mass_and_focuses_negatives(self):
        _, _, _, standard = b3_ega_gradients_and_loss(
            self.query,
            self.target,
            target_per_query=self.target_per_query,
            temperature=0.02,
            alpha=0.0,
            return_diagnostics=True,
        )
        grad_query, grad_target, loss, amplified = b3_ega_gradients_and_loss(
            self.query,
            self.target,
            target_per_query=self.target_per_query,
            temperature=0.02,
            alpha=20.0,
            return_diagnostics=True,
        )
        self.assertTrue(torch.isfinite(grad_query).all())
        self.assertTrue(torch.isfinite(grad_target).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(amplified["negative_mass_max_error"], 2e-6)
        self.assertLess(amplified["weight_row_sum_max_error"], 2e-6)
        self.assertGreater(
            amplified["top1_negative_share_mean"],
            standard["top1_negative_share_mean"],
        )
        self.assertLess(
            amplified["effective_negatives_mean"],
            standard["effective_negatives_mean"],
        )
        self.assertAlmostEqual(
            amplified["explicit_b3_negative_share_mean"]
            + amplified["ambient_inbatch_negative_share_mean"],
            1.0,
            places=6,
        )

    def test_ordinary_ega_contract_stays_isolated(self):
        with self.assertRaisesRegex(ValueError, "exactly one aligned target"):
            ega_gradients_and_loss(
                self.query,
                self.target,
                temperature=self.temperature,
                alpha=20.0,
            )

    def test_grad_cache_alpha_zero_matches_existing_b3_cache(self):
        loss_fn = SimpleContrastiveLoss(temperature=self.temperature)
        common = dict(
            models=[torch.nn.Identity(), torch.nn.Identity()],
            chunk_sizes=[8, 8],
            loss_fn=loss_fn,
            dynamic_limit=1,
        )
        standard_cache = GradCache(**common)
        b3_ega_cache = GradCache(
            **common,
            embedding_gradient_strategy="b3_ega",
            ega_alpha=0.0,
        )
        standard_grad, standard_loss = standard_cache.build_cache(
            self.query, self.target
        )
        b3_ega_grad, b3_ega_loss = b3_ega_cache.build_cache(
            self.query, self.target
        )

        torch.testing.assert_close(
            b3_ega_loss, standard_loss, rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            b3_ega_grad[0], standard_grad[0], rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            b3_ega_grad[1], standard_grad[1], rtol=2e-5, atol=2e-6
        )

    def test_simulated_ddp_cache_scaling_matches_global_b3_mean(self):
        world_size = 2
        local_queries = 3
        target_per_query = self.target_per_query
        input_dim = 5
        output_dim = 9
        generator = torch.Generator().manual_seed(91)
        query_input = torch.randn(
            world_size * local_queries, input_dim, generator=generator
        )
        target_input = torch.randn(
            world_size * local_queries * target_per_query,
            input_dim,
            generator=generator,
        )
        initial_weight = torch.randn(
            input_dim, output_dim, generator=generator
        )

        global_weight = initial_weight.clone().requires_grad_(True)
        global_query = F.normalize(query_input @ global_weight, dim=1)
        global_target = F.normalize(target_input @ global_weight, dim=1)
        labels = torch.arange(world_size * local_queries) * target_per_query
        global_loss = F.cross_entropy(
            (global_query @ global_target.T) / self.temperature, labels
        )
        global_loss.backward()

        cached_query, cached_target, _, _ = b3_ega_gradients_and_loss(
            global_query.detach(),
            global_target.detach(),
            target_per_query=target_per_query,
            temperature=self.temperature,
            alpha=0.0,
        )
        rank_parameter_grads = []
        for rank in range(world_size):
            query_start = rank * local_queries
            query_stop = query_start + local_queries
            target_start = rank * local_queries * target_per_query
            target_stop = target_start + local_queries * target_per_query
            rank_weight = initial_weight.clone().requires_grad_(True)
            rank_query = F.normalize(
                query_input[query_start:query_stop] @ rank_weight, dim=1
            )
            rank_target = F.normalize(
                target_input[target_start:target_stop] @ rank_weight, dim=1
            )
            surrogate = (
                rank_query
                * (cached_query[query_start:query_stop] / local_queries)
            ).sum() + (
                rank_target
                * (cached_target[target_start:target_stop] / local_queries)
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

    def test_requires_exact_b3_six_target_layout(self):
        cache = GradCache(
            models=[torch.nn.Identity(), torch.nn.Identity()],
            chunk_sizes=[8, 8],
            loss_fn=SimpleContrastiveLoss(temperature=self.temperature),
            dynamic_limit=1,
            embedding_gradient_strategy="b3_ega",
            ega_alpha=20.0,
        )
        with self.assertRaisesRegex(RuntimeError, "five hard negative"):
            cache.build_cache(self.query, self.target[:-1])


if __name__ == "__main__":
    unittest.main()
