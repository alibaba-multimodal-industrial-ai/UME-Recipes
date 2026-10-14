import unittest

import torch
import torch.nn.functional as F

from src.training.ega import b3_ega_gradients_and_loss


class Qwen35B3EGAContractTests(unittest.TestCase):
    def test_qwen35_hidden_size_rectangular_gradient_contract(self):
        generator = torch.Generator().manual_seed(20260927)
        local_queries = 3
        target_per_query = 6
        hidden_size = 2048
        query = F.normalize(
            torch.randn(local_queries, hidden_size, generator=generator), dim=1
        )
        target = F.normalize(
            torch.randn(
                local_queries * target_per_query,
                hidden_size,
                generator=generator,
            ),
            dim=1,
        )

        grad_query, grad_target, loss, diagnostics = b3_ega_gradients_and_loss(
            query,
            target,
            target_per_query=target_per_query,
            temperature=0.02,
            alpha=20.0,
            return_diagnostics=True,
        )

        self.assertEqual(grad_query.shape, query.shape)
        self.assertEqual(grad_target.shape, target.shape)
        self.assertTrue(torch.isfinite(grad_query).all())
        self.assertTrue(torch.isfinite(grad_target).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(diagnostics["negative_mass_max_error"], 2e-6)
        self.assertLess(diagnostics["weight_row_sum_max_error"], 2e-6)


if __name__ == "__main__":
    unittest.main()
