import unittest

import numpy as np

from src.training.dynamic_hard_batch_core import (
    exact_farthest_seeds,
    exact_hard_batch,
    pair_vectors,
)


class DynamicHardBatchTests(unittest.TestCase):
    def test_pair_vector_dot_is_four_term_average(self):
        q = np.array([[1., 0.], [0., 1.]])
        p = np.array([[.8, .2], [.3, .7]])
        z = pair_vectors(q, p)
        expected = (q[0] @ q[1] + q[0] @ p[1] + q[1] @ p[0] + p[0] @ p[1]) / 4
        self.assertAlmostEqual(float(z[0] @ z[1]), float(expected))

    def test_farthest_seed_order_is_unique_and_deterministic(self):
        vectors = np.array([[1., 0.], [.9, .1], [-1., 0.], [0., 1.]])
        first = exact_farthest_seeds(vectors, 4, initial_index=0)
        second = exact_farthest_seeds(vectors, 4, initial_index=0)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(first.tolist())), 4)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[1], 2)

    def test_hard_batch_stays_unique_and_contains_seed(self):
        q = np.eye(6, dtype=np.float32)
        p = q.copy()
        q[1] = q[0] * .9 + q[1] * .1
        p[1] = p[0] * .9 + p[1] * .1
        selected, candidates = exact_hard_batch(q, p, seed=0, batch_size=4, candidate_multiplier=2)
        self.assertEqual(selected[0], 0)
        self.assertEqual(len(set(selected.tolist())), 4)
        self.assertNotIn(0, candidates.tolist())


if __name__ == "__main__":
    unittest.main()
