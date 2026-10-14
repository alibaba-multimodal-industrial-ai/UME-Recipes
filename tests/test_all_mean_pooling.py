from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

# Allow both ``python tests/test_all_mean_pooling.py`` and unittest discovery.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.model import MMEBModel


class _DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()


def _model(normalize=False):
    return MMEBModel(
        encoder=_DummyEncoder(),
        pooling="mean",
        normalize=normalize,
    )


class AllMeanPoolingTest(unittest.TestCase):
    def test_excludes_left_and_right_padding(self):
        hidden = torch.tensor(
            [
                [[float("nan"), float("nan")], [1.0, 3.0], [3.0, 5.0]],
                [[2.0, 4.0], [4.0, 8.0], [float("nan"), float("nan")]],
            ],
            dtype=torch.bfloat16,
        )
        mask = torch.tensor([[0, 1, 1], [1, 1, 0]])

        actual = _model()._pooling(hidden, mask)

        expected = torch.tensor([[2.0, 4.0], [3.0, 6.0]], dtype=torch.float32)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.dtype, torch.float32)

    def test_normalizes_after_averaging(self):
        hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])
        mask = torch.ones((1, 2), dtype=torch.long)

        actual = _model(normalize=True)._pooling(hidden, mask)

        torch.testing.assert_close(
            torch.linalg.vector_norm(actual, dim=-1), torch.ones(1)
        )

    def test_rejects_empty_rows(self):
        hidden = torch.zeros((1, 2, 3))
        mask = torch.zeros((1, 2), dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "empty input rows"):
            _model()._pooling(hidden, mask)

    def test_backpropagates_only_through_active_tokens(self):
        hidden = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]]], requires_grad=True
        )
        mask = torch.tensor([[1, 1, 0]])

        _model()._pooling(hidden, mask).sum().backward()

        expected_gradient = torch.tensor(
            [[[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]]]
        )
        torch.testing.assert_close(hidden.grad, expected_gradient)

    def test_existing_last_pooling_is_unchanged(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]]])
        mask = torch.tensor([[1, 1, 0]])
        model = MMEBModel(
            encoder=_DummyEncoder(),
            pooling="eos",
            normalize=False,
        )

        actual = model._pooling(hidden, mask)

        torch.testing.assert_close(actual, torch.tensor([[3.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
