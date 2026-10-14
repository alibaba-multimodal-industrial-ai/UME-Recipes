import json
import random
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.training.unime_v2 import (
    _official_pair_or_dummy,
    UniMEV2Dataset,
    UniMEV2SoftDistributionLoss,
    expand_unime_v2_row,
    sort_processed_inputs_by_length,
    unime_v2_symmetric_kl,
)
from src.training.grad_cache.grad_cache import GradCache


def _row(index):
    return {
        "query_text": f"q{index}",
        "query_image": "",
        "pos_text": f"p{index}",
        "pos_image": "",
        "query_pos_scores": 0.9,
        "hard_negatives": [[f"h{hard}", ""] for hard in range(10)],
        "hard_negatives_scores": [0.8 - hard * 0.05 for hard in range(10)],
    }


class UniMEV2Tests(unittest.TestCase):
    def test_official_empty_positive_becomes_fixed_layout_dummy_pair(self):
        row = _row(0)
        row["pos_text"] = ""
        x_items, y_items = expand_unime_v2_row(row)
        x_item, y_item, is_dummy = _official_pair_or_dummy(x_items[0], y_items[0])
        self.assertTrue(is_dummy)
        self.assertEqual(x_item, ("  ", "", 0.0))
        self.assertEqual(y_item, ("  ", "", 0.0))

        # Only the invalid positive pair is replaced.  The four pairs carrying
        # the eight hard negatives retain their released ordering and scores.
        remaining = [
            _official_pair_or_dummy(x_item, y_item)
            for x_item, y_item in zip(x_items[1:], y_items[1:])
        ]
        self.assertTrue(all(not item[2] for item in remaining))

    def test_official_row_expansion_and_candidate_score_order(self):
        row = _row(0)
        x_items, y_items = expand_unime_v2_row(row)
        self.assertEqual(
            [item[0] for item in x_items], ["q0", "h0", "h2", "h4", "h6"]
        )
        self.assertEqual(
            [item[0] for item in y_items], ["p0", "h1", "h3", "h5", "h7"]
        )
        reconstructed = [y_items[0][2]]
        for position in range(1, 5):
            reconstructed.extend((x_items[position][2], y_items[position][2]))
        self.assertEqual(reconstructed, [0.9, *row["hard_negatives_scores"][:8]])

    def test_symmetric_kl_matches_released_formula(self):
        model_scores = torch.tensor([[0.7, 0.2, -0.1], [0.1, 0.3, 0.4]])
        teacher_scores = torch.tensor([[0.8, 0.1, -0.2], [0.0, 0.2, 0.5]])
        temperature = 0.2
        actual, pq, qp = unime_v2_symmetric_kl(
            model_scores, teacher_scores, temperature
        )
        teacher_prob = F.softmax(teacher_scores / temperature, dim=-1)
        model_prob = F.softmax(model_scores / temperature, dim=-1)
        expected_pq = F.kl_div(
            torch.log(model_prob + 1e-8), teacher_prob, reduction="batchmean"
        )
        expected_qp = F.kl_div(
            torch.log(teacher_prob + 1e-8), model_prob, reduction="batchmean"
        )
        torch.testing.assert_close(pq, expected_pq)
        torch.testing.assert_close(qp, expected_qp)
        torch.testing.assert_close(actual, 0.5 * (expected_pq + expected_qp))

    def test_loss_reconstructs_positive_then_hn0_through_hn7(self):
        x = torch.zeros(5, 9)
        y = torch.zeros(5, 9)
        x[0, 0] = 1.0
        y[0, 0] = 0.9
        for position in range(1, 5):
            x[position, 0] = 0.8 - (2 * position - 2) * 0.05
            y[position, 0] = 0.8 - (2 * position - 1) * 0.05
        score0 = torch.tensor([0.0, 0.8, 0.7, 0.6, 0.5])
        score1 = torch.tensor([0.9, 0.75, 0.65, 0.55, 0.45])
        loss_fn = UniMEV2SoftDistributionLoss(temperature=0.2, scale_loss=False)
        actual = loss_fn(x, y, score0, score1)
        ordered = torch.tensor(
            [[0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]]
        )
        expected, _, _ = unime_v2_symmetric_kl(ordered, ordered, temperature=0.2)
        torch.testing.assert_close(actual, expected)

    def test_length_sort_and_restore_leave_unime_loss_unchanged(self):
        generator = torch.Generator().manual_seed(7)
        x = F.normalize(torch.randn(10, 6, generator=generator), dim=-1).requires_grad_()
        y = F.normalize(torch.randn(10, 6, generator=generator), dim=-1).requires_grad_()
        score0 = torch.linspace(0.0, 0.8, 10)
        score1 = torch.linspace(0.9, 0.1, 10)
        qry_order = torch.tensor([4, 9, 0, 7, 2, 8, 1, 6, 3, 5])
        tgt_order = torch.tensor([8, 1, 6, 3, 9, 0, 5, 2, 7, 4])
        qry_restore = torch.argsort(qry_order)
        tgt_restore = torch.argsort(tgt_order)

        reference_fn = UniMEV2SoftDistributionLoss(0.2, scale_loss=False)
        packed_fn = UniMEV2SoftDistributionLoss(0.2, scale_loss=False)
        reference = reference_fn(x, y, score0, score1)
        reference.backward()
        reference_x_grad = x.grad.detach().clone()
        reference_y_grad = y.grad.detach().clone()
        packed_x = x.detach().index_select(0, qry_order).requires_grad_()
        packed_y = y.detach().index_select(0, tgt_order).requires_grad_()
        packed = packed_fn(
            packed_x,
            packed_y,
            score0,
            score1,
            qry_restore_order=qry_restore,
            tgt_restore_order=tgt_restore,
        )
        torch.testing.assert_close(packed, reference)
        packed.backward()
        torch.testing.assert_close(
            packed_x.grad.index_select(0, qry_restore), reference_x_grad
        )
        torch.testing.assert_close(
            packed_y.grad.index_select(0, tgt_restore), reference_y_grad
        )

    def test_length_aware_chunks_crop_padding_and_cover_every_row(self):
        # Canonical rows have active lengths [2, 10, 1, 6].  The collator
        # sorts them to [10, 6, 2, 1], and the inverse restores all fields.
        lengths = torch.tensor([2, 10, 1, 6])
        width = 10
        attention_mask = torch.zeros(4, width, dtype=torch.long)
        input_ids = torch.zeros(4, width, dtype=torch.long)
        for row, length in enumerate(lengths.tolist()):
            attention_mask[row, -length:] = 1
            input_ids[row, -length:] = row + 1
        processed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": torch.zeros_like(input_ids),
            "texts": [f"row-{row}" for row in range(4)],
            "images": [None] * 4,
            "pixel_values": [None] * 4,
            "image_grid_thw": [None] * 4,
            "pixel_values_videos": [None] * 4,
            "video_grid_thw": [None] * 4,
            "global_dataset_name": ["UniME-V2"] * 4,
        }
        sorted_inputs, restore = sort_processed_inputs_by_length(processed)
        self.assertEqual(
            sorted_inputs["attention_mask"].sum(dim=1).tolist(), [10, 6, 2, 1]
        )
        self.assertEqual(
            [sorted_inputs["texts"][index] for index in restore.tolist()],
            processed["texts"],
        )

        gc = GradCache(
            models=[None, None],
            chunk_sizes=[8, 8],
            loss_fn=lambda *_args, **_kwargs: torch.tensor(0.0),
            dynamic_limit=4,
        )
        chunks, chunk_sizes, chunk_widths = gc._split_inputs_length_aware(
            {"qry": sorted_inputs}, 8, gc.thresholds_qry
        )
        self.assertEqual(chunk_sizes, [2, 2])
        self.assertEqual(chunk_widths, [10, 2])
        self.assertEqual(sum(chunk_sizes), 4)
        self.assertEqual(
            [tuple(chunk["qry"]["input_ids"].shape) for chunk in chunks],
            [(2, 10), (2, 2)],
        )
        packed_ids = torch.cat(
            [
                chunk["qry"]["input_ids"][
                    chunk["qry"]["attention_mask"].bool()
                ]
                for chunk in chunks
            ]
        )
        expected_ids = sorted_inputs["input_ids"][
            sorted_inputs["attention_mask"].bool()
        ]
        torch.testing.assert_close(packed_ids, expected_ids)

    def test_dataset_uses_official_python_seed_42_shuffle(self):
        rows = [_row(index) for index in range(12)]
        expected = list(rows)
        random.Random(42).shuffle(expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_data_Qwen25VL_7B_scores.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            dataset = UniMEV2Dataset(str(path))
        self.assertEqual(
            [row["query_text"] for row in dataset.rows],
            [row["query_text"] for row in expected],
        )

    def test_dataset_accepts_official_empty_pair_records(self):
        rows = [_row(index) for index in range(3)]
        rows[1]["pos_text"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_data_Qwen25VL_7B_scores.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            dataset = UniMEV2Dataset(str(path))
        self.assertEqual(len(dataset), 3)


if __name__ == "__main__":
    unittest.main()
