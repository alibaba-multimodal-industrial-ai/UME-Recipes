from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peft import LoraConfig
from src.training.data.collator import _append_qwen2_embedding_token
from src.training.model import (
    MMEBModel,
    _register_qwen2_embedding_token,
    qwen2_embedding_token_checkpoint_metadata,
    validate_qwen2_embedding_token_checkpoint,
)


class _DummyTokenizer:
    def __init__(self):
        self._vocab = {"old": 0, "<image>": 1}
        self.additional_special_tokens = ["<image>"]

    @property
    def all_special_tokens(self):
        return list(self.additional_special_tokens)

    def get_vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, mapping):
        added = 0
        for token in mapping["additional_special_tokens"]:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
                added += 1
        # Match the current Transformers API's replacement behavior. The
        # production helper must therefore pass the complete preserved list.
        self.additional_special_tokens = list(
            mapping["additional_special_tokens"]
        )
        return added

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]

    def encode(self, token, add_special_tokens=False):
        assert add_special_tokens is False
        return [self._vocab[token]]

    def __len__(self):
        return len(self._vocab)


class _DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()


def _model():
    model = MMEBModel(encoder=_DummyEncoder(), pooling="eos", normalize=False)
    model.qwen2_embedding_token = True
    model.qwen2_embedding_token_text = "<embedding>"
    model.qwen2_embedding_token_id = 9
    return model


class Qwen2EmbeddingTokenTest(unittest.TestCase):
    def test_registration_adds_one_token_without_replacing_existing_specials(self):
        tokenizer = _DummyTokenizer()

        first = _register_qwen2_embedding_token(tokenizer, "<embedding>")
        second = _register_qwen2_embedding_token(tokenizer, "<embedding>")

        self.assertEqual(first["id"], 2)
        self.assertEqual(first["added"], 1)
        self.assertEqual(second["id"], 2)
        self.assertEqual(second["added"], 0)
        self.assertEqual(
            tokenizer.additional_special_tokens, ["<image>", "<embedding>"]
        )

    def test_collator_append_is_isolated_and_exactly_once(self):
        disabled = SimpleNamespace(_qwen2_embedding_token_enabled=False)
        enabled = SimpleNamespace(
            _qwen2_embedding_token_enabled=True,
            _qwen2_embedding_token_id=9,
        )

        self.assertEqual(_append_qwen2_embedding_token([1, 2], disabled), [1, 2])
        self.assertEqual(_append_qwen2_embedding_token([1, 2], enabled), [1, 2, 9])
        with self.assertRaisesRegex(ValueError, "exactly one copy"):
            _append_qwen2_embedding_token([1, 9], enabled)

    def test_batch_contract_accepts_left_padding_and_terminal_token(self):
        model = _model()
        input_ids = torch.tensor([[0, 4, 9], [5, 6, 9]])
        attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])

        model._validate_qwen2_embedding_token_inputs(input_ids, attention_mask)

    def test_batch_contract_rejects_nonterminal_or_duplicate_token(self):
        model = _model()
        with self.assertRaisesRegex(ValueError, "last active token"):
            model._validate_qwen2_embedding_token_inputs(
                torch.tensor([[9, 4]]), torch.ones((1, 2), dtype=torch.long)
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model._validate_qwen2_embedding_token_inputs(
                torch.tensor([[9, 9]]), torch.ones((1, 2), dtype=torch.long)
            )

    def test_existing_last_pooling_returns_embedding_token_hidden_state(self):
        model = _model()
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]]])
        mask = torch.ones((1, 3), dtype=torch.long)

        actual = model._pooling(hidden, mask)

        torch.testing.assert_close(actual, torch.tensor([[8.0, 9.0]]))

    def test_peft_config_targets_one_tied_token_row(self):
        config = LoraConfig(
            r=8,
            target_modules=["q_proj"],
            trainable_token_indices=[151657],
            ensure_weight_tying=True,
        )
        self.assertEqual(config.trainable_token_indices, [151657])
        self.assertTrue(config.ensure_weight_tying)

    def test_checkpoint_metadata_round_trip_is_strict(self):
        args = SimpleNamespace(
            qwen2_embedding_token_text="<embedding>", pooling="eos"
        )
        metadata = qwen2_embedding_token_checkpoint_metadata(args, 151657)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen2_embedding_token_config.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")

            validate_qwen2_embedding_token_checkpoint(
                directory, args, token_id=151657
            )
            with self.assertRaisesRegex(RuntimeError, "contract differs"):
                validate_qwen2_embedding_token_checkpoint(
                    directory, args, token_id=151658
                )


if __name__ == "__main__":
    unittest.main()
