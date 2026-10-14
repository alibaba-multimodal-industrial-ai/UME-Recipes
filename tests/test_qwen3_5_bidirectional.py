from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

from src.training.qwen3_5_bidirectional import (
    CONFIG_NAME,
    enable_qwen3_5_bidirectional_full_attention,
    read_checkpoint_config,
    validate_bidirectional_mean_training_profile,
    write_checkpoint_config,
)


class _Attention:
    def __init__(self):
        self.is_causal = True


class _Layer:
    def __init__(self, layer_type):
        if layer_type == "full_attention":
            self.self_attn = _Attention()
        else:
            self.linear_attn = object()


def _model():
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    text_config = SimpleNamespace(layer_types=layer_types, is_causal=True)
    language_model = SimpleNamespace(
        config=text_config,
        layers=[_Layer(kind) for kind in layer_types],
    )
    return SimpleNamespace(
        config=SimpleNamespace(
            model_type="qwen3_5",
            text_config=text_config,
        ),
        language_model=language_model,
    )


class Qwen35BidirectionalAttentionTest(unittest.TestCase):
    def test_changes_only_checkpoint_backed_full_attention(self):
        model = _model()
        config = enable_qwen3_5_bidirectional_full_attention(model)
        self.assertFalse(model.config.text_config.is_causal)
        self.assertFalse(model.language_model.layers[3].self_attn.is_causal)
        self.assertEqual(config["full_attention_layer_indices"], [3])
        self.assertEqual(config["gated_deltanet_layer_indices"], [0, 1, 2])
        self.assertTrue(config["gated_deltanet_causal"])

    def test_training_contract_rejects_moe_or_last_pooling(self):
        common = dict(
            model_backbone="qwen3_5_vl",
            enabled=True,
            pooling="mean",
            lora=True,
            moe_lora=False,
            negative_strategy="vlm2vec_inbatch",
            embedding_gradient_strategy="info_nce",
            pos_only=True,
            grad_cache=True,
        )
        validate_bidirectional_mean_training_profile(**common)
        with self.assertRaisesRegex(ValueError, "pooling must be mean"):
            validate_bidirectional_mean_training_profile(
                **(common | {"pooling": "eos"})
            )
        with self.assertRaisesRegex(ValueError, "MoE must remain disabled"):
            validate_bidirectional_mean_training_profile(
                **(common | {"moe_lora": True})
            )

    def test_checkpoint_metadata_roundtrip(self):
        config = enable_qwen3_5_bidirectional_full_attention(_model())
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint_config(directory, config)
            self.assertTrue((Path(directory) / CONFIG_NAME).is_file())
            self.assertEqual(read_checkpoint_config(directory), config)


if __name__ == "__main__":
    unittest.main()
