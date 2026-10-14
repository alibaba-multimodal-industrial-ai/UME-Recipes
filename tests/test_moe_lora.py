import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.training.moe_lora import (
    DenseMoEDoraLinear,
    MOE_LORA_CONFIG_NAME,
    build_dense_moe_lora,
    load_dense_moe_lora_adapter,
    moe_router_diagnostics,
    validate_moe_training_profile,
)


class _Config:
    def to_json_file(self, path):
        Path(path).write_text(json.dumps({"model_type": "dummy"}))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(6, 5, bias=False)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.config = _Config()

    def forward(self, x):
        return self.language_model.layers[0].down_proj(x)


class _Qwen2Config:
    model_type = "qwen2_vl"
    num_hidden_layers = 2

    def to_json_file(self, path):
        Path(path).write_text(json.dumps({"model_type": self.model_type}))


class _Qwen2Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(6, 6, bias=False)
        self.k_proj = nn.Linear(6, 6, bias=False)
        self.v_proj = nn.Linear(6, 6, bias=False)
        self.o_proj = nn.Linear(6, 6, bias=False)


class _Qwen2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(6, 6, bias=False)


class _Qwen2Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Qwen2Attention()
        self.mlp = _Qwen2MLP()


class _Qwen2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_Qwen2Layer() for _ in range(_Qwen2Config.num_hidden_layers)]
        )
        # A visual projection with a matching suffix must never be adapted.
        self.visual = nn.Module()
        self.visual.down_proj = nn.Linear(6, 6, bias=False)
        self.config = _Qwen2Config()


def _build(path):
    torch.manual_seed(13)
    return build_dense_moe_lora(
        _Backbone(),
        base_model_name_or_path=path,
        rank=2,
        alpha=4,
        dropout=0.0,
        target_modules=["down_proj"],
        num_experts=4,
        router_temperature=1.0,
    )


class DenseMoELoraTest(unittest.TestCase):
    def test_training_profiles_are_explicit_and_isolated(self):
        common = {
            "model_backbone": "qwen3_5_vl",
            "lora_enabled": False,
            "embedding_gradient_strategy": "info_nce",
            "grad_cache": True,
        }
        self.assertEqual(
            validate_moe_training_profile(
                **common,
                negative_strategy="vlm2vec_inbatch",
                pos_only=True,
                sdibn=False,
                odibn=False,
                rdibn=False,
                # Backward-compatible arguments are accepted but deliberately
                # unconstrained by the structural training contract.
                num_experts=3,
                rank=8,
                alpha=32,
                router_temperature=0.5,
            ),
            "vlm2vec_inbatch_moe",
        )
        self.assertEqual(
            validate_moe_training_profile(
                **common,
                negative_strategy="b3",
                pos_only=False,
                sdibn=True,
                odibn=True,
                rdibn=False,
            ),
            "b3_moe",
        )
        with self.assertRaisesRegex(ValueError, "retain five HNs"):
            validate_moe_training_profile(
                **common,
                negative_strategy="b3",
                pos_only=True,
                sdibn=True,
                odibn=True,
                rdibn=False,
            )
        qwen2 = dict(common)
        qwen2["model_backbone"] = "qwen2_vl"
        self.assertEqual(
            validate_moe_training_profile(
                **qwen2,
                negative_strategy="vlm2vec_inbatch",
                pos_only=True,
                sdibn=False,
                odibn=False,
                rdibn=False,
            ),
            "vlm2vec_inbatch_moe",
        )

    def test_qwen2_mapping_covers_decoder_only(self):
        model = build_dense_moe_lora(
            _Qwen2Backbone(),
            base_model_name_or_path="/tmp/qwen2-dummy",
            rank=2,
            alpha=4,
            dropout=0.0,
            target_modules=(
                "qkv_proj",
                "o_proj",
                "gate_up_proj",
                "down_proj",
                "k_proj",
                "q_proj",
                "out_proj",
                "v_proj",
            ),
            num_experts=4,
            router_temperature=1.0,
        )
        modules = [
            (name, module)
            for name, module in model.base_model.named_modules()
            if isinstance(module, DenseMoEDoraLinear)
        ]
        self.assertEqual(len(modules), 10)
        self.assertTrue(all(name.startswith("model.layers.") for name, _ in modules))
        self.assertIsInstance(model.base_model.visual.down_proj, nn.Linear)
        self.assertNotIsInstance(
            model.base_model.visual.down_proj, DenseMoEDoraLinear
        )

    def test_fresh_adapter_preserves_checkpoint_forward(self):
        torch.manual_seed(7)
        backbone = _Backbone()
        x = torch.randn(3, 4, 6)
        expected = backbone(x).detach().clone()
        model = build_dense_moe_lora(
            backbone,
            base_model_name_or_path="/tmp/dummy",
            rank=2,
            alpha=4,
            dropout=0.0,
            target_modules=["down_proj"],
            num_experts=4,
            router_temperature=1.0,
        )
        actual = model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        module = model.base_model.language_model.layers[0].down_proj
        self.assertIsInstance(module, DenseMoEDoraLinear)
        self.assertEqual(torch.count_nonzero(module.lora_B).item(), 0)
        self.assertTrue(torch.count_nonzero(module.lora_A).item() > 0)
        self.assertTrue(torch.count_nonzero(module.lora_router).item() > 0)
        expected_magnitude = torch.linalg.vector_norm(
            module.base_layer.weight,
            dim=1,
            dtype=torch.float32,
        ).unsqueeze(0).expand_as(module.lora_magnitude_vector)
        torch.testing.assert_close(
            module.lora_magnitude_vector,
            expected_magnitude,
            rtol=0,
            atol=0,
        )
        self.assertTrue(all(not p.requires_grad for p in module.base_layer.parameters()))

    def test_dense_router_and_all_experts_receive_gradients(self):
        model = _build("/tmp/dummy")
        module = model.base_model.language_model.layers[0].down_proj
        with torch.no_grad():
            module.lora_B.normal_(mean=0.0, std=0.1)
        loss = model(torch.randn(3, 4, 6)).square().mean()
        loss.backward()
        self.assertIsNotNone(module.lora_router.grad)
        self.assertTrue(torch.count_nonzero(module.lora_router.grad).item() > 0)
        self.assertIsNotNone(module.lora_magnitude_vector.grad)
        self.assertTrue(
            torch.count_nonzero(module.lora_magnitude_vector.grad).item() > 0
        )
        for expert in range(module.num_experts):
            self.assertTrue(torch.count_nonzero(module.lora_A.grad[expert]).item() > 0)
            self.assertTrue(torch.count_nonzero(module.lora_B.grad[expert]).item() > 0)
        probs = module._last_router_mean
        self.assertIsNotNone(probs)
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
        diagnostics = moe_router_diagnostics(model)
        self.assertEqual(diagnostics["observed_modules"], 1)
        self.assertGreaterEqual(diagnostics["normalized_router_entropy"], 0.0)
        self.assertLessEqual(diagnostics["normalized_router_entropy"], 1.0 + 1e-6)

    def test_adapter_only_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _build(directory)
            module = model.base_model.language_model.layers[0].down_proj
            with torch.no_grad():
                module.lora_B.normal_(mean=0.0, std=0.1)
                module.lora_router.add_(0.25)
                module.lora_magnitude_vector.mul_(1.05)
            probe = torch.randn(2, 3, 6)
            model.eval()
            expected_output = model(probe).detach().clone()
            expected = {
                key: value.detach().clone()
                for key, value in model.adapter_state_dict().items()
            }
            model.save_pretrained(directory, safe_serialization=False)
            self.assertTrue((Path(directory) / MOE_LORA_CONFIG_NAME).is_file())

            restored = _build(directory)
            load_dense_moe_lora_adapter(
                restored, directory, is_trainable=True
            )
            actual = restored.adapter_state_dict()
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])
            restored.eval()
            torch.testing.assert_close(
                restored(probe),
                expected_output,
                rtol=0,
                atol=0,
            )
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for name, parameter in restored.named_parameters()
                    if any(
                        token in name
                        for token in (
                            "lora_A",
                            "lora_B",
                            "lora_router",
                            "lora_magnitude_vector",
                        )
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
