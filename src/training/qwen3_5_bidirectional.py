"""Isolated bidirectional full-attention configuration for Qwen3.5.

Qwen3.5 is a hybrid decoder: standard full-attention layers are interleaved
with causal Gated DeltaNet layers.  This ablation makes every checkpoint-backed
full-attention layer bidirectional without replacing DeltaNet layers (which
would require nonexistent/randomly initialized checkpoint weights).
"""

from __future__ import annotations

import json
from pathlib import Path


CONFIG_NAME = "qwen3_5_bidirectional_mean_config.json"


def enable_qwen3_5_bidirectional_full_attention(base_model) -> dict:
    config = getattr(base_model, "config", None)
    if getattr(config, "model_type", None) != "qwen3_5":
        raise ValueError(
            "Bidirectional Qwen3.5 attention requires model_type='qwen3_5', "
            f"got {getattr(config, 'model_type', None)!r}"
        )
    language_model = getattr(base_model, "language_model", None)
    if language_model is None:
        raise RuntimeError("Qwen3.5 base model has no language_model")
    text_config = getattr(config, "text_config", None)
    runtime_config = getattr(language_model, "config", None)
    if text_config is None or runtime_config is None:
        raise RuntimeError("Qwen3.5 text configuration is unavailable")

    layer_types = list(getattr(text_config, "layer_types", ()) or ())
    layers = list(getattr(language_model, "layers", ()) or ())
    if not layer_types or len(layer_types) != len(layers):
        raise RuntimeError(
            "Qwen3.5 bidirectional setup requires one layer type per layer: "
            f"types={len(layer_types)} layers={len(layers)}"
        )

    # Transformers 5.9 create_causal_mask falls back to its bidirectional mask
    # factory when text_config.is_causal is false. FlashAttention additionally
    # consults each attention module's is_causal attribute when a dense mask can
    # be skipped, so both levels must be changed.
    text_config.is_causal = False
    runtime_config.is_causal = False
    full_attention_indices = []
    linear_attention_indices = []
    for index, (layer_type, layer) in enumerate(zip(layer_types, layers)):
        if layer_type == "full_attention":
            attention = getattr(layer, "self_attn", None)
            if attention is None or not hasattr(attention, "is_causal"):
                raise RuntimeError(
                    f"Qwen3.5 full-attention layer {index} has no causal flag"
                )
            attention.is_causal = False
            full_attention_indices.append(index)
        elif layer_type == "linear_attention":
            if getattr(layer, "linear_attn", None) is None:
                raise RuntimeError(
                    f"Qwen3.5 linear-attention layer {index} is malformed"
                )
            linear_attention_indices.append(index)
        else:
            raise RuntimeError(
                f"Unsupported Qwen3.5 layer type at {index}: {layer_type!r}"
            )

    if not full_attention_indices or not linear_attention_indices:
        raise RuntimeError(
            "Expected Qwen3.5 hybrid full-attention/Gated-DeltaNet layers"
        )
    if any(
        getattr(layers[index].self_attn, "is_causal", True)
        for index in full_attention_indices
    ):
        raise RuntimeError("A Qwen3.5 full-attention layer remained causal")

    return {
        "method": "qwen3_5_bidirectional_full_attention_mean_pooling",
        "full_attention_bidirectional": True,
        "full_attention_layer_indices": full_attention_indices,
        "gated_deltanet_causal": True,
        "gated_deltanet_layer_indices": linear_attention_indices,
        "pooling": "mean_all_active_tokens",
        "padding_excluded": True,
        "mean_accumulation_dtype": "float32",
    }


def validate_bidirectional_mean_training_profile(
    *,
    model_backbone: str,
    enabled: bool,
    pooling: str,
    lora: bool,
    moe_lora: bool,
    negative_strategy: str,
    embedding_gradient_strategy: str,
    pos_only: bool,
    grad_cache: bool,
) -> None:
    if not enabled:
        return
    errors = []
    if model_backbone != "qwen3_5_vl":
        errors.append(f"backbone must be qwen3_5_vl, got {model_backbone!r}")
    if pooling != "mean":
        errors.append(f"pooling must be mean, got {pooling!r}")
    if not lora:
        errors.append("LoRA/DoRA must be enabled")
    if moe_lora:
        errors.append("MoE must remain disabled")
    if negative_strategy != "vlm2vec_inbatch":
        errors.append("negative_strategy must be vlm2vec_inbatch")
    if embedding_gradient_strategy != "info_nce":
        errors.append("embedding_gradient_strategy must be info_nce")
    if not pos_only:
        errors.append("pos_only must be True")
    if not grad_cache:
        errors.append("GradCache must be enabled")
    if errors:
        raise ValueError(
            "Qwen3.5 bidirectional-attention + mean-pooling contract failed: "
            + "; ".join(errors)
        )


def write_checkpoint_config(output_dir: str, config: dict) -> None:
    path = Path(output_dir) / CONFIG_NAME
    path.write_text(
        json.dumps(config, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_checkpoint_config(checkpoint: str) -> dict:
    path = Path(checkpoint) / CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing bidirectional checkpoint config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
