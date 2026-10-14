"""Isolated dense MoE-DoRA implementation for the TSEmbed architecture ablation.

This module intentionally implements only Eq. (3)-(4) of TSEmbed: a frozen
linear projection plus a softmax-weighted mixture of low-rank experts.  Each
expert follows the same DoRA parameterization as the corresponding single-
adapter baseline. Qwen3.5 and Qwen2-VL use different decoder module paths, so
their target mappings are validated separately before any layer is replaced.
It does not implement EANS, routing-signature losses, or a two-stage schedule.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


MOE_LORA_CONFIG_NAME = "moe_lora_config.json"
MOE_LORA_WEIGHTS_NAME = "adapter_model.bin"
MOE_LORA_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
MOE_LORA_FORMAT_VERSION = 2
MOE_ADAPTER_MARKERS = (
    ".lora_A",
    ".lora_B",
    ".lora_router",
    ".lora_magnitude_vector",
)


def validate_moe_training_profile(
    *,
    model_backbone: str,
    lora_enabled: bool,
    negative_strategy: str,
    embedding_gradient_strategy: str,
    pos_only: bool,
    grad_cache: bool,
    sdibn: bool,
    odibn: bool,
    rdibn: bool,
    # Retain the previous call signature for OSS/FUSE mixed-version safety.
    # These are intentionally accepted but not validated: scripts may freely
    # override the MoE/LoRA hyperparameters.
    num_experts: int | None = None,
    rank: int | None = None,
    alpha: float | None = None,
    router_temperature: float | None = None,
) -> str:
    """Validate the two isolated training profiles supported by MoE-DoRA.

    The architecture can be paired with either the established ordinary
    in-batch control or the reproduced B3 batch construction.  Keeping the
    complete profile contract here prevents a partially mixed configuration
    from silently becoming a third algorithm.
    """

    errors = []
    supported_backbones = {"qwen2_vl", "qwen3_5_vl"}
    if model_backbone not in supported_backbones:
        errors.append(
            f"backbone must be one of {sorted(supported_backbones)}, "
            f"found {model_backbone!r}"
        )
    if lora_enabled:
        errors.append("lora and moe_lora must not both be enabled")
    if embedding_gradient_strategy != "info_nce":
        errors.append(
            "embedding_gradient_strategy must remain info_nce "
            "(EANS and EGA are disabled)"
        )
    if not grad_cache:
        errors.append("grad_cache must remain enabled")

    if negative_strategy == "vlm2vec_inbatch":
        profile = "vlm2vec_inbatch_moe"
        if not pos_only:
            errors.append("in-batch MoE requires pos_only=True")
        if sdibn or odibn or rdibn:
            errors.append(
                "in-batch MoE requires sdibn=False, odibn=False, rdibn=False"
            )
    elif negative_strategy == "b3":
        profile = "b3_moe"
        if pos_only:
            errors.append("B3+MoE requires pos_only=False to retain five HNs")
        if not sdibn or not odibn or rdibn:
            errors.append(
                "B3+MoE requires sdibn=True, odibn=True, rdibn=False"
            )
    else:
        profile = "unsupported"
        errors.append(
            "negative_strategy must be vlm2vec_inbatch or b3, found "
            f"{negative_strategy!r}"
        )

    if errors:
        raise ValueError(
            "TSEmbed MoE training contract failed: "
            + "; ".join(errors)
        )
    return profile


class DenseMoEDoraLinear(nn.Module):
    """Dense, per-token softmax mixture of DoRA experts.

    The frozen base projection is evaluated once.  Each expert has an
    independent A/B pair and magnitude vector. Every expert participates in
    the forward pass, weighted by ``softmax(W_g x / temperature)``. The DoRA
    direction norm uses FP32 accumulation and is detached exactly as in PEFT's
    DoRA implementation; this is required by the verified Qwen3.5 baseline.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        num_experts: int,
        router_temperature: float,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                "DenseMoEDoraLinear requires nn.Linear, got "
                f"{type(base_layer).__name__}"
            )
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if num_experts <= 1:
            raise ValueError(
                f"num_experts must be greater than one, got {num_experts}"
            )
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be finite and positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not math.isfinite(router_temperature) or router_temperature <= 0:
            raise ValueError(
                "router_temperature must be finite and positive, got "
                f"{router_temperature}"
            )

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout_p = float(dropout)
        self.num_experts = int(num_experts)
        self.router_temperature = float(router_temperature)
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)

        # Keep the small trainable tensors in FP32, but cast their compute view
        # to the activation dtype. This avoids materializing full-sequence FP32
        # expert outputs for Qwen3.5's wide projections while retaining FP32
        # optimizer states/parameters.
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.rank,
                self.in_features,
                device=device,
                dtype=torch.float32,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.out_features,
                self.rank,
                device=device,
                dtype=torch.float32,
            )
        )
        self.lora_router = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.in_features,
                device=device,
                dtype=torch.float32,
            )
        )
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / float(self.rank))
        nn.init.normal_(self.lora_router, mean=0.0, std=0.02)

        # PEFT initializes DoRA magnitude from the row-wise norm of the frozen
        # checkpoint weight because B starts at zero. Keep magnitude and norm
        # accumulation in FP32: BF16 norm previously produced zero rows for
        # Qwen3.5's DeltaNet QKV projection.
        with torch.no_grad():
            base_norm = torch.linalg.vector_norm(
                base_layer.weight.detach(),
                dim=1,
                dtype=torch.float32,
            ).clamp_min(torch.finfo(torch.float32).tiny)
        self.lora_magnitude_vector = nn.Parameter(
            base_norm.unsqueeze(0).repeat(self.num_experts, 1).contiguous()
        )

        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

        self._last_router_mean = None
        self._last_router_entropy = None

    @property
    def weight(self) -> Tensor:
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        base_output = self.base_layer(x, *args, **kwargs)
        compute_dtype = base_output.dtype
        adapter_input = x.to(dtype=compute_dtype)
        base_without_bias = base_output
        if self.base_layer.bias is not None:
            base_without_bias = base_without_bias - self.base_layer.bias.to(
                dtype=compute_dtype
            )
        router_logits = F.linear(
            adapter_input,
            self.lora_router.to(dtype=compute_dtype),
        )
        router_weights = torch.softmax(
            router_logits / self.router_temperature,
            dim=-1,
            dtype=torch.float32,
        )

        delta = torch.zeros_like(base_output)
        for expert_index in range(self.num_experts):
            # Match PEFT 0.19.1 DoraLinearVariant: while training with nonzero
            # dropout, each expert applies its own dropout mask and recomputes
            # the bias-free base projection from that same dropped input.
            # GradCache captures and replays this RNG stream deterministically.
            if self.training and self.dropout_p > 0.0:
                expert_input = F.dropout(
                    adapter_input,
                    p=self.dropout_p,
                    training=True,
                )
                expert_base_result = F.linear(
                    expert_input,
                    self.base_layer.weight.to(dtype=compute_dtype),
                    bias=None,
                )
            else:
                expert_input = adapter_input
                expert_base_result = base_without_bias
            low_rank = F.linear(
                expert_input,
                self.lora_A[expert_index].to(dtype=compute_dtype),
            )
            expert_output = F.linear(
                low_rank,
                self.lora_B[expert_index].to(dtype=compute_dtype),
            )

            # DoRA separates magnitude from direction. As in PEFT 0.19.1,
            # ||W + sBA|| is dynamically refreshed but detached from the
            # gradient graph; A/B receive gradients through expert_output.
            with torch.no_grad(), torch.autocast(
                device_type=adapter_input.device.type,
                enabled=False,
            ):
                lora_weight = torch.matmul(
                    self.lora_B[expert_index].float(),
                    self.lora_A[expert_index].float(),
                )
                direction_norm = torch.linalg.vector_norm(
                    self.base_layer.weight.detach().float()
                    + self.scaling * lora_weight,
                    dim=1,
                    dtype=torch.float32,
                ).clamp_min(torch.finfo(torch.float32).tiny)
            magnitude_scale = (
                self.lora_magnitude_vector[expert_index] / direction_norm
            ).to(dtype=compute_dtype)
            output_shape = (1,) * (base_output.ndim - 1) + (self.out_features,)
            magnitude_scale = magnitude_scale.view(output_shape)
            expert_delta = (
                (magnitude_scale - 1.0) * expert_base_result
                + magnitude_scale * expert_output * self.scaling
            )
            delta = delta + (
                router_weights[..., expert_index]
                .to(dtype=compute_dtype)
                .unsqueeze(-1)
                * expert_delta
            )

        # Detached summaries are diagnostics only and never enter EANS/loss.
        with torch.no_grad():
            reduce_dims = tuple(range(router_weights.ndim - 1))
            self._last_router_mean = router_weights.detach().mean(dim=reduce_dims)
            entropy = -(
                router_weights.detach()
                * router_weights.detach().clamp_min(1e-12).log()
            ).sum(dim=-1)
            self._last_router_entropy = entropy.mean()

        return base_output + delta


class DenseMoEDoraModel(nn.Module):
    """Backbone wrapper with adapter-only save/load semantics."""

    def __init__(self, base_model: nn.Module, adapter_config: Mapping) -> None:
        super().__init__()
        self.base_model = base_model
        self.moe_lora_config = dict(adapter_config)

    @property
    def config(self):
        return self.base_model.config

    def get_base_model(self) -> nn.Module:
        return self.base_model

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def adapter_state_dict(
        self, state_dict: Mapping[str, Tensor] | None = None
    ) -> Dict[str, Tensor]:
        source = self.state_dict() if state_dict is None else state_dict
        adapter = {
            key: value.detach().cpu().contiguous()
            for key, value in source.items()
            if any(marker in key for marker in MOE_ADAPTER_MARKERS)
        }
        expected = {
            key
            for key in self.state_dict()
            if any(marker in key for marker in MOE_ADAPTER_MARKERS)
        }
        if set(adapter) != expected:
            raise RuntimeError(
                "MoE-DoRA checkpoint state is incomplete: "
                f"missing={sorted(expected - set(adapter))[:5]}, "
                f"unexpected={sorted(set(adapter) - expected)[:5]}"
            )
        return adapter

    def save_pretrained(
        self,
        output_dir: str,
        state_dict: Mapping[str, Tensor] | None = None,
        safe_serialization: bool = False,
        **_kwargs,
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        adapter = self.adapter_state_dict(state_dict)
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(adapter, str(output / MOE_LORA_SAFE_WEIGHTS_NAME))
        else:
            torch.save(adapter, output / MOE_LORA_WEIGHTS_NAME)
        with (output / MOE_LORA_CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(self.moe_lora_config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self.config.to_json_file(str(output / "config.json"))


def _target_module_names(
    base_model: nn.Module, target_modules: Iterable[str]
) -> Sequence[str]:
    targets = {str(name).strip() for name in target_modules if str(name).strip()}
    model_type = getattr(getattr(base_model, "config", None), "model_type", None)
    if model_type == "qwen3_5":
        decoder_prefix = "language_model.layers."
    elif model_type == "qwen2_vl":
        decoder_prefix = "model.layers."
    elif hasattr(base_model, "language_model"):
        # Small synthetic models used by the adapter unit tests follow the
        # Qwen3.5-shaped module tree without carrying a Transformers config.
        decoder_prefix = "language_model.layers."
    else:
        raise ValueError(
            "Unsupported backbone for dense MoE-DoRA target discovery: "
            f"model_type={model_type!r}"
        )
    matches = []
    for name, module in base_model.named_modules():
        if not name.startswith(decoder_prefix):
            continue
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"MoE-DoRA target {name} is {type(module).__name__}, expected Linear"
            )
        matches.append(name)
    return tuple(matches)


def _replace_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, child_name = path.rsplit(".", 1)
    parent = root.get_submodule(parent_path)
    setattr(parent, child_name, module)


def _validate_qwen3_5_target_mapping(
    base_model: nn.Module, matched_names: Sequence[str]
) -> None:
    config = getattr(base_model, "config", None)
    if getattr(config, "model_type", None) != "qwen3_5":
        return
    text_config = getattr(config, "text_config", None)
    layer_types = list(getattr(text_config, "layer_types", ()) or ())
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or len(layer_types) != num_layers:
        raise RuntimeError(
            "Qwen3.5 MoE-DoRA requires one layer_type per decoder layer: "
            f"layers={num_layers}, layer_types={len(layer_types)}"
        )
    expected = set()
    for layer_index, layer_type in enumerate(layer_types):
        prefix = f"language_model.layers.{layer_index}"
        if layer_type == "full_attention":
            projections = (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            )
        elif layer_type == "linear_attention":
            projections = (
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.in_proj_b",
                "linear_attn.in_proj_a",
                "linear_attn.out_proj",
            )
        else:
            raise RuntimeError(
                "Unsupported Qwen3.5 layer type for MoE-DoRA: "
                f"layer={layer_index}, type={layer_type!r}"
            )
        expected.update(f"{prefix}.{name}" for name in (*projections, "mlp.down_proj"))
    actual = set(matched_names)
    if actual != expected:
        raise RuntimeError(
            "Qwen3.5 MoE-DoRA target mapping differs from the verified "
            "full-attention/Gated-DeltaNet/MLP-down scope: "
            f"missing={sorted(expected - actual)[:5]}, "
            f"unexpected={sorted(actual - expected)[:5]}, "
            f"expected_count={len(expected)}, actual_count={len(actual)}"
        )


def _validate_qwen2_vl_target_mapping(
    base_model: nn.Module, matched_names: Sequence[str]
) -> None:
    """Require exactly the successful Qwen2-VL LoRA/DoRA decoder scope."""

    config = getattr(base_model, "config", None)
    if getattr(config, "model_type", None) != "qwen2_vl":
        return
    num_layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise RuntimeError(
            "Qwen2-VL MoE-DoRA requires a positive num_hidden_layers, got "
            f"{num_layers!r}"
        )
    projections = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.down_proj",
    )
    expected = {
        f"model.layers.{layer_index}.{projection}"
        for layer_index in range(num_layers)
        for projection in projections
    }
    actual = set(matched_names)
    if actual != expected:
        raise RuntimeError(
            "Qwen2-VL MoE-DoRA target mapping differs from the reproduced "
            "VLM2Vec/B3 adapter scope: "
            f"missing={sorted(expected - actual)[:5]}, "
            f"unexpected={sorted(actual - expected)[:5]}, "
            f"expected_count={len(expected)}, actual_count={len(actual)}"
        )


def _validate_target_mapping(
    base_model: nn.Module, matched_names: Sequence[str]
) -> None:
    _validate_qwen3_5_target_mapping(base_model, matched_names)
    _validate_qwen2_vl_target_mapping(base_model, matched_names)


def build_dense_moe_lora(
    base_model: nn.Module,
    *,
    base_model_name_or_path: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Iterable[str],
    num_experts: int,
    router_temperature: float,
) -> DenseMoEDoraModel:
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    target_modules = tuple(
        str(name).strip() for name in target_modules if str(name).strip()
    )
    matched_names = _target_module_names(base_model, target_modules)
    if not matched_names:
        raise RuntimeError(
            f"No supported language-model projections matched {target_modules}"
        )
    _validate_target_mapping(base_model, matched_names)
    for name in matched_names:
        original = base_model.get_submodule(name)
        replacement = DenseMoEDoraLinear(
            original,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_experts=num_experts,
            router_temperature=router_temperature,
        )
        _replace_submodule(base_model, name, replacement)

    adapter_config = {
        "format_version": MOE_LORA_FORMAT_VERSION,
        "method": "tsembed_moe_dora_only",
        "architecture": "dense_softmax_per_token_moe_dora",
        "base_model_name_or_path": os.path.realpath(base_model_name_or_path),
        "rank_per_expert": int(rank),
        "alpha_per_expert": float(alpha),
        "dropout": float(dropout),
        "num_experts": int(num_experts),
        "router_temperature": float(router_temperature),
        "target_modules": list(target_modules),
        "matched_module_names": list(matched_names),
        "use_dora": True,
        "dora_norm_accumulation_dtype": "float32",
        "dora_direction_norm_gradient": "detached",
        "dora_dropout_semantics": (
            "peft_0.19.1_per_expert_input_and_base_recompute"
        ),
        "eans": False,
        "two_stage_training": False,
        "routing_scope": "per_token_per_projection",
        "expert_combination": "dense_softmax",
    }
    model = DenseMoEDoraModel(base_model, adapter_config)
    validate_dense_moe_lora(model, expect_fresh_initialization=True)
    return model


def load_dense_moe_lora_adapter(
    model: DenseMoEDoraModel,
    checkpoint_path: str,
    *,
    is_trainable: bool,
) -> None:
    checkpoint = Path(checkpoint_path)
    config_path = checkpoint / MOE_LORA_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing MoE-DoRA config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        saved_config = json.load(handle)
    if saved_config != model.moe_lora_config:
        differing = {
            key: (saved_config.get(key), model.moe_lora_config.get(key))
            for key in sorted(set(saved_config) | set(model.moe_lora_config))
            if saved_config.get(key) != model.moe_lora_config.get(key)
        }
        raise RuntimeError(
            "MoE-DoRA checkpoint configuration differs from the requested "
            f"architecture: {differing}"
        )

    bin_path = checkpoint / MOE_LORA_WEIGHTS_NAME
    safe_path = checkpoint / MOE_LORA_SAFE_WEIGHTS_NAME
    if safe_path.is_file():
        from safetensors.torch import load_file

        adapter_state = load_file(str(safe_path), device="cpu")
    elif bin_path.is_file():
        adapter_state = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"Missing {MOE_LORA_WEIGHTS_NAME} or {MOE_LORA_SAFE_WEIGHTS_NAME} "
            f"under {checkpoint}"
        )

    expected = {
        key
        for key in model.state_dict()
        if any(
            marker in key for marker in MOE_ADAPTER_MARKERS
        )
    }
    if set(adapter_state) != expected:
        raise RuntimeError(
            "MoE-DoRA checkpoint tensors do not match the instantiated model: "
            f"missing={sorted(expected - set(adapter_state))[:5]}, "
            f"unexpected={sorted(set(adapter_state) - expected)[:5]}"
        )
    incompatible = model.load_state_dict(adapter_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys while loading MoE-DoRA: "
            + ", ".join(incompatible.unexpected_keys[:5])
        )
    for name, parameter in model.named_parameters():
        is_adapter = any(
            marker in name for marker in MOE_ADAPTER_MARKERS
        )
        parameter.requires_grad_(bool(is_trainable and is_adapter))
    validate_dense_moe_lora(model, expect_fresh_initialization=False)


def validate_dense_moe_lora(
    model: DenseMoEDoraModel,
    *,
    expect_fresh_initialization: bool,
) -> None:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DenseMoEDoraLinear)
    ]
    configured = tuple(model.moe_lora_config["matched_module_names"])
    actual = tuple(name.removeprefix("base_model.") for name, _ in modules)
    if actual != configured:
        raise RuntimeError(
            "MoE-DoRA module mapping changed: "
            f"configured={configured[:5]}, actual={actual[:5]}"
        )
    if not modules:
        raise RuntimeError("MoE-DoRA model contains no expert modules")

    bad_finite = []
    bad_shapes = []
    bad_initial_a = []
    bad_initial_b = []
    bad_initial_router = []
    bad_initial_magnitude = []
    for name, module in modules:
        expected_shapes = (
            (module.num_experts, module.rank, module.in_features),
            (module.num_experts, module.out_features, module.rank),
            (module.num_experts, module.in_features),
            (module.num_experts, module.out_features),
        )
        actual_shapes = tuple(
            tuple(tensor.shape)
            for tensor in (
                module.lora_A,
                module.lora_B,
                module.lora_router,
                module.lora_magnitude_vector,
            )
        )
        if actual_shapes != expected_shapes:
            bad_shapes.append((name, actual_shapes, expected_shapes))
        if not all(
            torch.isfinite(tensor).all()
            for tensor in (
                module.lora_A,
                module.lora_B,
                module.lora_router,
                module.lora_magnitude_vector,
            )
        ):
            bad_finite.append(name)
        if expect_fresh_initialization:
            if not torch.count_nonzero(module.lora_A):
                bad_initial_a.append(name)
            if torch.count_nonzero(module.lora_B):
                bad_initial_b.append(name)
            if not torch.count_nonzero(module.lora_router):
                bad_initial_router.append(name)
            expected_magnitude = torch.linalg.vector_norm(
                module.base_layer.weight.detach(),
                dim=1,
                dtype=torch.float32,
            ).clamp_min(torch.finfo(torch.float32).tiny)
            actual_magnitude = module.lora_magnitude_vector.detach()
            expected_magnitude = expected_magnitude.unsqueeze(0).expand_as(
                actual_magnitude
            )
            if (
                (actual_magnitude <= 0).any()
                or not torch.equal(actual_magnitude, expected_magnitude)
            ):
                bad_initial_magnitude.append(name)
    trainable_non_adapter = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(
            marker in name for marker in MOE_ADAPTER_MARKERS
        )
    ]
    if (
        bad_finite
        or bad_shapes
        or bad_initial_a
        or bad_initial_b
        or bad_initial_router
        or bad_initial_magnitude
        or trainable_non_adapter
    ):
        raise RuntimeError(
            "MoE-DoRA validation failed: "
            f"nonfinite={bad_finite[:5]}, bad_shapes={bad_shapes[:3]}, "
            f"zero_fresh_A={bad_initial_a[:5]}, "
            f"nonzero_fresh_B={bad_initial_b[:5]}, "
            f"zero_fresh_router={bad_initial_router[:5]}, "
            f"invalid_fresh_magnitude={bad_initial_magnitude[:5]}, "
            f"trainable_backbone={trainable_non_adapter[:5]}"
        )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    first_name, first_module = modules[0]
    uniform_entropy = math.log(first_module.num_experts)
    print(
        "[TSEMBED_MOE_DORA_OK] "
        f"modules={len(modules)} experts={first_module.num_experts} "
        f"rank_per_expert={first_module.rank} alpha={first_module.alpha:g} "
        f"dropout={first_module.dropout_p:g} "
        f"router_temperature={first_module.router_temperature:g} "
        f"routing=dense_softmax_per_token use_dora=True "
        f"eans=False two_stage=False trainable_parameters={trainable} "
        f"uniform_router_entropy={uniform_entropy:.8g} first_module={first_name}",
        flush=True,
    )


def moe_router_diagnostics(model: nn.Module) -> Dict[str, object]:
    means = []
    entropies = []
    for module in model.modules():
        if not isinstance(module, DenseMoEDoraLinear):
            continue
        if module._last_router_mean is not None:
            means.append(module._last_router_mean.float().cpu())
        if module._last_router_entropy is not None:
            entropies.append(module._last_router_entropy.float().cpu())
    if not means:
        return {"observed_modules": 0}
    stacked_means = torch.stack(means)
    return {
        "observed_modules": len(means),
        "mean_expert_probability": stacked_means.mean(dim=0).tolist(),
        "mean_router_entropy": float(torch.stack(entropies).mean().item()),
        "normalized_router_entropy": float(
            torch.stack(entropies).mean().item() / math.log(stacked_means.shape[-1])
        ),
        "max_mean_expert_probability": float(
            stacked_means.mean(dim=0).max().item()
        ),
        "expert_probability_cv": float(
            stacked_means.mean(dim=0).std(unbiased=False).item()
            / stacked_means.mean(dim=0).mean().clamp_min(1e-12).item()
        ),
    }
