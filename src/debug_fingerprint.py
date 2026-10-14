"""Opt-in, side-effect-free fingerprints for B3 reproduction debugging."""

import hashlib
import importlib.metadata
import json
import os
from collections import defaultdict

import torch


_ENABLED = os.environ.get("B3_FINGERPRINT", "0") == "1"
_MAX_CALLS = int(os.environ.get("B3_FINGERPRINT_CALLS", "20"))
_MAX_ELEMENTS = max(2, int(os.environ.get("B3_FINGERPRINT_ELEMENTS", "1048576")))
_COUNTS = defaultdict(int)


def fingerprint_enabled():
    return _ENABLED


def _rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _rank_enabled(rank):
    configured = os.environ.get("B3_FINGERPRINT_RANKS", "0").strip().lower()
    if configured == "all":
        return True
    return rank in {int(value) for value in configured.split(",") if value.strip()}


def _sample_tensor(tensor):
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= _MAX_ELEMENTS:
        return flat.contiguous()
    # Build indices with integer arithmetic. torch.linspace uses float32 on
    # CUDA by default, which can round the final index past very large tensors.
    indices = torch.arange(
        _MAX_ELEMENTS,
        device=flat.device,
        dtype=torch.long,
    )
    indices = indices * (flat.numel() - 1) // (_MAX_ELEMENTS - 1)
    return flat.index_select(0, indices).contiguous()


def tensor_fingerprint(tensor):
    sample = _sample_tensor(tensor).cpu()
    raw = sample.view(torch.uint8).numpy().tobytes()
    result = {
        "kind": "tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "sampled": sample.numel(),
        "sha256": hashlib.sha256(raw).hexdigest()[:20],
    }
    if sample.numel():
        numeric = sample.float()
        finite = torch.isfinite(numeric)
        result["finite"] = int(finite.sum().item())
        if finite.any():
            values = numeric[finite]
            result.update(
                min=float(values.min().item()),
                max=float(values.max().item()),
                mean=float(values.mean().item()),
                std=float(values.std(unbiased=False).item()),
                l2=float(torch.linalg.vector_norm(values).item()),
            )
    return result


def value_fingerprint(value):
    if isinstance(value, torch.Tensor):
        return tensor_fingerprint(value)
    if isinstance(value, dict):
        return {
            str(key): value_fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if isinstance(item, (torch.Tensor, dict, list, tuple, bool, int, float, str)) or item is None
        }
    if isinstance(value, (list, tuple)):
        length = len(value)
        if length <= 16:
            indices = list(range(length))
        else:
            indices = sorted({round(i * (length - 1) / 15) for i in range(16)})
        sampled = [value_fingerprint(value[index]) for index in indices]
        canonical = json.dumps(sampled, sort_keys=True, separators=(",", ":"))
        return {
            "kind": type(value).__name__,
            "length": length,
            "indices": indices,
            "sha256": hashlib.sha256(canonical.encode()).hexdigest()[:20],
            "items": sampled,
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"kind": type(value).__name__, "repr": repr(value)[:160]}


def log_fingerprint(stage, **values):
    if not _ENABLED:
        return
    rank = _rank()
    if not _rank_enabled(rank):
        return
    _COUNTS[stage] += 1
    call = _COUNTS[stage]
    if call > _MAX_CALLS:
        return
    payload = {
        "rank": rank,
        "stage": stage,
        "call": call,
        "values": {key: value_fingerprint(value) for key, value in values.items()},
    }
    print(
        "[B3_FINGERPRINT] "
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def log_runtime_and_model_state(model):
    if not _ENABLED or not _rank_enabled(_rank()):
        return
    versions = {}
    for package in ("torch", "transformers", "peft", "flash_attn", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    log_fingerprint("runtime", versions=versions)

    parameters = dict(model.named_parameters())
    selected = {}
    patterns = {
        "vision_patch": ("visual.patch_embed.proj.weight",),
        "token_embedding": (
            "model.embed_tokens.weight",
            "model.embed_tokens.token_adapter.base_layer.weight",
        ),
        "qwen2_trainable_embedding_token": (
            "model.embed_tokens.token_adapter.trainable_tokens_delta.default",
        ),
        "layer0_q_base": ("model.layers.0.self_attn.q_proj.base_layer.weight", "model.layers.0.self_attn.q_proj.weight"),
        "layer0_q_lora_a": ("model.layers.0.self_attn.q_proj.lora_A.default.weight",),
        "layer0_q_lora_b": ("model.layers.0.self_attn.q_proj.lora_B.default.weight",),
        "layer0_q_dora": ("model.layers.0.self_attn.q_proj.lora_magnitude_vector.default.weight",),
        "layer27_o_base": ("model.layers.27.self_attn.o_proj.base_layer.weight", "model.layers.27.self_attn.o_proj.weight"),
        "final_norm": ("model.norm.weight",),
        # Qwen3.5 keeps the decoder under ``language_model`` and alternates
        # Gated DeltaNet layers with full-attention layers.  Keep the Qwen2
        # labels above untouched and add architecture-specific sentinels so
        # the two B3 runs can be inspected with the same fingerprint format.
        "qwen3_5_token_embedding": ("language_model.embed_tokens.weight",),
        "qwen3_5_layer0_linear_qkv_base": (
            "language_model.layers.0.linear_attn.in_proj_qkv.base_layer.weight",
            "language_model.layers.0.linear_attn.in_proj_qkv.weight",
        ),
        "qwen3_5_layer0_linear_qkv_lora_a": (
            "language_model.layers.0.linear_attn.in_proj_qkv.lora_A.default.weight",
        ),
        "qwen3_5_layer0_linear_qkv_lora_b": (
            "language_model.layers.0.linear_attn.in_proj_qkv.lora_B.default.weight",
        ),
        "qwen3_5_layer0_linear_qkv_moe_lora_a": (
            "language_model.layers.0.linear_attn.in_proj_qkv.lora_A",
        ),
        "qwen3_5_layer0_linear_qkv_moe_lora_b": (
            "language_model.layers.0.linear_attn.in_proj_qkv.lora_B",
        ),
        "qwen3_5_layer0_linear_qkv_moe_router": (
            "language_model.layers.0.linear_attn.in_proj_qkv.lora_router",
        ),
        "qwen3_5_layer0_linear_qkv_moe_dora": (
            "language_model.layers.0.linear_attn.in_proj_qkv.lora_magnitude_vector",
        ),
        "qwen3_5_layer0_linear_qkv_dora": (
            "language_model.layers.0.linear_attn.in_proj_qkv.lora_magnitude_vector.default.weight",
        ),
        "qwen3_5_layer3_q_base": (
            "language_model.layers.3.self_attn.q_proj.base_layer.weight",
            "language_model.layers.3.self_attn.q_proj.weight",
        ),
        "qwen3_5_layer3_q_lora_a": (
            "language_model.layers.3.self_attn.q_proj.lora_A.default.weight",
        ),
        "qwen3_5_layer3_q_lora_b": (
            "language_model.layers.3.self_attn.q_proj.lora_B.default.weight",
        ),
        "qwen3_5_layer3_q_moe_lora_a": (
            "language_model.layers.3.self_attn.q_proj.lora_A",
        ),
        "qwen3_5_layer3_q_moe_lora_b": (
            "language_model.layers.3.self_attn.q_proj.lora_B",
        ),
        "qwen3_5_layer3_q_moe_router": (
            "language_model.layers.3.self_attn.q_proj.lora_router",
        ),
        "qwen3_5_layer3_q_moe_dora": (
            "language_model.layers.3.self_attn.q_proj.lora_magnitude_vector",
        ),
        "qwen3_5_layer3_q_dora": (
            "language_model.layers.3.self_attn.q_proj.lora_magnitude_vector.default.weight",
        ),
        "qwen3_5_layer23_down_base": (
            "language_model.layers.23.mlp.down_proj.base_layer.weight",
            "language_model.layers.23.mlp.down_proj.weight",
        ),
        "qwen3_5_final_norm": ("language_model.norm.weight",),
    }
    for label, suffixes in patterns.items():
        matches = [
            (name, parameter)
            for name, parameter in parameters.items()
            if any(name.endswith(suffix) for suffix in suffixes)
        ]
        if matches:
            name, parameter = matches[0]
            selected[label] = {"name": name, "value": parameter}

    inv_freq = [
        (name, buffer)
        for name, buffer in model.named_buffers()
        if name.endswith("inv_freq")
    ]
    rope = {"count": len(inv_freq)}
    for index, (name, buffer) in enumerate(inv_freq):
        if index in {0, 1, len(inv_freq) - 1}:
            rope[f"buffer_{index}"] = {"name": name, "value": buffer}
    log_fingerprint("model_state", parameters=selected, rope=rope)


def install_model_hooks(model):
    if not _ENABLED:
        return
    suffixes = {
        "vision_patch_output": "visual.patch_embed",
        "vision_merger_output": "visual.merger",
        "token_embedding_output": "model.embed_tokens",
        "decoder_layer0_output": "model.layers.0",
        "decoder_final_norm_output": "model.norm",
        "qwen3_5_token_embedding_output": "language_model.embed_tokens",
        "qwen3_5_decoder_layer0_output": "language_model.layers.0",
        "qwen3_5_decoder_layer3_output": "language_model.layers.3",
        "qwen3_5_decoder_final_norm_output": "language_model.norm",
    }
    modules = list(model.named_modules())
    for stage, suffix in suffixes.items():
        matches = [(name, module) for name, module in modules if name.endswith(suffix)]
        if not matches:
            continue
        name, module = matches[0]

        def hook(_module, inputs, output, stage=stage, name=name):
            log_fingerprint(stage, module=name, inputs=inputs, output=output)

        module.register_forward_hook(hook)
