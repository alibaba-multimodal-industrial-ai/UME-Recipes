from typing import Dict, Optional
import json
import os
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.training.arguments import ModelArguments
from src.training.utils import print_master
from src.debug_fingerprint import log_fingerprint

import logging
logger = logging.getLogger(__name__)

# ==================== Backbone Constants ====================
PHI3V = 'phi3_v'
QWEN2_VL = 'qwen2_vl'
QWEN3_5_VL = 'qwen3_5_vl'

MODEL2BACKBONE = {
    'phi3_v': PHI3V,
    'qwen2_vl': QWEN2_VL,
    'qwen3_5': QWEN3_5_VL,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
    QWEN2_VL: "<|image_pad|>",
    QWEN3_5_VL: "<|image_pad|>",
}

VLM_VIDEO_TOKENS = {
    QWEN2_VL: "<|video_pad|>",
    QWEN3_5_VL: "<|video_pad|>",
}


def _register_qwen2_embedding_token(tokenizer, token_text: str):
    """Register one indivisible special token without replacing Qwen's list."""
    token_text = str(token_text)
    if not token_text or token_text.strip() != token_text:
        raise ValueError(
            "--qwen2_embedding_token_text must be non-empty and have no "
            "leading/trailing whitespace"
        )

    vocabulary_before = tokenizer.get_vocab()
    existed_before = token_text in vocabulary_before
    if existed_before and token_text not in tokenizer.all_special_tokens:
        raise ValueError(
            f"Qwen2 embedding token {token_text!r} already exists as an "
            "ordinary vocabulary token; refusing to change its semantics"
        )

    # Transformers 5.9's PreTrainedTokenizerBase no longer accepts the
    # replace_additional_special_tokens keyword here.  Preserve the existing
    # list explicitly and pass the complete list through the common API.  This
    # has the same append-without-replacement semantics on both old and new
    # Transformers versions.
    existing_additional_tokens = list(
        getattr(tokenizer, "additional_special_tokens", None) or []
    )
    existing_additional_texts = [
        str(token) for token in existing_additional_tokens
    ]
    complete_additional_tokens = list(existing_additional_tokens)
    if token_text not in existing_additional_texts:
        complete_additional_tokens.append(token_text)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": complete_additional_tokens}
    )
    token_id = tokenizer.convert_tokens_to_ids(token_text)
    encoded = tokenizer.encode(token_text, add_special_tokens=False)
    if encoded != [token_id]:
        raise RuntimeError(
            "Qwen2 embedding token must encode to exactly one ID: "
            f"token={token_text!r}, token_id={token_id}, encoded={encoded}"
        )
    if token_text not in tokenizer.all_special_tokens:
        raise RuntimeError(
            f"Qwen2 embedding token {token_text!r} was not registered as special"
        )
    missing_existing_specials = [
        token
        for token in existing_additional_texts
        if token not in tokenizer.all_special_tokens
    ]
    if missing_existing_specials:
        raise RuntimeError(
            "Registering the Qwen2 embedding token removed existing special "
            f"tokens: {missing_existing_specials}"
        )
    expected_added = 0 if existed_before else 1
    if added != expected_added:
        raise RuntimeError(
            "Unexpected Qwen2 tokenizer growth while registering the embedding "
            f"token: existed={existed_before}, added={added}"
        )
    return {
        "text": token_text,
        "id": int(token_id),
        "tokenizer_size": int(len(tokenizer)),
        "added": int(added),
    }


def _load_qwen2_embedding_token_spec(model_name_or_path: str, token_text: str):
    """Resolve the token exactly as the vendored B3 Qwen2 tokenizer does."""
    from tokenizers import Tokenizer
    from src.training.vlm_backbone.qwen2_vl import Qwen2TokenizerFast

    tokenizer_file = os.path.join(model_name_or_path, "tokenizer.json")
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(
            f"Qwen2 embedding-token tokenizer.json not found: {tokenizer_file}"
        )
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        model_name_or_path,
        tokenizer_object=Tokenizer.from_file(tokenizer_file),
    )
    tokenizer.padding_side = "left"
    return _register_qwen2_embedding_token(tokenizer, token_text)


def _prepare_qwen2_embedding_token_base_model(
    base_model,
    model_args: ModelArguments,
    tokenizer_source: str,
):
    """Initialize one preallocated Qwen2 row and return its tokenizer spec."""
    if not getattr(model_args, "qwen2_embedding_token", False):
        return None
    if model_args.qwen2_model_impl != "b3":
        raise ValueError(
            "The isolated Qwen2 embedding-token experiment requires "
            "--qwen2_model_impl b3"
        )
    if model_args.pooling not in {"last", "eos"}:
        raise ValueError(
            "The Qwen2 embedding-token experiment uses existing last-token "
            f"pooling; got --pooling {model_args.pooling!r}"
        )
    if not model_args.lora:
        raise ValueError(
            "The Qwen2 embedding-token experiment requires --lora so PEFT can "
            "train and persist only the new token row"
        )

    spec = _load_qwen2_embedding_token_spec(
        tokenizer_source,
        model_args.qwen2_embedding_token_text,
    )
    input_embeddings = base_model.get_input_embeddings()
    output_embeddings = base_model.get_output_embeddings()
    num_embeddings = int(input_embeddings.weight.shape[0])
    if spec["tokenizer_size"] > num_embeddings or spec["id"] >= num_embeddings:
        raise RuntimeError(
            "The verified Qwen2 checkpoint is expected to have preallocated "
            "vocabulary rows for one new token; refusing to resize the backbone "
            f"embedding matrix: tokenizer_size={spec['tokenizer_size']}, "
            f"token_id={spec['id']}, model_rows={num_embeddings}"
        )

    if getattr(base_model.config, "tie_word_embeddings", False):
        if output_embeddings is None:
            raise RuntimeError("Qwen2 declares tied embeddings but has no LM head")
        if input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr():
            raise RuntimeError(
                "Qwen2 input embeddings and LM head must be tied before adding "
                "the trainable embedding token"
            )

    eos_token_id = int(base_model.config.eos_token_id)
    if not (0 <= eos_token_id < num_embeddings):
        raise ValueError(f"Invalid Qwen2 eos_token_id={eos_token_id}")
    with torch.no_grad():
        # This row is preallocated but unreachable from the stock tokenizer.
        # Copying EOS avoids making the ablation depend on an arbitrary unused
        # checkpoint row; PEFT then trains and saves only this one row.
        input_embeddings.weight[spec["id"]].copy_(
            input_embeddings.weight[eos_token_id]
        )

    spec.update(
        {
            "model_rows": num_embeddings,
            "initialization": "copy_eos_embedding",
            "initialization_token_id": eos_token_id,
        }
    )
    model_args.qwen2_embedding_token_id = spec["id"]
    print_master(
        "[QWEN2_EMBED_TOKEN_BASE_OK] "
        f"token={spec['text']!r} id={spec['id']} "
        f"tokenizer_size={spec['tokenizer_size']} model_rows={num_embeddings} "
        f"free_rows={num_embeddings - spec['tokenizer_size']} "
        f"initialization={spec['initialization']} source_id={eos_token_id}"
    )
    return spec


def _validate_qwen2_embedding_token_adapter(
    peft_model,
    token_spec,
    expect_trainable: bool,
):
    """Fail if PEFT did not isolate and restore exactly one token row."""
    if token_spec is None:
        return
    try:
        import peft
        from packaging.version import Version

        if Version(peft.__version__) < Version("0.19.1"):
            raise RuntimeError(
                "Qwen2 trainable embedding tokens require peft>=0.19.1, got "
                f"{peft.__version__}"
            )
    except ImportError as error:
        raise RuntimeError("Cannot validate the PEFT trainable-token runtime") from error

    adapter_name = peft_model.active_adapter
    if isinstance(adapter_name, (list, tuple)):
        if len(adapter_name) != 1:
            raise RuntimeError(
                "Qwen2 embedding-token validation requires exactly one active "
                f"adapter, got {adapter_name}"
            )
        adapter_name = adapter_name[0]
    config = peft_model.peft_config[adapter_name]
    indices = getattr(config, "trainable_token_indices", None)
    if isinstance(indices, dict):
        flattened = sorted({index for values in indices.values() for index in values})
    else:
        flattened = list(indices or [])
    if flattened != [token_spec["id"]]:
        raise RuntimeError(
            "PEFT checkpoint/config does not target exactly the Qwen2 embedding "
            f"token row: expected={[token_spec['id']]}, actual={indices}"
        )
    if not getattr(config, "ensure_weight_tying", False):
        raise RuntimeError(
            "Qwen2 embedding-token PEFT config must set ensure_weight_tying=True"
        )

    token_parameters = [
        (name, parameter)
        for name, parameter in peft_model.named_parameters()
        if "trainable_tokens_delta" in name
    ]
    if not token_parameters:
        raise RuntimeError("PEFT created no trainable-token delta parameter")
    unique_parameters = {}
    for name, parameter in token_parameters:
        unique_parameters.setdefault(parameter.data_ptr(), (name, parameter))
    if len(unique_parameters) != 1:
        raise RuntimeError(
            "Tied Qwen2 input/output embeddings must share one trainable-token "
            f"parameter, found {len(unique_parameters)}"
        )
    name, parameter = next(iter(unique_parameters.values()))
    hidden_size = int(peft_model.get_input_embeddings().weight.shape[1])
    expected_shape = (1, hidden_size)
    if tuple(parameter.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected trainable-token shape for {name}: "
            f"{tuple(parameter.shape)} != {expected_shape}"
        )
    if bool(parameter.requires_grad) != bool(expect_trainable):
        raise RuntimeError(
            f"Trainable-token requires_grad mismatch for {name}: "
            f"{parameter.requires_grad} != {expect_trainable}"
        )
    if not torch.isfinite(parameter.detach()).all():
        raise FloatingPointError(f"Non-finite Qwen2 trainable token: {name}")
    if expect_trainable and not getattr(
        peft_model, "_qwen2_embedding_token_gradient_hook", False
    ):
        gradient_reported = {"value": False}

        def report_first_gradient(gradient):
            if gradient_reported["value"]:
                return gradient
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    "Non-finite gradient for the Qwen2 embedding token"
                )
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                "[QWEN2_EMBED_TOKEN_GRAD_OK] "
                f"rank={rank} token_id={token_spec['id']} "
                f"shape={tuple(gradient.shape)} "
                f"norm={float(gradient.float().norm().item()):.8g} "
                f"nonzero={int(torch.count_nonzero(gradient).item())}/"
                f"{gradient.numel()}",
                flush=True,
            )
            gradient_reported["value"] = True
            return gradient

        parameter.register_hook(report_first_gradient)
        peft_model._qwen2_embedding_token_gradient_hook = True
    print_master(
        "[QWEN2_EMBED_TOKEN_ADAPTER_OK] "
        f"token={token_spec['text']!r} id={token_spec['id']} "
        f"shape={tuple(parameter.shape)} trainable={parameter.requires_grad} "
        "saved_as_single_peft_row=True tied_input_output=True"
    )


def qwen2_embedding_token_checkpoint_metadata(model_args, token_id: int):
    return {
        "format": "qwen2_trainable_embedding_token_v1",
        "token": model_args.qwen2_embedding_token_text,
        "token_id": int(token_id),
        "pooling_argument": model_args.pooling,
        "representation": "last_active_hidden_state",
        "append_to": ["query", "target"],
        "trainable_token_indices": [int(token_id)],
        "ensure_weight_tying": True,
        "initialization": "copy_eos_embedding",
    }


def validate_qwen2_embedding_token_checkpoint(
    checkpoint_path: str,
    model_args,
    token_id: int,
):
    metadata_path = os.path.join(
        checkpoint_path, "qwen2_embedding_token_config.json"
    )
    if not os.path.isfile(metadata_path):
        raise RuntimeError(
            "Refusing to load a Qwen2 embedding-token checkpoint without "
            f"{metadata_path}"
        )
    with open(metadata_path, "r", encoding="utf-8") as handle:
        actual = json.load(handle)
    expected = qwen2_embedding_token_checkpoint_metadata(model_args, token_id)
    if actual != expected:
        raise RuntimeError(
            "Qwen2 embedding-token checkpoint contract differs from the "
            f"current run: checkpoint={actual}, current={expected}"
        )
    print_master(
        "[QWEN2_EMBED_TOKEN_CHECKPOINT_OK] "
        f"checkpoint={checkpoint_path} values={actual}"
    )


def _enable_qwen3_5_stable_dora_norm() -> None:
    """Compute PEFT 0.19.1 DoRA norms in FP32 for Qwen3.5.

    The released Qwen3.5-2B checkpoint has no exactly-zero rows in its first
    DeltaNet QKV projection, but PEFT's BF16 ``torch.linalg.norm`` rounds 14
    row norms to zero. DoRA subsequently divides by those norms. Patch only
    processes that build Qwen3.5 so initialization and every later forward use
    the same FP32 accumulation, while preserving the model's original dtype.
    """
    from peft.tuners.lora.dora import DoraLinearLayer
    from peft.utils.other import transpose

    if getattr(DoraLinearLayer, "_qwen3_5_fp32_norm", False):
        return

    def get_weight_norm_fp32(
        self,
        weight,
        lora_weight,
        scaling,
        adapter_name=None,
    ):
        del adapter_name
        output_dtype = weight.dtype
        combined = transpose(weight, self.fan_in_fan_out)
        combined = combined + float(scaling) * lora_weight
        accumulation_dtype = (
            torch.float32
            if output_dtype in (torch.float16, torch.bfloat16)
            else output_dtype
        )
        weight_norm = torch.linalg.vector_norm(
            combined,
            dim=1,
            dtype=accumulation_dtype,
        )
        if output_dtype.is_floating_point:
            weight_norm = weight_norm.clamp_min(torch.finfo(output_dtype).tiny)
        return weight_norm.to(output_dtype)

    DoraLinearLayer.get_weight_norm = get_weight_norm_fp32
    DoraLinearLayer._qwen3_5_fp32_norm = True
    print_master(
        "[QWEN3_5_DORA_NORM_OK] PEFT DoRA row norms use FP32 accumulation "
        "with a dtype-safe positive floor"
    )


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, \
        f"Unknown backbone name {hf_config.model_type}. Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def _reset_and_validate_qwen2_b3_rope_buffers(base_model: nn.Module) -> None:
    """Restore non-persistent RoPE buffers after Transformers low-memory load.

    The vendored B3 Qwen2-VL implementation registers its text and vision
    ``inv_freq`` tensors with ``persistent=False``. They are therefore absent
    from the checkpoint. Transformers 5.x can materialize those buffers from a
    meta-model without running their normal initialization, leaving different
    uninitialized values on different ranks. Rebuild them deterministically
    before PEFT wrapping or the first forward.
    """
    try:
        model_device = next(base_model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    if model_device.type == "meta":
        model_device = torch.device("cpu")

    reset_buffers = []
    for module_name, module in base_model.named_modules():
        reset_inv_freq = getattr(module, "reset_inv_freq", None)
        if not callable(reset_inv_freq):
            continue
        reset_inv_freq(device=model_device)
        inv_freq = module.inv_freq
        if inv_freq.device.type == "meta":
            raise RuntimeError(
                f"B3 RoPE buffer {module_name}.inv_freq remained on meta"
            )
        if not torch.isfinite(inv_freq).all():
            raise FloatingPointError(
                f"B3 RoPE buffer {module_name}.inv_freq is non-finite "
                "immediately after deterministic reset"
            )
        if inv_freq.numel() == 0 or inv_freq.min().item() <= 0:
            raise ValueError(
                f"B3 RoPE buffer {module_name}.inv_freq has invalid values"
            )
        if inv_freq.max().item() > 1.000001:
            raise ValueError(
                f"B3 RoPE buffer {module_name}.inv_freq has unexpected "
                f"maximum {inv_freq.max().item()}"
            )
        reset_buffers.append(module_name)

    if not reset_buffers:
        raise RuntimeError(
            "No B3 Qwen2-VL RoPE buffers were found after model loading"
        )
    print_master(
        "[B3_ROPE_RESET] deterministically rebuilt and validated "
        f"{len(reset_buffers)} non-persistent inv_freq buffers on "
        f"{model_device}"
    )


def _strict_reload_qwen2_b3_safetensors(
    base_model: nn.Module,
    model_name_or_path: str,
) -> None:
    """Reload B3's legacy model after the Transformers 5.x construction pass.

    Transformers 5.9 can successfully enumerate the legacy Qwen2-VL
    checkpoint and then re-run ``_init_weights`` on Linear/Conv/Embedding
    modules from this Transformers-4.48-era model.  The load progress therefore
    reaches 100% while most pretrained matrices silently become fresh N(0,
    initializer_range) tensors.  Stream the safetensors shards once more after
    ``from_pretrained`` and fail closed on any key or shape mismatch.
    """
    from safetensors import safe_open

    if not os.path.isdir(model_name_or_path):
        raise ValueError(
            "B3 strict checkpoint reload currently requires a local model "
            f"directory, got {model_name_or_path!r}"
        )

    index_path = os.path.join(
        model_name_or_path, "model.safetensors.index.json"
    )
    single_shard_path = os.path.join(model_name_or_path, "model.safetensors")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
    elif os.path.isfile(single_shard_path):
        with safe_open(single_shard_path, framework="pt", device="cpu") as shard:
            weight_map = {key: "model.safetensors" for key in shard.keys()}
    else:
        raise FileNotFoundError(
            "B3 strict checkpoint reload requires model.safetensors or "
            f"model.safetensors.index.json under {model_name_or_path}"
        )

    model_state = base_model.state_dict()
    checkpoint_keys = set(weight_map)
    unexpected_keys = sorted(checkpoint_keys.difference(model_state))
    if unexpected_keys:
        raise RuntimeError(
            "B3 checkpoint contains keys absent from the vendored model; "
            f"first keys: {unexpected_keys[:8]}"
        )

    tied_keys = getattr(base_model, "_tied_weights_keys", {})
    if isinstance(tied_keys, dict):
        allowed_missing = set(tied_keys)
    else:
        allowed_missing = set(tied_keys or ())
    missing_keys = sorted(
        set(model_state).difference(checkpoint_keys).difference(allowed_missing)
    )
    if missing_keys:
        raise RuntimeError(
            "B3 checkpoint is missing non-tied model weights; first keys: "
            f"{missing_keys[:8]}"
        )

    keys_by_shard = {}
    for key, shard_name in weight_map.items():
        keys_by_shard.setdefault(shard_name, []).append(key)

    loaded_count = 0
    with torch.no_grad():
        for shard_name, expected_keys in sorted(keys_by_shard.items()):
            shard_path = os.path.join(model_name_or_path, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                actual_keys = set(shard.keys())
                absent_keys = sorted(set(expected_keys).difference(actual_keys))
                if absent_keys:
                    raise RuntimeError(
                        f"Safetensors shard {shard_name} is missing indexed "
                        f"keys: {absent_keys[:8]}"
                    )
                for key in expected_keys:
                    source = shard.get_tensor(key)
                    target = model_state[key]
                    if tuple(source.shape) != tuple(target.shape):
                        raise RuntimeError(
                            f"B3 checkpoint shape mismatch for {key}: "
                            f"checkpoint={tuple(source.shape)}, "
                            f"model={tuple(target.shape)}"
                        )
                    if target.device.type == "meta":
                        raise RuntimeError(
                            f"Cannot strictly reload {key}: target is still meta"
                        )
                    target.copy_(source.to(device=target.device, dtype=target.dtype))
                    loaded_count += 1

    base_model.tie_weights()
    print_master(
        "[B3_WEIGHT_RELOAD] strictly restored "
        f"{loaded_count} checkpoint tensors from "
        f"{len(keys_by_shard)} safetensors shard(s); "
        f"allowed_tied_missing={sorted(allowed_missing)}"
    )


def _validate_qwen3_5_runtime_state(base_model: nn.Module) -> None:
    """Fail closed on the native Qwen3.5 state needed by B3/GradCache."""
    config = base_model.config
    if getattr(config, "model_type", None) != "qwen3_5":
        raise ValueError(
            "Qwen3.5 loader returned an incompatible config: "
            f"model_type={getattr(config, 'model_type', None)!r}"
        )
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    if text_config is None or vision_config is None:
        raise ValueError("Qwen3.5 config must contain text_config and vision_config")
    if getattr(vision_config, "patch_size", None) != 16:
        raise ValueError(
            "Qwen3.5 model/processor contract requires vision patch_size=16, "
            f"got {getattr(vision_config, 'patch_size', None)}"
        )
    if getattr(vision_config, "spatial_merge_size", None) != 2:
        raise ValueError(
            "Qwen3.5 model/processor contract requires spatial_merge_size=2, "
            f"got {getattr(vision_config, 'spatial_merge_size', None)}"
        )

    meta_parameters = [
        name for name, parameter in base_model.named_parameters()
        if parameter.device.type == "meta"
    ]
    if meta_parameters:
        raise RuntimeError(
            "Qwen3.5 parameters remained on the meta device after loading: "
            + ", ".join(meta_parameters[:5])
        )

    rotary_buffers = [
        (name, buffer)
        for name, buffer in base_model.named_buffers()
        if name.endswith("inv_freq") or name.endswith("original_inv_freq")
    ]
    if not rotary_buffers:
        raise RuntimeError("No Qwen3.5 rotary-frequency buffers were found")
    for name, buffer in rotary_buffers:
        if buffer.device.type == "meta":
            raise RuntimeError(f"Qwen3.5 RoPE buffer {name} remained on meta")
        if buffer.numel() == 0 or not torch.isfinite(buffer).all():
            raise FloatingPointError(f"Qwen3.5 RoPE buffer {name} is invalid")
        if buffer.min().item() <= 0 or buffer.max().item() > 1.000001:
            raise ValueError(
                f"Qwen3.5 RoPE buffer {name} has unexpected range "
                f"[{buffer.min().item()}, {buffer.max().item()}]"
            )
    print_master(
        "[QWEN3_5_RUNTIME_OK] native Transformers Qwen3.5 state validated: "
        f"hidden_size={text_config.hidden_size}, layers={text_config.num_hidden_layers}, "
        f"rope_buffers={len(rotary_buffers)}, patch_size={vision_config.patch_size}, "
        f"merge_size={vision_config.spatial_merge_size}"
    )


def _verify_qwen3_5_checkpoint_sentinels(
    base_model: nn.Module,
    model_name_or_path: str,
) -> None:
    """Verify native Qwen3.5 weights survived ``from_pretrained`` unchanged.

    The Qwen2 reproduction exposed a Transformers lifecycle where a successful
    loading progress bar was followed by reinitialization.  Qwen3.5 is native
    to Transformers 5.9 and should not need a second full checkpoint load, but
    compare fixed slices from several independent checkpoint tensors so that a
    similar failure cannot remain silent.
    """
    from safetensors import safe_open

    if not os.path.isdir(model_name_or_path):
        raise ValueError(
            "Qwen3.5 B3 reproduction requires a local base-model directory, "
            f"got {model_name_or_path!r}"
        )
    index_path = os.path.join(model_name_or_path, "model.safetensors.index.json")
    single_path = os.path.join(model_name_or_path, "model.safetensors")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
    elif os.path.isfile(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as shard:
            weight_map = {key: "model.safetensors" for key in shard.keys()}
    else:
        raise FileNotFoundError(
            "Qwen3.5 checkpoint verification requires safetensors weights under "
            f"{model_name_or_path}"
        )

    model_state = base_model.state_dict()
    text_config = getattr(base_model.config, "text_config", None)
    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise RuntimeError(
            "Qwen3.5 checkpoint verification could not determine the text "
            f"layer count: {num_hidden_layers!r}"
        )
    last_layer = num_hidden_layers - 1
    candidates = (
        "model.visual.patch_embed.proj.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        f"model.language_model.layers.{last_layer}.mlp.down_proj.weight",
        "model.language_model.norm.weight",
    )
    verified = []
    with torch.no_grad():
        for checkpoint_key in candidates:
            if checkpoint_key not in weight_map:
                continue
            possible_model_keys = (
                checkpoint_key,
                checkpoint_key.removeprefix("model."),
            )
            model_key = next(
                (key for key in possible_model_keys if key in model_state),
                None,
            )
            if model_key is None:
                raise RuntimeError(
                    "Qwen3.5 checkpoint sentinel has no matching model tensor: "
                    f"{checkpoint_key}"
                )
            shard_path = os.path.join(
                model_name_or_path, weight_map[checkpoint_key]
            )
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                checkpoint_slice = shard.get_slice(checkpoint_key)
                shape = tuple(checkpoint_slice.get_shape())
                windows = []
                for fraction in (0.0, 0.5, 1.0):
                    slices = []
                    for dimension in shape:
                        width = min(4, dimension)
                        start = int((dimension - width) * fraction)
                        slices.append(slice(start, start + width))
                    windows.append(tuple(slices))

                sampled_values = []
                for window_index, slices in enumerate(windows):
                    expected = checkpoint_slice[slices]
                    actual = model_state[model_key][slices].detach().cpu()
                    expected = expected.to(dtype=actual.dtype)
                    if not torch.equal(actual, expected):
                        max_abs = (
                            actual.float() - expected.float()
                        ).abs().max().item()
                        raise RuntimeError(
                            "Qwen3.5 base weight differs from its checkpoint "
                            "after from_pretrained: "
                            f"checkpoint={checkpoint_key}, model={model_key}, "
                            f"window={window_index}, max_abs={max_abs}"
                        )
                    sampled_values.append(actual.float().reshape(-1))
            sampled = torch.cat(sampled_values)
            print_master(
                "[QWEN3_5_WEIGHT_SENTINEL] "
                f"key={checkpoint_key}, dtype={model_state[model_key].dtype}, "
                f"shape={shape}, samples={sampled.numel()}, "
                f"min={sampled.min().item():.8g}, "
                f"max={sampled.max().item():.8g}, "
                f"mean={sampled.mean().item():.8g}, "
                f"std={sampled.std(unbiased=False).item():.8g}"
            )
            verified.append(checkpoint_key)
    if len(verified) < 4:
        raise RuntimeError(
            "Could not verify enough Qwen3.5 checkpoint sentinels; found "
            f"{verified}. The checkpoint layout is not a supported native "
            "Qwen3.5 model."
        )
    print_master(
        "[QWEN3_5_WEIGHT_VERIFY] matched native model tensors against "
        f"{len(verified)} checkpoint sentinels"
    )

    # Cover every base-model tensor as well. Released Qwen3.5 checkpoints may
    # also ship MTP tensors that Qwen3_5Model intentionally does not instantiate;
    # those are the only checkpoint entries excluded here. Sampling three windows
    # from every tensor catches partial/layer-local reinitialization without
    # making a second 4.5 GB copy of the checkpoint on every rank.
    checkpoint_base_keys = {
        checkpoint_key.removeprefix("model."): checkpoint_key
        for checkpoint_key in weight_map
        if checkpoint_key.startswith("model.")
    }
    missing_checkpoint_keys = sorted(set(model_state) - set(checkpoint_base_keys))
    missing_model_keys = sorted(set(checkpoint_base_keys) - set(model_state))
    if missing_checkpoint_keys or missing_model_keys:
        raise RuntimeError(
            "Qwen3.5 model/checkpoint key coverage mismatch: "
            f"missing_checkpoint={missing_checkpoint_keys[:5]}, "
            f"missing_model={missing_model_keys[:5]}"
        )

    keys_by_shard = {}
    for model_key, checkpoint_key in checkpoint_base_keys.items():
        keys_by_shard.setdefault(weight_map[checkpoint_key], []).append(
            (model_key, checkpoint_key)
        )
    sampled_tensor_count = 0
    with torch.no_grad():
        for shard_name, shard_keys in sorted(keys_by_shard.items()):
            shard_path = os.path.join(model_name_or_path, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for model_key, checkpoint_key in sorted(shard_keys):
                    checkpoint_slice = shard.get_slice(checkpoint_key)
                    checkpoint_shape = tuple(checkpoint_slice.get_shape())
                    model_tensor = model_state[model_key]
                    if checkpoint_shape != tuple(model_tensor.shape):
                        raise RuntimeError(
                            "Qwen3.5 checkpoint shape mismatch: "
                            f"checkpoint={checkpoint_key} {checkpoint_shape}, "
                            f"model={model_key} {tuple(model_tensor.shape)}"
                        )
                    for window_index, fraction in enumerate((0.0, 0.5, 1.0)):
                        slices = []
                        for dimension in checkpoint_shape:
                            width = min(2, dimension)
                            start = int((dimension - width) * fraction)
                            slices.append(slice(start, start + width))
                        slices = tuple(slices)
                        expected = checkpoint_slice[slices]
                        actual = model_tensor[slices].detach().cpu()
                        expected = expected.to(dtype=actual.dtype)
                        if not torch.equal(actual, expected):
                            max_abs = (
                                actual.float() - expected.float()
                            ).abs().max().item()
                            raise RuntimeError(
                                "Qwen3.5 base tensor differs from checkpoint: "
                                f"checkpoint={checkpoint_key}, model={model_key}, "
                                f"window={window_index}, max_abs={max_abs}"
                            )
                    sampled_tensor_count += 1
    print_master(
        "[QWEN3_5_WEIGHT_COVERAGE] exactly matched first/middle/last windows "
        f"for all {sampled_tensor_count} base-model tensors; excluded_mtp="
        f"{sum(key.startswith('mtp.') for key in weight_map)}"
    )


def _load_qwen3_5_base_model(model_name_or_path: str) -> nn.Module:
    """Load the exact Transformers-5.9 native Qwen3.5 base model."""
    import importlib.metadata

    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_attn_2_available,
        is_flash_linear_attention_available,
    )

    transformers_version = importlib.metadata.version("transformers")
    if transformers_version != "5.9.0":
        raise RuntimeError(
            "The Qwen3.5 B3 path is validated against transformers==5.9.0, "
            f"but the active environment has {transformers_version}."
        )
    has_causal_conv1d = is_causal_conv1d_available()
    has_flash_linear_attention = is_flash_linear_attention_available()
    has_flash_attention_2 = is_flash_attn_2_available()
    has_fast_linear_attention = bool(modeling_qwen3_5.is_fast_path_available)
    print_master(
        "[QWEN3_5_KERNELS] "
        f"causal_conv1d={has_causal_conv1d}, "
        f"flash_linear_attention={has_flash_linear_attention}, "
        f"flash_attention_2={has_flash_attention_2}, "
        f"linear_fast_path={has_fast_linear_attention}"
    )
    if not has_flash_attention_2:
        raise RuntimeError(
            "Qwen3.5 B3 is configured for flash_attention_2, but a compatible "
            "FlashAttention-2 installation is unavailable."
        )
    if (
        not has_fast_linear_attention
        and os.environ.get("QWEN3_5_REQUIRE_FAST_KERNELS", "0") == "1"
    ):
        raise RuntimeError(
            "Qwen3.5 fast Gated DeltaNet kernels are unavailable. Its official "
            "PyTorch fallback is correct but impractically slow for this B3 "
            "training run and can look like a GPU/NCCL hang. Install compatible "
            "flash-linear-attention and causal-conv1d builds, or set "
            "QWEN3_5_REQUIRE_FAST_KERNELS=0 only for a short correctness probe."
        )

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
    )
    config.use_cache = False
    config.padding_side = "left"
    config._attn_implementation = "flash_attention_2"
    if hasattr(config, "text_config"):
        config.text_config.use_cache = False
        config.text_config._attn_implementation = "flash_attention_2"
    if hasattr(config, "vision_config"):
        config.vision_config._attn_implementation = "flash_attention_2"
    base_model = Qwen3_5Model.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    _validate_qwen3_5_runtime_state(base_model)
    _verify_qwen3_5_checkpoint_sentinels(base_model, model_name_or_path)
    return base_model


def _validate_qwen3_5_b3_adapter(
    lora_model: nn.Module,
    expect_fresh_initialization: bool = False,
) -> None:
    """Require the exact architecture-aware B3 adapter scope."""
    peft_configs = getattr(lora_model, "peft_config", {})
    active_adapter = getattr(lora_model, "active_adapter", "default")
    adapter_config = peft_configs.get(active_adapter)
    if adapter_config is None and len(peft_configs) == 1:
        adapter_config = next(iter(peft_configs.values()))
    if adapter_config is None:
        raise RuntimeError("Qwen3.5 B3 adapter has no active PEFT config")
    # Rank, alpha and dropout are experiment hyperparameters, not Qwen3.5
    # architecture invariants.  Record their actual values below, but do not
    # force every B3/in-batch/EGA ablation to use the original r=8 profile.
    config_errors = []
    if not getattr(adapter_config, "use_dora", False):
        config_errors.append("use_dora=False (expected True)")
    configured_exclusions = getattr(adapter_config, "exclude_modules", None)
    if configured_exclusions:
        config_errors.append(
            f"exclude_modules={sorted(configured_exclusions)} (expected none)"
        )
    if config_errors:
        raise RuntimeError(
            "Qwen3.5 adapter structural invariant mismatch: "
            + ", ".join(config_errors)
        )

    lora_parameter_names = [
        name
        for name, _ in lora_model.named_parameters()
        if ".lora_A." in name
    ]
    required_suffixes = (
        ".self_attn.q_proj.lora_A.default.weight",
        ".self_attn.k_proj.lora_A.default.weight",
        ".self_attn.v_proj.lora_A.default.weight",
        ".self_attn.o_proj.lora_A.default.weight",
        ".linear_attn.in_proj_qkv.lora_A.default.weight",
        ".linear_attn.in_proj_z.lora_A.default.weight",
        ".linear_attn.in_proj_b.lora_A.default.weight",
        ".linear_attn.in_proj_a.lora_A.default.weight",
        ".linear_attn.out_proj.lora_A.default.weight",
        ".mlp.down_proj.lora_A.default.weight",
    )
    missing = [
        suffix
        for suffix in required_suffixes
        if not any(name.endswith(suffix) for name in lora_parameter_names)
    ]
    allowed_fragments = (
        ".self_attn.q_proj.",
        ".self_attn.k_proj.",
        ".self_attn.v_proj.",
        ".self_attn.o_proj.",
        ".linear_attn.in_proj_qkv.",
        ".linear_attn.in_proj_z.",
        ".linear_attn.in_proj_b.",
        ".linear_attn.in_proj_a.",
        ".linear_attn.out_proj.",
        ".mlp.down_proj.",
    )
    unexpected = [
        name
        for name in lora_parameter_names
        if not any(fragment in name for fragment in allowed_fragments)
    ]
    visual_lora = [name for name in lora_parameter_names if ".visual." in name]
    base_model = lora_model.get_base_model()
    text_config = getattr(base_model.config, "text_config", None)
    layer_types = list(getattr(text_config, "layer_types", ()) or ())
    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    if (
        not isinstance(num_hidden_layers, int)
        or num_hidden_layers <= 0
        or len(layer_types) != num_hidden_layers
    ):
        raise RuntimeError(
            "Qwen3.5 B3 LoRA validation requires one layer_type per text "
            f"layer; layers={num_hidden_layers}, layer_types={len(layer_types)}"
        )
    expected_lora_parameters = set()
    full_attention_layers = 0
    linear_attention_layers = 0
    for layer_index, layer_type in enumerate(layer_types):
        layer_prefix = (
            "base_model.model.language_model.layers."
            f"{layer_index}"
        )
        if layer_type == "full_attention":
            full_attention_layers += 1
            projections = (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            )
        elif layer_type == "linear_attention":
            linear_attention_layers += 1
            projections = (
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.in_proj_b",
                "linear_attn.in_proj_a",
                "linear_attn.out_proj",
            )
        else:
            raise RuntimeError(
                "Unsupported Qwen3.5 text layer type for B3 LoRA mapping: "
                f"layer={layer_index}, type={layer_type!r}"
            )
        for projection in (*projections, "mlp.down_proj"):
            expected_lora_parameters.add(
                f"{layer_prefix}.{projection}.lora_A.default.weight"
            )
    actual_lora_parameters = set(lora_parameter_names)
    missing_layer_targets = sorted(
        expected_lora_parameters - actual_lora_parameters
    )
    extra_layer_targets = sorted(
        actual_lora_parameters - expected_lora_parameters
    )
    if (
        missing
        or unexpected
        or visual_lora
        or missing_layer_targets
        or extra_layer_targets
    ):
        raise RuntimeError(
            "Qwen3.5 B3 LoRA target validation failed: "
            f"missing={missing}, unexpected={unexpected[:5]}, "
            f"visual_targets={visual_lora[:5]}, "
            f"missing_layer_targets={missing_layer_targets[:5]}, "
            f"extra_layer_targets={extra_layer_targets[:5]}"
        )

    trainable_non_adapter = [
        name
        for name, parameter in lora_model.named_parameters()
        if parameter.requires_grad
        and "lora_" not in name
        and "lora_magnitude_vector" not in name
    ]
    if trainable_non_adapter:
        raise RuntimeError(
            "Qwen3.5 B3 unexpectedly left backbone parameters trainable: "
            + ", ".join(trainable_non_adapter[:5])
        )

    if expect_fresh_initialization:
        lora_a = [
            parameter
            for name, parameter in lora_model.named_parameters()
            if ".lora_A." in name
        ]
        lora_b = [
            parameter
            for name, parameter in lora_model.named_parameters()
            if ".lora_B." in name
        ]
        dora_magnitude = [
            (name, parameter)
            for name, parameter in lora_model.named_parameters()
            if "lora_magnitude_vector" in name
        ]
        if not lora_a or not lora_b or not dora_magnitude:
            raise RuntimeError(
                "Qwen3.5 B3 fresh adapter is missing LoRA-A, LoRA-B or DoRA "
                "magnitude parameters"
            )
        if not all(torch.isfinite(parameter).all() for parameter in lora_a):
            raise FloatingPointError("Qwen3.5 fresh LoRA-A contains NaN/Inf")
        if not any(torch.count_nonzero(parameter).item() for parameter in lora_a):
            raise RuntimeError("Qwen3.5 fresh LoRA-A is unexpectedly all zero")
        if any(torch.count_nonzero(parameter).item() for parameter in lora_b):
            raise RuntimeError("Qwen3.5 fresh LoRA-B must initialize exactly zero")
        invalid_dora_magnitude = []
        for name, parameter in dora_magnitude:
            finite = torch.isfinite(parameter)
            nonpositive = finite & (parameter <= 0)
            if not finite.all() or nonpositive.any():
                finite_values = parameter.detach()[finite]
                invalid_dora_magnitude.append(
                    f"{name}: shape={tuple(parameter.shape)}, "
                    f"finite={int(finite.sum().item())}/{parameter.numel()}, "
                    f"nonpositive={int(nonpositive.sum().item())}, "
                    f"min={float(finite_values.min().item()) if finite_values.numel() else 'n/a'}, "
                    f"max={float(finite_values.max().item()) if finite_values.numel() else 'n/a'}"
                )
        if invalid_dora_magnitude:
            raise FloatingPointError(
                "Qwen3.5 fresh DoRA magnitude must be finite and positive; "
                "offending parameters: " + "; ".join(invalid_dora_magnitude)
            )
        print_master(
            "[QWEN3_5_ADAPTER_INIT_OK] fresh LoRA-A is finite/nonzero, "
            "LoRA-B is exactly zero, DoRA magnitude is finite/positive, and "
            "the frozen backbone has no trainable parameters"
        )
    print_master(
        "[QWEN3_5_LORA_OK] LoRA/DoRA covers the B3-mapped full-attention, "
        "Gated DeltaNet and MLP-down projections exactly; "
        f"r={getattr(adapter_config, 'r', None)}, "
        f"alpha={getattr(adapter_config, 'lora_alpha', None)}, "
        f"dropout={getattr(adapter_config, 'lora_dropout', None)}, "
        f"layers={num_hidden_layers}, full_attention={full_attention_layers}, "
        f"linear_attention={linear_attention_layers}, "
        f"modules={len(lora_parameter_names)}"
    )


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    """
    from transformers import AutoProcessor
    model_name_or_path = (
        model_args.processor_name
        or model_args.checkpoint_path
        or model_args.model_name
    )
    print_master(f'Loading processor from: {model_name_or_path}')

    if model_args.model_backbone == QWEN2_VL:
        from src.training.vlm_backbone.qwen2_vl import (
            Qwen2TokenizerFast,
            Qwen2VLImageProcessor,
            Qwen2VLProcessor,
        )
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        # B3 uses its vendored Qwen2TokenizerFast explicitly rather than
        # AutoTokenizer.
        # Under Transformers 5.9, from_pretrained() can choose the slow-to-fast
        # conversion path even though this Qwen2 checkpoint contains a complete
        # tokenizer.json.  That conversion silently produces a BPE backend with
        # zero merges.  Load the checkpoint's Rust backend explicitly and pass
        # it into from_pretrained(), which still restores the HF tokenizer
        # metadata while bypassing the incompatible conversion.
        from tokenizers import Tokenizer

        tokenizer_file = os.path.join(model_name_or_path, "tokenizer.json")
        if not os.path.isfile(tokenizer_file):
            raise FileNotFoundError(
                "B3 Qwen2 tokenizer.json was not found at "
                f"{tokenizer_file}"
            )
        tokenizer_object = Tokenizer.from_file(tokenizer_file)
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_name_or_path,
            tokenizer_object=tokenizer_object,
        )
        # Transformers 5.x may silently rebuild this vendored fast tokenizer
        # through an incompatible slow-tokenizer class.  The resulting object
        # still has the right vocabulary size, but loses every BPE merge and
        # tokenizes the same B3 samples into much longer sequences.  Validate
        # the actual Rust backend rather than trusting vocab_size.
        tokenizer_backend = json.loads(tokenizer.backend_tokenizer.to_str())
        tokenizer_model = tokenizer_backend.get("model") or {}
        tokenizer_merges = tokenizer_model.get("merges") or []
        tokenizer_pre_tokenizer = tokenizer_backend.get("pre_tokenizer")
        tokenizer_normalizer = tokenizer_backend.get("normalizer")
        if len(tokenizer_merges) != 151387:
            raise RuntimeError(
                "B3 Qwen2 tokenizer backend is corrupted: expected 151387 "
                f"BPE merges, found {len(tokenizer_merges)}. Check that the "
                "vendored tokenization_qwen2.py is being used."
            )
        if tokenizer_pre_tokenizer is None:
            raise RuntimeError(
                "B3 Qwen2 tokenizer backend is missing its pre-tokenizer"
            )
        if tokenizer_normalizer != {"type": "NFC"}:
            raise RuntimeError(
                "B3 Qwen2 tokenizer backend has the wrong normalizer: "
                f"{tokenizer_normalizer!r}"
            )
        print_master(
            "[B3_TOKENIZER_OK] "
            f"vocab={len(tokenizer_model.get('vocab') or {})}, "
            f"merges={len(tokenizer_merges)}, "
            f"pre_tokenizer={tokenizer_pre_tokenizer.get('type')}, "
            f"normalizer={tokenizer_normalizer.get('type')}"
        )
        tokenizer.padding_side = "left"
        embedding_token_spec = None
        if getattr(model_args, "qwen2_embedding_token", False):
            embedding_token_spec = _register_qwen2_embedding_token(
                tokenizer,
                model_args.qwen2_embedding_token_text,
            )
            expected_token_id = getattr(
                model_args, "qwen2_embedding_token_id", embedding_token_spec["id"]
            )
            if embedding_token_spec["id"] != expected_token_id:
                raise RuntimeError(
                    "Qwen2 model/tokenizer disagree on the embedding token ID: "
                    f"model={expected_token_id}, tokenizer={embedding_token_spec['id']}"
                )
        # Match B3's construction path. Passing the already-created components
        # keeps the vendored implementations while from_pretrained restores
        # processor metadata from the model directory.
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        if processor.tokenizer is not tokenizer:
            raise RuntimeError(
                "Qwen2VLProcessor replaced the validated B3 tokenizer during "
                "construction"
            )
        processor_backend = json.loads(
            processor.tokenizer.backend_tokenizer.to_str()
        )
        if len((processor_backend.get("model") or {}).get("merges") or []) != 151387:
            raise RuntimeError(
                "Qwen2VLProcessor contains a tokenizer without the 151387 "
                "B3 Qwen2 BPE merges"
            )
        if embedding_token_spec:
            processor._qwen2_embedding_token_enabled = True
            processor._qwen2_embedding_token_text = embedding_token_spec["text"]
            processor._qwen2_embedding_token_id = embedding_token_spec["id"]
            print_master(
                "[QWEN2_EMBED_TOKEN_PROCESSOR_OK] "
                f"token={embedding_token_spec['text']!r} "
                f"id={embedding_token_spec['id']} "
                f"tokenizer_size={embedding_token_spec['tokenizer_size']} "
                "padding_side=left append_position=last_active"
            )
        print_master(
            "Qwen2-VL B3 processor stack: "
            f"processor={type(processor).__module__}.{type(processor).__name__}, "
            f"image_processor={type(image_processor).__module__}.{type(image_processor).__name__}, "
            f"tokenizer={type(tokenizer).__module__}.{type(tokenizer).__name__}, "
            f"padding_side={tokenizer.padding_side}, "
            f"min_pixels={image_processor.min_pixels}, "
            f"max_pixels={image_processor.max_pixels}, "
            f"serialized_size={getattr(image_processor, 'size', None)}"
        )
    elif model_args.model_backbone == QWEN3_5_VL:
        min_pixels = data_args.resize_min_pixels if data_args else 28 * 28 * 4
        max_pixels = data_args.resize_max_pixels if data_args else 28 * 28 * 1280
        processor_source = model_name_or_path
        if (
            model_args.checkpoint_path
            and not os.path.isfile(
                os.path.join(model_args.checkpoint_path, "processor_config.json")
            )
        ):
            processor_source = model_args.model_name
        processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=False,
            padding_side='left',
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        processor.tokenizer.padding_side = "left"
        # Transformers 5.9 may construct Qwen3VLProcessor with a
        # Qwen2VLImageProcessor that does not persist arbitrary min/max pixel
        # attributes. Keep the B3 resize contract on the top-level processor;
        # qwen-vl-utils consumes these explicit values and the image processor
        # is called with do_resize=False afterward.
        processor._b3_resize_min_pixels = int(min_pixels)
        processor._b3_resize_max_pixels = int(max_pixels)
        if type(processor).__name__ != "Qwen3VLProcessor":
            raise TypeError(
                "Qwen3.5 must use Transformers Qwen3VLProcessor, got "
                f"{type(processor).__module__}.{type(processor).__name__}"
            )
        if getattr(processor.image_processor, "patch_size", None) != 16:
            raise ValueError(
                "Qwen3.5 processor has the wrong image patch size: "
                f"{getattr(processor.image_processor, 'patch_size', None)}"
            )
        if getattr(processor.image_processor, "merge_size", None) != 2:
            raise ValueError(
                "Qwen3.5 processor has the wrong image merge size: "
                f"{getattr(processor.image_processor, 'merge_size', None)}"
            )
        tokenizer_backend = json.loads(
            processor.tokenizer.backend_tokenizer.to_str()
        )
        tokenizer_model = tokenizer_backend.get("model") or {}
        tokenizer_vocab = tokenizer_model.get("vocab") or {}
        tokenizer_merges = tokenizer_model.get("merges") or []
        tokenizer_normalizer = tokenizer_backend.get("normalizer")
        tokenizer_pre_tokenizer = tokenizer_backend.get("pre_tokenizer")
        if tokenizer_model.get("type") != "BPE":
            raise RuntimeError(
                "Qwen3.5 tokenizer backend is not BPE: "
                f"{tokenizer_model.get('type')!r}"
            )
        if len(tokenizer_vocab) != 248044 or len(tokenizer_merges) != 247587:
            raise RuntimeError(
                "Qwen3.5 tokenizer backend is corrupted: expected "
                "vocab=248044 and merges=247587, found "
                f"vocab={len(tokenizer_vocab)} and merges={len(tokenizer_merges)}"
            )
        if tokenizer_normalizer != {"type": "NFC"}:
            raise RuntimeError(
                "Qwen3.5 tokenizer has the wrong normalizer: "
                f"{tokenizer_normalizer!r}"
            )
        if tokenizer_pre_tokenizer is None:
            raise RuntimeError("Qwen3.5 tokenizer has no pre-tokenizer")
        base_config = AutoConfig.from_pretrained(
            model_args.model_name,
            trust_remote_code=False,
        )
        expected_special_ids = {
            VLM_IMAGE_TOKENS[QWEN3_5_VL]: base_config.image_token_id,
            VLM_VIDEO_TOKENS[QWEN3_5_VL]: base_config.video_token_id,
        }
        for token, expected_id in expected_special_ids.items():
            actual_id = processor.tokenizer.convert_tokens_to_ids(token)
            if actual_id != expected_id:
                raise RuntimeError(
                    f"Qwen3.5 special token {token} has id {actual_id}, "
                    f"expected {expected_id}"
                )
        print_master(
            "[QWEN3_5_TOKENIZER_OK] "
            f"vocab={len(tokenizer_vocab)}, merges={len(tokenizer_merges)}, "
            f"pre_tokenizer={tokenizer_pre_tokenizer.get('type')}, "
            f"normalizer={tokenizer_normalizer.get('type')}, "
            f"image_token_id={base_config.image_token_id}, "
            f"video_token_id={base_config.video_token_id}"
        )
        print_master(
            "[QWEN3_5_PROCESSOR_OK] "
            f"processor={type(processor).__module__}.{type(processor).__name__}, "
            f"padding_side={processor.tokenizer.padding_side}, "
            f"patch_size={processor.image_processor.patch_size}, "
            f"merge_size={processor.image_processor.merge_size}, "
            f"min_pixels={processor._b3_resize_min_pixels}, "
            f"max_pixels={processor._b3_resize_max_pixels}"
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


# ==================== MMEBModel ====================
class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.qwen2_model_impl = "native"
        self.qwen2_b3_legacy_position_ids = False
        self.qwen2_embedding_token = False
        self.qwen2_embedding_token_text = None
        self.qwen2_embedding_token_id = None
        self._qwen2_embedding_token_contract_calls = 0
        self._qwen2_b3_position_ids_logged = False
        self._numeric_debug_enabled = int(
            os.environ.get("B3_NUMERIC_DEBUG_STEPS", "0")
        ) > 0
        self._numeric_hidden_reported = False
        self._numeric_pool_reported = False
        self._mean_pool_contract_reported = False
        self._numeric_trainable_preflight_done = False
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _validate_qwen2_embedding_token_inputs(self, input_ids, attention_mask):
        if not self.qwen2_embedding_token:
            return
        if self.pooling not in {"last", "eos"}:
            raise RuntimeError(
                "Qwen2 embedding token reached a non-last pooling path: "
                f"{self.pooling!r}"
            )
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "Qwen2 embedding-token input_ids/attention_mask must be equal "
                f"rank-2 tensors, got {tuple(input_ids.shape)} and "
                f"{tuple(attention_mask.shape)}"
            )
        token_id = self.qwen2_embedding_token_id
        if not isinstance(token_id, int):
            raise RuntimeError("Qwen2 embedding token ID was not initialized")
        active = attention_mask.bool()
        lengths = active.sum(dim=1)
        if (lengths <= 0).any():
            raise ValueError("Qwen2 embedding-token batch contains an empty row")
        token_counts = ((input_ids == token_id) & active).sum(dim=1)
        if not torch.all(token_counts == 1):
            raise ValueError(
                "Each Qwen2 input must contain exactly one active embedding "
                f"token (id={token_id}); counts={token_counts.tolist()}"
            )
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0).expand_as(input_ids)
        last_indices = positions.masked_fill(~active, -1).max(dim=1).values
        last_ids = input_ids.gather(1, last_indices.unsqueeze(1)).squeeze(1)
        if not torch.all(last_ids == token_id):
            raise ValueError(
                "Qwen2 embedding token must be the last active token; "
                f"last_ids={last_ids.tolist()} expected={token_id}"
            )
        if self._qwen2_embedding_token_contract_calls < 2:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                "[QWEN2_EMBED_TOKEN_BATCH_OK] "
                f"rank={rank} call={self._qwen2_embedding_token_contract_calls + 1} "
                f"batch={input_ids.shape[0]} padded_length={input_ids.shape[1]} "
                f"original_active_min={int((lengths - 1).min().item())} "
                f"original_active_max={int((lengths - 1).max().item())} "
                f"final_active_min={int(lengths.min().item())} "
                f"final_active_max={int(lengths.max().item())} "
                f"token_id={token_id} pooling=last",
                flush=True,
            )
        self._qwen2_embedding_token_contract_calls += 1

    def encode_input(self, input):
        if (
            self._numeric_debug_enabled
            and not self._numeric_trainable_preflight_done
        ):
            trainable_parameters = [
                (name, parameter)
                for name, parameter in self.encoder.named_parameters()
                if parameter.requires_grad
            ]
            with torch.no_grad():
                parameter_checks = [
                    torch.isfinite(parameter).all()
                    for _, parameter in trainable_parameters
                ]
                all_trainable_finite = bool(
                    torch.stack(parameter_checks).all().item()
                )
                first_bad_parameter = None
                if not all_trainable_finite:
                    for name, parameter in trainable_parameters:
                        if not torch.isfinite(parameter).all().item():
                            first_bad_parameter = name
                            break
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"[B3_MODEL_PREFLIGHT] rank={rank} component=trainable "
                f"parameters_finite={all_trainable_finite} "
                f"parameter_count={len(trainable_parameters)} "
                f"first_bad_parameter={first_bad_parameter}",
                flush=True,
            )
            self._numeric_trainable_preflight_done = True

        # Filter non-model inputs before passing to encoder
        model_input_keys = {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw',
                            'pixel_values_videos', 'video_grid_thw', 'mm_token_type_ids'}
        filtered_input = {k: v for k, v in input.items() if k in model_input_keys}

        if (
            getattr(self, "model_backbone", None) == QWEN2_VL
            and self.qwen2_model_impl == "native"
        ):
            # B3 keeps multimodal values as per-example lists so text-only and
            # visual samples can coexist in one batch. Its vendored Qwen2-VL
            # model packs those lists internally; the Transformers-native
            # model expects concatenated tensors at its public boundary.
            filtered_input = dict(filtered_input)
            device = filtered_input["input_ids"].device
            # Transformers 5.9 Qwen2-VL consumes this modality map when it
            # computes M-RoPE. The collator creates the same convention used
            # by the native processor: text=0, image=1, video=2.
            packed_visuals = []
            for values_key, grid_key in (
                ("pixel_values", "image_grid_thw"),
                ("pixel_values_videos", "video_grid_thw"),
            ):
                values = filtered_input.get(values_key)
                grids = filtered_input.get(grid_key)
                if values is None:
                    # The B3 collator always emits both keys, including for
                    # text-only chunks. Native Transformers treats an absent
                    # visual pair differently from a half-present pair.
                    filtered_input.pop(values_key, None)
                    filtered_input.pop(grid_key, None)
                elif isinstance(values, (list, tuple)):
                    if not isinstance(grids, (list, tuple)) or len(values) != len(grids):
                        raise ValueError(
                            f"Mismatched B3 multimodal lists: "
                            f"{values_key}={len(values)}, "
                            f"{grid_key}={type(grids).__name__}"
                        )
                    valid_values = []
                    valid_grids = []
                    for value, grid in zip(values, grids):
                        if value is None or grid is None:
                            continue
                        value_tensor = (
                            value
                            if isinstance(value, torch.Tensor)
                            else torch.from_numpy(value)
                        )
                        grid_tensor = (
                            grid
                            if isinstance(grid, torch.Tensor)
                            else torch.from_numpy(grid)
                        ).reshape(-1, 3)
                        if grid_tensor.numel() == 0 or grid_tensor.sum().item() <= 0:
                            continue
                        valid_values.append(value_tensor)
                        valid_grids.append(grid_tensor)

                    if valid_values:
                        filtered_input[values_key] = torch.cat(valid_values, dim=0).to(device)
                        filtered_input[grid_key] = torch.cat(valid_grids, dim=0).to(
                            device=device,
                            dtype=torch.long,
                        )
                        packed_visuals.append(
                            f"{values_key}={tuple(filtered_input[values_key].shape)},"
                            f"{grid_key}={tuple(filtered_input[grid_key].shape)}"
                        )
                    else:
                        filtered_input.pop(values_key, None)
                        filtered_input.pop(grid_key, None)
                elif isinstance(values, torch.Tensor):
                    if not isinstance(grids, torch.Tensor):
                        raise TypeError(
                            f"{values_key} is a tensor but {grid_key} is "
                            f"{type(grids).__name__}"
                        )
                    filtered_input[values_key] = values.to(device)
                    filtered_input[grid_key] = grids.reshape(-1, 3).to(
                        device=device,
                        dtype=torch.long,
                    )
                    packed_visuals.append(
                        f"{values_key}={tuple(filtered_input[values_key].shape)},"
                        f"{grid_key}={tuple(filtered_input[grid_key].shape)}"
                    )
                else:
                    raise TypeError(
                        f"Unsupported {values_key} type at native Qwen2-VL "
                        f"boundary: {type(values).__name__}"
                    )

            if self.qwen2_b3_legacy_position_ids:
                # Experimental approximation only: B3's vendored model uses
                # stateful rope_deltas, so its first chunk and later chunks do
                # not share one stable rule. Applying sequential positions to
                # every chunk is replay-safe, but is not an exact reproduction.
                batch_size, sequence_length = filtered_input["input_ids"].shape
                filtered_input["position_ids"] = torch.arange(
                    sequence_length,
                    device=device,
                    dtype=torch.long,
                ).view(1, 1, -1).expand(3, batch_size, -1)
                if not self._qwen2_b3_position_ids_logged:
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    print(
                        f"[B3_QWEN2_POSITION_IDS] rank={rank} "
                        "mode=legacy_sequential_with_left_padding",
                        flush=True,
                    )
                    self._qwen2_b3_position_ids_logged = True

            pack_log_flag = (
                "_native_qwen2_visual_pack_logged"
                if packed_visuals
                else "_native_qwen2_text_pack_logged"
            )
            if not getattr(self, pack_log_flag, False):
                rank = dist.get_rank() if dist.is_initialized() else 0
                visual_summary = "; ".join(packed_visuals) or "text-only"
                input_ids = filtered_input["input_ids"]
                image_token_id = getattr(self.config, "image_token_id", None)
                video_token_id = getattr(self.config, "video_token_id", None)
                image_tokens = (
                    (input_ids == image_token_id).sum().item()
                    if image_token_id is not None
                    else 0
                )
                video_tokens = (
                    (input_ids == video_token_id).sum().item()
                    if video_token_id is not None
                    else 0
                )
                print(
                    f"[B3_NATIVE_PACK] rank={rank} "
                    f"input_ids={tuple(filtered_input['input_ids'].shape)} "
                    f"{visual_summary} "
                    f"image_tokens={image_tokens} "
                    f"video_tokens={video_tokens}",
                    flush=True,
                )
                setattr(self, pack_log_flag, True)
        elif getattr(self, "model_backbone", None) == QWEN3_5_VL:
            # GradCache slices multimodal inputs by sample, while native
            # Qwen3.5 consumes one concatenated patch tensor and one concatenated
            # grid tensor. Repack only the current chunk at the model boundary.
            filtered_input = dict(filtered_input)
            device = filtered_input["input_ids"].device
            packed_visuals = []
            for values_key, grid_key in (
                ("pixel_values", "image_grid_thw"),
                ("pixel_values_videos", "video_grid_thw"),
            ):
                values = filtered_input.get(values_key)
                grids = filtered_input.get(grid_key)
                if not isinstance(values, (list, tuple)) or not isinstance(
                    grids, (list, tuple)
                ):
                    raise TypeError(
                        "Qwen3.5 B3 inputs must preserve per-sample visual "
                        f"lists until the model boundary; got {values_key}="
                        f"{type(values).__name__}, {grid_key}={type(grids).__name__}"
                    )
                if len(values) != len(grids):
                    raise ValueError(
                        f"Qwen3.5 visual list mismatch: {values_key}="
                        f"{len(values)}, {grid_key}={len(grids)}"
                    )
                valid_values = []
                valid_grids = []
                for value, grid in zip(values, grids):
                    if value is None and grid is None:
                        continue
                    if value is None or grid is None:
                        raise ValueError(
                            f"Qwen3.5 {values_key}/{grid_key} must be paired"
                        )
                    value_tensor = (
                        value
                        if isinstance(value, torch.Tensor)
                        else torch.from_numpy(value)
                    )
                    grid_tensor = (
                        grid
                        if isinstance(grid, torch.Tensor)
                        else torch.from_numpy(grid)
                    ).reshape(-1, 3)
                    if grid_tensor.numel() == 0 or (grid_tensor <= 0).any():
                        raise ValueError(
                            f"Qwen3.5 {grid_key} contains an invalid grid"
                        )
                    valid_values.append(value_tensor)
                    valid_grids.append(grid_tensor)
                if valid_values:
                    filtered_input[values_key] = torch.cat(
                        valid_values, dim=0
                    ).to(device)
                    filtered_input[grid_key] = torch.cat(
                        valid_grids, dim=0
                    ).to(device=device, dtype=torch.long)
                    packed_visuals.append(
                        f"{values_key}={tuple(filtered_input[values_key].shape)},"
                        f"{grid_key}={tuple(filtered_input[grid_key].shape)}"
                    )
                else:
                    filtered_input.pop(values_key, None)
                    filtered_input.pop(grid_key, None)

            input_ids = filtered_input["input_ids"]
            attention_mask = filtered_input["attention_mask"]
            mm_token_type_ids = filtered_input.get("mm_token_type_ids")
            if mm_token_type_ids is None:
                raise ValueError("Qwen3.5 requires mm_token_type_ids")
            if mm_token_type_ids.shape != input_ids.shape:
                raise ValueError(
                    "Qwen3.5 input_ids/mm_token_type_ids shape mismatch: "
                    f"{tuple(input_ids.shape)} != {tuple(mm_token_type_ids.shape)}"
                )
            active_types = mm_token_type_ids[attention_mask.bool()]
            if not torch.all((active_types >= 0) & (active_types <= 2)):
                raise ValueError("Qwen3.5 mm_token_type_ids must be 0, 1 or 2")

            merge_size = self.config.vision_config.spatial_merge_size
            modality_specs = (
                ("image_grid_thw", self.config.image_token_id, 1),
                ("video_grid_thw", self.config.video_token_id, 2),
            )
            for grid_key, token_id, token_type in modality_specs:
                placeholder_count = int(
                    ((input_ids == token_id) & attention_mask.bool()).sum().item()
                )
                typed_count = int(
                    ((mm_token_type_ids == token_type) & attention_mask.bool())
                    .sum()
                    .item()
                )
                if placeholder_count != typed_count:
                    raise ValueError(
                        f"Qwen3.5 {grid_key} placeholder/type mismatch: "
                        f"{placeholder_count} != {typed_count}"
                    )
                grid = filtered_input.get(grid_key)
                expected_features = (
                    int((grid.prod(dim=-1) // merge_size**2).sum().item())
                    if grid is not None
                    else 0
                )
                if placeholder_count != expected_features:
                    raise ValueError(
                        f"Qwen3.5 {grid_key} placeholder/feature mismatch: "
                        f"tokens={placeholder_count}, features={expected_features}"
                    )

            pack_log_flag = (
                "_qwen3_5_visual_pack_logged"
                if packed_visuals
                else "_qwen3_5_text_pack_logged"
            )
            if not getattr(self, pack_log_flag, False):
                print(
                    f"[B3_QWEN3_5_PACK] rank="
                    f"{dist.get_rank() if dist.is_initialized() else 0} "
                    f"input_ids={tuple(input_ids.shape)} "
                    f"{'; '.join(packed_visuals) or 'text-only'} "
                    f"image_tokens={(input_ids == self.config.image_token_id).sum().item()} "
                    f"video_tokens={(input_ids == self.config.video_token_id).sum().item()}",
                    flush=True,
                )
                setattr(self, pack_log_flag, True)
        elif (
            getattr(self, "model_backbone", None) == QWEN2_VL
            and self.qwen2_model_impl == "b3"
        ):
            # B3's vendored Qwen2-VL consumes the collator's per-example
            # pixel/grid lists directly and computes M-RoPE internally. The
            # Transformers-5.9-only modality map is not part of its forward
            # signature.
            filtered_input.pop("mm_token_type_ids", None)

        self._validate_qwen2_embedding_token_inputs(
            filtered_input["input_ids"],
            filtered_input["attention_mask"],
        )

        log_fingerprint("model_input", inputs=filtered_input)
        model_output = self.encoder(
            **filtered_input, return_dict=True, output_hidden_states=True
        )
        if getattr(model_output, "hidden_states", None) is not None:
            hidden_states = model_output.hidden_states[-1]
        elif getattr(model_output, "last_hidden_state", None) is not None:
            hidden_states = model_output.last_hidden_state
        else:
            raise RuntimeError(
                f"{type(self.encoder).__name__} returned neither hidden_states "
                "nor last_hidden_state"
            )
        log_fingerprint("last_hidden_state", hidden_states=hidden_states)
        if self._numeric_debug_enabled and not self._numeric_hidden_reported:
            finite_by_row = torch.isfinite(hidden_states).reshape(
                hidden_states.shape[0], -1
            ).all(dim=1)
            if not finite_by_row.all():
                rank = dist.get_rank() if dist.is_initialized() else 0
                bad_rows = torch.nonzero(
                    ~finite_by_row, as_tuple=False
                ).flatten().tolist()
                print(
                    f"[B3_MODEL_NUMERIC] rank={rank} stage=last_hidden_state "
                    f"shape={tuple(hidden_states.shape)} bad_rows={bad_rows}",
                    flush=True,
                )
                self._numeric_hidden_reported = True
        pooled_output = self._pooling(hidden_states, input['attention_mask'])
        log_fingerprint("pooled_output", pooled_output=pooled_output)
        return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        elif self.pooling == 'mean':
            if attention_mask.ndim != 2:
                raise ValueError(
                    "Mean pooling expects a rank-2 attention mask, got "
                    f"shape={tuple(attention_mask.shape)}"
                )
            if last_hidden_state.shape[:2] != attention_mask.shape:
                raise ValueError(
                    "Mean pooling hidden-state/mask shape mismatch: "
                    f"hidden={tuple(last_hidden_state.shape)} "
                    f"mask={tuple(attention_mask.shape)}"
                )
            active_mask = attention_mask.to(
                device=last_hidden_state.device, dtype=torch.bool
            )
            token_counts = active_mask.sum(dim=1)
            if torch.any(token_counts == 0):
                empty_rows = torch.nonzero(
                    token_counts == 0, as_tuple=False
                ).flatten().tolist()
                raise ValueError(
                    "All-token mean pooling received empty input rows: "
                    f"{empty_rows}"
                )

            # Accumulate in FP32 for long visual sequences. masked_fill is used
            # instead of multiplying by zero so a non-finite padding state can
            # never contaminate the pooled representation.
            masked_hidden = last_hidden_state.float().masked_fill(
                ~active_mask.unsqueeze(-1), 0.0
            )
            reps = masked_hidden.sum(dim=1) / token_counts.unsqueeze(-1).float()

            if not self._mean_pool_contract_reported:
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    f"[ALL_MEAN_POOLING_OK] rank={rank} "
                    f"tokens_per_sample={token_counts.tolist()} "
                    f"hidden_dtype={last_hidden_state.dtype} "
                    "accumulation_dtype=torch.float32 "
                    "padding_excluded=True content_scope=all_active_tokens",
                    flush=True,
                )
                self._mean_pool_contract_reported = True
        else:
            raise NotImplementedError(f"Unsupported pooling method: {self.pooling!r}")
        if self._numeric_debug_enabled and not self._numeric_pool_reported:
            finite_by_row = torch.isfinite(reps).reshape(
                reps.shape[0], -1
            ).all(dim=1)
            if not finite_by_row.all():
                rank = dist.get_rank() if dist.is_initialized() else 0
                bad_rows = torch.nonzero(
                    ~finite_by_row, as_tuple=False
                ).flatten().tolist()
                mask_lengths = attention_mask.sum(dim=1).tolist()
                print(
                    f"[B3_MODEL_NUMERIC] rank={rank} stage=pooled_before_normalize "
                    f"bad_rows={bad_rows} mask_lengths={mask_lengths}",
                    flush=True,
                )
                self._numeric_pool_reported = True
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        base_model_name_or_path = model_args.model_name
        if model_args.init_lora_model:
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(model_args.model_name)
            base_model_name_or_path = (
                model_args.init_lora_base_model
                or peft_config.base_model_name_or_path
            )
            config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)

        model_backbone = get_backbone_name(hf_config=config)
        # Make build() self-contained. Training used to write this back only
        # after build returned, while standalone callers such as the Qwen3.5
        # preflight immediately call load_processor() and otherwise fall into
        # its generic processor branch.
        model_args.model_backbone = model_backbone
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')
        qwen2_embedding_token_spec = None

        # Loading the base model
        if model_backbone == QWEN2_VL:
            if model_args.qwen2_model_impl not in {"native", "b3"}:
                raise ValueError(
                    "--qwen2_model_impl must be either 'native' or 'b3', "
                    f"got {model_args.qwen2_model_impl!r}"
                )
            if model_args.qwen2_model_impl == "b3":
                from src.training.vlm_backbone.qwen2_vl import (
                    Qwen2VLConfig,
                    Qwen2VLForConditionalGeneration,
                )
                if model_args.qwen2_b3_attention_impl not in {
                    "eager",
                    "sdpa",
                    "flash_attention_2",
                }:
                    raise ValueError(
                        "--qwen2_b3_attention_impl must be one of "
                        "'eager', 'sdpa', or 'flash_attention_2', got "
                        f"{model_args.qwen2_b3_attention_impl!r}"
                    )
                qwen2_config_cls = Qwen2VLConfig
                attention_implementation = (
                    model_args.qwen2_b3_attention_impl
                )
                attention_message = (
                    "sdpa(math_backend)"
                    if attention_implementation == "sdpa"
                    else attention_implementation
                )
                implementation_message = (
                    "Using repository B3-compatible "
                    "Qwen2VLForConditionalGeneration under Transformers 5.9 "
                    f"with attention={attention_message}"
                )
            else:
                from transformers import Qwen2VLForConditionalGeneration
                qwen2_config_cls = AutoConfig
                attention_implementation = "flash_attention_2"
                implementation_message = (
                    "Using Transformers-native Qwen2VLForConditionalGeneration "
                    "for Transformers 5.9/PyTorch 2.8 compatibility"
                )
            # The repository B3 model is the Transformers 4.48-era
            # implementation and requires its matching flat Qwen2VLConfig.
            # Transformers 5.9 AutoConfig creates a nested text_config instead
            # (and therefore has no top-level vocab_size/hidden_size).
            config = qwen2_config_cls.from_pretrained(
                base_model_name_or_path,
                trust_remote_code=True,
            )
            if model_args.qwen2_model_impl == "b3":
                required_config_fields = (
                    "vocab_size",
                    "hidden_size",
                    "num_hidden_layers",
                )
                missing_config_fields = [
                    field
                    for field in required_config_fields
                    if not hasattr(config, field)
                ]
                if missing_config_fields:
                    raise ValueError(
                        "B3 Qwen2-VL was loaded with an incompatible config "
                        f"{type(config).__name__}; missing top-level fields: "
                        f"{missing_config_fields}"
                    )
            config._attn_implementation = attention_implementation
            if hasattr(config, "vision_config"):
                config.vision_config._attn_implementation = (
                    attention_implementation
                )
            config.padding_side = "left"
            config.use_cache = False
            print_master(implementation_message)
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_name_or_path,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attention_implementation,
            )
            if model_args.qwen2_model_impl == "b3":
                _strict_reload_qwen2_b3_safetensors(
                    base_model, base_model_name_or_path
                )
                _reset_and_validate_qwen2_b3_rope_buffers(base_model)
            qwen2_embedding_token_spec = _prepare_qwen2_embedding_token_base_model(
                base_model,
                model_args,
                base_model_name_or_path,
            )
        elif model_backbone == QWEN3_5_VL:
            # Qwen3.5 is native in the pinned Transformers 5.9 environment.
            # Use the base multimodal model explicitly: this avoids loading an
            # unused 0.5B-token LM head and fixes the PEFT module namespace for
            # both training and evaluation.
            print_master(
                "Using Transformers-native Qwen3_5Model with B3 training "
                "semantics"
            )
            base_model = _load_qwen3_5_base_model(base_model_name_or_path)
            if model_args.qwen3_5_bidirectional_attention:
                from src.training.qwen3_5_bidirectional import (
                    enable_qwen3_5_bidirectional_full_attention,
                )

                bidirectional_config = (
                    enable_qwen3_5_bidirectional_full_attention(base_model)
                )
                model_args.qwen3_5_bidirectional_runtime_config = (
                    bidirectional_config
                )
                print_master(
                    "[QWEN3_5_BIDIRECTIONAL_ATTENTION_OK] "
                    f"full_attention_layers="
                    f"{bidirectional_config['full_attention_layer_indices']} "
                    f"gated_deltanet_layers="
                    f"{bidirectional_config['gated_deltanet_layer_indices']} "
                    "full_attention_causal=False gated_deltanet_causal=True"
                )
            _enable_qwen3_5_stable_dora_norm()
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        if model_args.init_lora_model:
            model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
            print_master(f'Loading LoRA from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=False)
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=False)
            lora_model = lora_model.merge_and_unload()
            base_model = lora_model.to(dtype=torch.bfloat16)

        if model_args.lora and model_args.moe_lora:
            raise ValueError(
                "--lora and --moe_lora are mutually exclusive. The isolated "
                "TSEmbed branch supplies its own dense MoE-DoRA experts."
            )

        if model_args.moe_lora:
            if model_backbone not in {QWEN2_VL, QWEN3_5_VL}:
                raise ValueError(
                    "The isolated TSEmbed MoE-DoRA architecture currently "
                    "supports Qwen2-VL and Qwen3.5-VL, got "
                    f"{model_backbone!r}"
                )
            if model_args.init_lora_model:
                raise ValueError(
                    "--init_lora_model cannot be combined with --moe_lora"
                )
            from src.training.moe_lora import build_dense_moe_lora

            moe_model = build_dense_moe_lora(
                base_model,
                base_model_name_or_path=base_model_name_or_path,
                rank=model_args.lora_r,
                alpha=model_args.lora_alpha,
                dropout=model_args.lora_dropout,
                target_modules=model_args.lora_target_modules.split(','),
                num_experts=model_args.moe_num_experts,
                router_temperature=model_args.moe_router_temperature,
            )
            model = cls(
                encoder=moe_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        elif model_args.lora:
            print_master(
                "Initializing fresh LoRA/DoRA adapter on "
                f"{base_model.__class__.__name__}"
            )
            embedding_token_lora_kwargs = {}
            if qwen2_embedding_token_spec is not None:
                embedding_token_lora_kwargs = {
                    "trainable_token_indices": [
                        qwen2_embedding_token_spec["id"]
                    ],
                    "ensure_weight_tying": True,
                }
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False,
                **embedding_token_lora_kwargs,
            )
            lora_model = get_peft_model(base_model, lora_config)
            _validate_qwen2_embedding_token_adapter(
                lora_model,
                qwen2_embedding_token_spec,
                expect_trainable=True,
            )
            if model_backbone == QWEN3_5_VL:
                _validate_qwen3_5_b3_adapter(
                    lora_model,
                    expect_fresh_initialization=True,
                )
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        # encode_input needs this before the first training forward to adapt
        # B3's per-example multimodal lists for the native Qwen2-VL model.
        model.model_backbone = model_backbone
        model.qwen2_model_impl = model_args.qwen2_model_impl
        model.qwen2_b3_legacy_position_ids = (
            model_args.qwen2_b3_legacy_position_ids
        )
        model.qwen2_embedding_token = bool(qwen2_embedding_token_spec)
        model.qwen2_embedding_token_text = (
            qwen2_embedding_token_spec["text"]
            if qwen2_embedding_token_spec
            else None
        )
        model.qwen2_embedding_token_id = (
            qwen2_embedding_token_spec["id"]
            if qwen2_embedding_token_spec
            else None
        )
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_args.model_backbone}] from {model_name_or_path}')
        qwen2_embedding_token_spec = None

        if model_args.model_backbone == QWEN2_VL:
            if model_args.qwen2_model_impl not in {"native", "b3"}:
                raise ValueError(
                    "--qwen2_model_impl must be either 'native' or 'b3', "
                    f"got {model_args.qwen2_model_impl!r}"
                )
            if model_args.qwen2_model_impl == "b3":
                from src.training.vlm_backbone.qwen2_vl import (
                    Qwen2VLConfig,
                    Qwen2VLForConditionalGeneration,
                )
                if model_args.qwen2_b3_attention_impl not in {
                    "eager",
                    "sdpa",
                    "flash_attention_2",
                }:
                    raise ValueError(
                        "--qwen2_b3_attention_impl must be one of "
                        "'eager', 'sdpa', or 'flash_attention_2', got "
                        f"{model_args.qwen2_b3_attention_impl!r}"
                    )
                qwen2_config_cls = Qwen2VLConfig
                attention_implementation = (
                    model_args.qwen2_b3_attention_impl
                )
                attention_message = (
                    "sdpa(math_backend)"
                    if attention_implementation == "sdpa"
                    else attention_implementation
                )
                implementation_message = (
                    "Using repository B3-compatible "
                    "Qwen2VLForConditionalGeneration under Transformers 5.9 "
                    f"with attention={attention_message}"
                )
            else:
                from transformers import Qwen2VLForConditionalGeneration
                qwen2_config_cls = AutoConfig
                attention_implementation = "flash_attention_2"
                implementation_message = (
                    "Using Transformers-native Qwen2VLForConditionalGeneration "
                    "for Transformers 5.9/PyTorch 2.8 compatibility"
                )
            config = qwen2_config_cls.from_pretrained(
                model_args.model_name,
                trust_remote_code=True,
            )
            if model_args.qwen2_model_impl == "b3":
                required_config_fields = (
                    "vocab_size",
                    "hidden_size",
                    "num_hidden_layers",
                )
                missing_config_fields = [
                    field
                    for field in required_config_fields
                    if not hasattr(config, field)
                ]
                if missing_config_fields:
                    raise ValueError(
                        "B3 Qwen2-VL was loaded with an incompatible config "
                        f"{type(config).__name__}; missing top-level fields: "
                        f"{missing_config_fields}"
                    )
            config._attn_implementation = attention_implementation
            if hasattr(config, "vision_config"):
                config.vision_config._attn_implementation = (
                    attention_implementation
                )
            config.padding_side = "left"
            config.use_cache = False
            print_master(implementation_message)
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config,
                attn_implementation=attention_implementation,
            )
            if model_args.qwen2_model_impl == "b3":
                _strict_reload_qwen2_b3_safetensors(
                    base_model, model_args.model_name
                )
                _reset_and_validate_qwen2_b3_rope_buffers(base_model)
            qwen2_embedding_token_spec = _prepare_qwen2_embedding_token_base_model(
                base_model,
                model_args,
                model_name_or_path,
            )
        elif model_args.model_backbone == QWEN3_5_VL:
            print_master(
                "Loading Transformers-native Qwen3_5Model for B3-consistent "
                "evaluation"
            )
            base_model = _load_qwen3_5_base_model(model_args.model_name)
            if model_args.qwen3_5_bidirectional_attention:
                from src.training.qwen3_5_bidirectional import (
                    enable_qwen3_5_bidirectional_full_attention,
                )

                bidirectional_config = (
                    enable_qwen3_5_bidirectional_full_attention(base_model)
                )
                model_args.qwen3_5_bidirectional_runtime_config = (
                    bidirectional_config
                )
                print_master(
                    "[QWEN3_5_BIDIRECTIONAL_ATTENTION_OK] "
                    f"full_attention_layers="
                    f"{bidirectional_config['full_attention_layer_indices']} "
                    f"gated_deltanet_layers="
                    f"{bidirectional_config['gated_deltanet_layer_indices']} "
                    "full_attention_causal=False gated_deltanet_causal=True"
                )
            _enable_qwen3_5_stable_dora_norm()
        else:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        # Building the model on top of the base
        if model_args.lora and model_args.moe_lora:
            raise ValueError(
                "--lora and --moe_lora are mutually exclusive. The isolated "
                "TSEmbed branch supplies its own dense MoE-DoRA experts."
            )

        if model_args.moe_lora:
            if model_args.model_backbone not in {QWEN2_VL, QWEN3_5_VL}:
                raise ValueError(
                    "The isolated TSEmbed MoE-DoRA checkpoint path supports "
                    "Qwen2-VL and Qwen3.5-VL, got "
                    f"{model_args.model_backbone!r}"
                )
            from src.training.moe_lora import (
                build_dense_moe_lora,
                load_dense_moe_lora_adapter,
            )

            moe_model = build_dense_moe_lora(
                base_model,
                base_model_name_or_path=model_args.model_name,
                rank=model_args.lora_r,
                alpha=model_args.lora_alpha,
                dropout=model_args.lora_dropout,
                target_modules=model_args.lora_target_modules.split(','),
                num_experts=model_args.moe_num_experts,
                router_temperature=model_args.moe_router_temperature,
            )
            load_dense_moe_lora_adapter(
                moe_model,
                model_name_or_path,
                is_trainable=is_trainable,
            )
            model = cls(
                encoder=moe_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        elif model_args.lora:
            print_master(f'Loading LoRA from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)
            _validate_qwen2_embedding_token_adapter(
                lora_model,
                qwen2_embedding_token_spec,
                expect_trainable=is_trainable,
            )
            if model_args.model_backbone == QWEN3_5_VL:
                _validate_qwen3_5_b3_adapter(lora_model)
            lora_b_parameters = [
                (name, parameter)
                for name, parameter in lora_model.named_parameters()
                if "lora_B." in name
            ]
            if not lora_b_parameters:
                raise RuntimeError(
                    f"No LoRA-B parameters were created while loading "
                    f"{model_name_or_path}. The checkpoint and backbone are "
                    "not compatible."
                )
            nonzero_lora_b = sum(
                int(torch.count_nonzero(parameter.detach()).item() > 0)
                for _, parameter in lora_b_parameters
            )
            nonfinite_lora = [
                name
                for name, parameter in lora_model.named_parameters()
                if (
                    ("lora_" in name or "lora_magnitude_vector" in name)
                    and not torch.isfinite(parameter.detach()).all()
                )
            ]
            if nonfinite_lora:
                raise FloatingPointError(
                    "Non-finite LoRA parameters after checkpoint load: "
                    + ", ".join(nonfinite_lora[:5])
                )
            if nonzero_lora_b == 0:
                raise RuntimeError(
                    "All loaded LoRA-B tensors are zero. This normally means "
                    "the adapter keys did not match the selected backbone "
                    "(for example, a Transformers-native Qwen2-VL checkpoint "
                    "was loaded into B3's vendored Qwen2-VL model)."
                )
            print_master(
                f"LoRA checkpoint integrity: {nonzero_lora_b}/"
                f"{len(lora_b_parameters)} LoRA-B tensors are non-zero"
            )
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        model.model_backbone = model_args.model_backbone
        model.qwen2_model_impl = model_args.qwen2_model_impl
        model.qwen2_b3_legacy_position_ids = (
            model_args.qwen2_b3_legacy_position_ids
        )
        model.qwen2_embedding_token = bool(qwen2_embedding_token_spec)
        model.qwen2_embedding_token_text = (
            qwen2_embedding_token_spec["text"]
            if qwen2_embedding_token_spec
            else None
        )
        model.qwen2_embedding_token_id = (
            qwen2_embedding_token_spec["id"]
            if qwen2_embedding_token_spec
            else None
        )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt) if tgt else None  # (bsz_per_device, dim)

        # Store reps for monitoring (used by B3DebugCallback)
        if self.training:
            self._last_qry_reps = qry_reps.detach() if qry_reps is not None else None
            self._last_tgt_reps = tgt_reps.detach() if tgt_reps is not None else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
