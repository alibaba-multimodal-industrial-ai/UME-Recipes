"""Read-only single-GPU preflight for the Qwen3.5 B3 training path."""

import argparse
import os

import torch
from PIL import Image

from src.training.arguments import DataArguments, ModelArguments
from src.training.data.collator import Qwen3_5_VL_process_fn
from src.training.model import MMEBModel, load_processor
from src.training.utils import batch_to_device


LORA_TARGETS = (
    "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,"
    "in_proj_a,out_proj,down_proj"
)


def _check_representations(name, representations, hidden_size):
    if representations.shape != (1, hidden_size):
        raise RuntimeError(
            f"{name} embedding has shape {tuple(representations.shape)}, "
            f"expected (1, {hidden_size})"
        )
    if not torch.isfinite(representations).all():
        raise FloatingPointError(f"{name} embedding contains NaN/Inf")
    norm = torch.linalg.vector_norm(representations.float(), dim=-1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=2e-3, rtol=0):
        raise RuntimeError(f"{name} embedding is not normalized: norm={norm}")
    print(
        f"[QWEN3_5_PREFLIGHT_REP_OK] side={name}, "
        f"shape={tuple(representations.shape)}, norm={norm.item():.8g}, "
        f"min={representations.min().item():.8g}, "
        f"max={representations.max().item():.8g}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3.5-2B",
    )
    parser.add_argument("--min_pixels", type=int, default=4096)
    parser.add_argument("--max_pixels", type=int, default=2257920)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", default=LORA_TARGETS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 B3 preflight requires one CUDA GPU")
    os.environ.setdefault("QWEN3_5_REQUIRE_FAST_KERNELS", "1")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    model_args = ModelArguments(
        model_name=args.model_name,
        pooling="eos",
        normalize=True,
        temperature=0.02,
        lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    data_args = DataArguments(
        resize_use_processor=True,
        resize_min_pixels=args.min_pixels,
        resize_max_pixels=args.max_pixels,
    )

    model = MMEBModel.build(model_args)
    if model_args.model_backbone != "qwen3_5_vl":
        raise RuntimeError(
            "Qwen3.5 preflight resolved the wrong backbone: "
            f"{model_args.model_backbone!r}"
        )
    processor = load_processor(model_args, data_args)
    model.model_backbone = "qwen3_5_vl"
    model.processor = processor
    hidden_size = model.encoder.get_base_model().config.text_config.hidden_size
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    image = Image.new("RGB", (96, 64), color=(37, 113, 211))
    image_batch = Qwen3_5_VL_process_fn(
        {
            "text": ["<|image_pad|>Describe this image."],
            "images": [[image]],
        },
        processor,
    )
    text_batch = Qwen3_5_VL_process_fn(
        {
            "text": ["A short text-only B3 preflight sample."],
            "images": [[None]],
        },
        processor,
    )

    image_batch = batch_to_device(image_batch, device)
    text_batch = batch_to_device(text_batch, device)
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        image_reps = model(qry=image_batch)["qry_reps"]
        text_reps = model(tgt=text_batch)["tgt_reps"]
    _check_representations("image", image_reps, hidden_size)
    _check_representations("text", text_reps, hidden_size)
    print(
        "[QWEN3_5_PREFLIGHT_OK] checkpoint, runtime kernels, tokenizer, "
        "processor, adapter initialization and image/text forwards all passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
