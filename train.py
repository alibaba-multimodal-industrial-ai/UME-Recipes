# Adapted from Tevatron code
import logging
import os.path
import sys

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import wandb
import yaml
import transformers
from transformers import HfArgumentParser
from transformers.trainer_callback import TrainerCallback
from src.training.arguments import ModelArguments, DataArguments, TrainingArguments
from src.training.data.collator import MultimodalDataCollator
from src.training.data.loader import init_concat_dataset
from src.training.data.dataset import CustomRandomSampler, VLM2VecInBatchSampler
from src.training.model import (
    MMEBModel,
    validate_qwen2_embedding_token_checkpoint,
)
from src.training.trainer import GradCacheLateProcessTrainer
from src.training.utils import print_rank, print_master, find_latest_checkpoint
from src.training.model import load_processor, get_backbone_name
from src.debug_fingerprint import install_model_hooks, log_runtime_and_model_state


import os
import shutil
import torch.distributed as dist
import numpy as np
import math
import hashlib
import json

def delete_pycache(root='.'):
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            if dirname == '__pycache__':
                full_path = os.path.join(dirpath, dirname)
                print(f"Deleting: {full_path}")
                try:
                    shutil.rmtree(full_path)
                except:
                    print(">>>>>", "Module not exists", full_path, flush=True)
                    pass
delete_pycache()

# B3 debug utilities
_B3_DEBUG = os.environ.get("B3_DEBUG", "0") == "1"
_B3_DEBUG_INTERVAL = int(os.environ.get("B3_DEBUG_INTERVAL", "100"))


def _b3_log_model_structure(model):
    """Log model class, top-level modules, LoRA adapters, and parameter counts."""
    if not _B3_DEBUG:
        return
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("[B3_DEBUG] === MODEL STRUCTURE DEBUG ===")
    print("=" * 80)

    # 1. Model class name
    print(f"\n[B3_DEBUG] Model class: {type(model).__name__}")
    print(f"[B3_DEBUG] Encoder class: {type(model.encoder).__name__}")

    # 2. Top-level modules (check for visual encoder)
    print(f"\n[B3_DEBUG] Top-level modules:")
    for name, child in model.named_children():
        print(f"  {name}: {type(child).__name__}")

    # 3. Check for visual encoder
    has_visual = hasattr(model.encoder, 'visual') or hasattr(model.encoder, 'model') and hasattr(model.encoder.model, 'visual')
    print(f"\n[B3_DEBUG] Has visual encoder: {has_visual}")
    if hasattr(model.encoder, 'visual'):
        print(f"[B3_DEBUG] Visual encoder: {type(model.encoder.visual).__name__}")
    elif hasattr(model.encoder, 'model') and hasattr(model.encoder.model, 'visual'):
        print(f"[B3_DEBUG] Visual encoder (via .model): {type(model.encoder.model.visual).__name__}")

    # 4. LoRA module names
    try:
        from peft import PeftModel
        if isinstance(model.encoder, PeftModel):
            peft_config = model.encoder.peft_config
            for adapter_name, config in peft_config.items():
                print(f"\n[B3_DEBUG] LoRA config [{adapter_name}]:")
                print(f"  r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
                print(f"  target_modules={config.target_modules}")
                print(f"  use_dora={config.use_dora}")
                print(f"  init_lora_weights={config.init_lora_weights}")

            # List actual LoRA module paths
            lora_modules = []
            for name, module in model.encoder.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_modules.append(name)
            print(f"\n[B3_DEBUG] LoRA modules ({len(lora_modules)} total):")
            for m in lora_modules[:20]:
                print(f"  {m}")
            if len(lora_modules) > 20:
                print(f"  ... and {len(lora_modules) - 20} more")
    except Exception as e:
        print(f"[B3_DEBUG] Could not inspect LoRA: {e}")

    try:
        from src.training.moe_lora import DenseMoEDoraLinear

        moe_modules = [
            (name, module)
            for name, module in model.encoder.named_modules()
            if isinstance(module, DenseMoEDoraLinear)
        ]
        if moe_modules:
            first = moe_modules[0][1]
            print(f"\n[B3_DEBUG] TSEmbed MoE-DoRA modules ({len(moe_modules)} total):")
            print(
                f"  experts={first.num_experts}, rank_per_expert={first.rank}, "
                f"alpha={first.alpha:g}, dropout={first.dropout_p:g}, "
                f"router_temperature={first.router_temperature:g}, "
                "routing=dense_softmax_per_token, use_dora=True, "
                "EANS=False, two_stage=False"
            )
            for name, _ in moe_modules[:20]:
                print(f"  {name}")
            if len(moe_modules) > 20:
                print(f"  ... and {len(moe_modules) - 20} more")
    except Exception as e:
        print(f"[B3_DEBUG] Could not inspect MoE-DoRA: {e}")

    # 5. Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[B3_DEBUG] Total parameters: {total_params:,}")
    print(f"[B3_DEBUG] Trainable parameters: {trainable_params:,}")
    print(f"[B3_DEBUG] Trainable ratio: {trainable_params / total_params:.4f}")

    # 6. Check if model output has last_hidden_state
    print(f"\n[B3_DEBUG] Model backbone: {getattr(model, 'model_backbone', 'unknown')}")
    print(f"[B3_DEBUG] Pooling: {model.pooling}")
    print(f"[B3_DEBUG] Normalize: {model.normalize}")
    print(f"[B3_DEBUG] Temperature: {model.temperature}")
    print(f"[B3_DEBUG] Mean subtraction: {getattr(model, 'mean_subtraction', False)}")

    print("=" * 80 + "\n")


def _validate_qwen2_vlm2vec_adapter(model, model_args):
    """Fail fast if PEFT silently changes the controlled Qwen2 adapter."""
    if not model_args.lora:
        raise ValueError("Qwen2-VL VLM2Vec control requires LoRA/DoRA")

    expected_requested_targets = {
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "k_proj",
        "q_proj",
        "out_proj",
        "v_proj",
    }
    requested_targets = {
        target.strip()
        for target in model_args.lora_target_modules.split(",")
        if target.strip()
    }
    config_errors = []
    if model_args.lora_r <= 0:
        config_errors.append(f"lora_r={model_args.lora_r}, expected a positive rank")
    if model_args.lora_alpha != 64:
        config_errors.append(f"lora_alpha={model_args.lora_alpha}, expected 64")
    if not math.isclose(model_args.lora_dropout, 0.1, rel_tol=0.0, abs_tol=1e-12):
        config_errors.append(
            f"lora_dropout={model_args.lora_dropout}, expected 0.1"
        )
    if requested_targets != expected_requested_targets:
        config_errors.append(
            f"target_modules={sorted(requested_targets)}, expected "
            f"{sorted(expected_requested_targets)}"
        )
    if config_errors:
        raise ValueError(
            "Qwen2-VL controlled in-batch adapter configuration changed: "
            + "; ".join(config_errors)
        )

    named_parameters = dict(model.named_parameters())
    lora_a = {
        name: parameter
        for name, parameter in named_parameters.items()
        if ".lora_A.default.weight" in name
    }
    lora_b = {
        name: parameter
        for name, parameter in named_parameters.items()
        if ".lora_B.default.weight" in name
    }
    dora = {
        name: parameter
        for name, parameter in named_parameters.items()
        if ".lora_magnitude_vector.default.weight" in name
    }
    num_layers = int(model.config.num_hidden_layers)
    expected_suffix_counts = {
        "q_proj": num_layers,
        "k_proj": num_layers,
        "v_proj": num_layers,
        "o_proj": num_layers,
        "down_proj": num_layers,
    }
    actual_suffix_counts = {suffix: 0 for suffix in expected_suffix_counts}
    unexpected_modules = []
    for name in lora_a:
        module_name = name.split(".lora_A.default.weight", 1)[0]
        suffix = module_name.rsplit(".", 1)[-1]
        if suffix in actual_suffix_counts:
            actual_suffix_counts[suffix] += 1
        else:
            unexpected_modules.append(module_name)
    expected_module_count = sum(expected_suffix_counts.values())
    if (
        actual_suffix_counts != expected_suffix_counts
        or unexpected_modules
        or len(lora_a) != expected_module_count
        or len(lora_b) != expected_module_count
        or len(dora) != expected_module_count
    ):
        raise RuntimeError(
            "Qwen2-VL LoRA/DoRA coverage differs from the successful B3 "
            "control: "
            f"expected_counts={expected_suffix_counts}, "
            f"actual_counts={actual_suffix_counts}, "
            f"A/B/DoRA={len(lora_a)}/{len(lora_b)}/{len(dora)}, "
            f"unexpected={unexpected_modules[:5]}"
        )

    invalid_a = [
        name
        for name, parameter in lora_a.items()
        if not torch.isfinite(parameter).all() or not torch.any(parameter != 0)
    ]
    invalid_b = [
        name
        for name, parameter in lora_b.items()
        if not torch.isfinite(parameter).all() or torch.any(parameter != 0)
    ]
    invalid_dora = [
        name
        for name, parameter in dora.items()
        if not torch.isfinite(parameter).all() or torch.any(parameter <= 0)
    ]
    allowed_trainable_markers = [
        ".lora_A.",
        ".lora_B.",
        ".lora_magnitude_vector.",
    ]
    if model_args.qwen2_embedding_token:
        allowed_trainable_markers.append(".trainable_tokens_delta.")
    unexpected_trainable = [
        name
        for name, parameter in named_parameters.items()
        if parameter.requires_grad
        and not any(marker in name for marker in allowed_trainable_markers)
    ]
    if invalid_a or invalid_b or invalid_dora or unexpected_trainable:
        raise FloatingPointError(
            "Qwen2-VL fresh adapter initialization is invalid: "
            f"bad_lora_a={invalid_a[:3]}, bad_lora_b={invalid_b[:3]}, "
            f"bad_dora={invalid_dora[:3]}, "
            f"unexpected_trainable={unexpected_trainable[:3]}"
        )
    trainable_token_summary = (
        " trainable_token_rows=1" if model_args.qwen2_embedding_token else ""
    )
    print_master(
        "[VLM2VEC_INBATCH_QWEN2_ADAPTER_OK] "
        f"r={model_args.lora_r} alpha={model_args.lora_alpha} "
        f"dropout={model_args.lora_dropout} "
        f"modules={expected_module_count} counts={actual_suffix_counts} "
        "lora_a=finite_nonzero lora_b=finite_zero "
        "dora=finite_positive frozen_backbone=True"
        f"{trainable_token_summary}"
    )


def _validate_qwen2_vlm2vec_moe_adapter(model, model_args):
    """Verify that Qwen2-VL received only the intended dense MoE adapters."""

    from src.training.moe_lora import DenseMoEDoraLinear

    if model_args.lora or not model_args.moe_lora:
        raise ValueError(
            "Qwen2-VL MoE control requires moe_lora=True and lora=False"
        )
    modules = [
        (name, module)
        for name, module in model.encoder.named_modules()
        if isinstance(module, DenseMoEDoraLinear)
    ]
    num_layers = int(model.config.num_hidden_layers)
    expected_suffix_counts = {
        "q_proj": num_layers,
        "k_proj": num_layers,
        "v_proj": num_layers,
        "o_proj": num_layers,
        "down_proj": num_layers,
    }
    actual_suffix_counts = {suffix: 0 for suffix in expected_suffix_counts}
    unexpected_modules = []
    for name, _module in modules:
        suffix = name.rsplit(".", 1)[-1]
        if suffix in actual_suffix_counts:
            actual_suffix_counts[suffix] += 1
        else:
            unexpected_modules.append(name)
    expected_module_count = sum(expected_suffix_counts.values())
    if (
        len(modules) != expected_module_count
        or actual_suffix_counts != expected_suffix_counts
        or unexpected_modules
    ):
        raise RuntimeError(
            "Qwen2-VL MoE-DoRA coverage differs from the reproduced "
            "single-adapter control: "
            f"expected_counts={expected_suffix_counts}, "
            f"actual_counts={actual_suffix_counts}, modules={len(modules)}, "
            f"unexpected={unexpected_modules[:5]}"
        )

    allowed_trainable_markers = (
        ".lora_A",
        ".lora_B",
        ".lora_router",
        ".lora_magnitude_vector",
    )
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(marker in name for marker in allowed_trainable_markers)
    ]
    frozen_adapter = [
        name
        for name, parameter in model.named_parameters()
        if any(marker in name for marker in allowed_trainable_markers)
        and not parameter.requires_grad
    ]
    if unexpected_trainable or frozen_adapter:
        raise RuntimeError(
            "Qwen2-VL MoE-DoRA trainability is invalid: "
            f"unexpected_trainable={unexpected_trainable[:5]}, "
            f"frozen_adapter={frozen_adapter[:5]}"
        )

    first_module = modules[0][1]
    print_master(
        "[VLM2VEC_INBATCH_QWEN2_MOE_ADAPTER_OK] "
        f"modules={len(modules)} counts={actual_suffix_counts} "
        f"experts={first_module.num_experts} "
        f"rank_per_expert={first_module.rank} alpha={first_module.alpha:g} "
        f"dropout={first_module.dropout_p:g} "
        f"router_temperature={first_module.router_temperature:g} "
        "routing=dense_softmax_per_token use_dora=True "
        "visual_adapter_modules=0 frozen_backbone=True"
    )


def compute_embedding_metrics(qry_reps, tgt_reps, temperature=0.02, normalize=False):
    """
    Compute embedding quality metrics from captured reps (no extra forward pass).

    Metrics:
      alignment:       avg squared L2 distance between positive pairs (lower = better)
      uniformity_*:    how uniformly distributed on the hypersphere (lower = better)
      pos/neg_sim:     mean cosine similarity of positive / negative pairs
      sim_margin:      pos_sim - neg_sim (higher = better separation)
      hard_neg_ratio:  fraction of negatives with cosine sim > 0.5
      eff_rank_*:      effective rank (RankMe) — spectral entropy of embedding matrix
      participation_*: (sum(var))² / sum(var²) — how many dims are "active"
      train_acc:       top-1 contrastive accuracy on this batch
    """
    metrics = {}
    device = qry_reps.device
    qry = qry_reps.float()
    tgt = tgt_reps.float()
    B, D = qry.shape
    tpq = tgt.size(0) // B

    # L2-normalize for alignment / uniformity if needed
    if not normalize:
        qry_n = F.normalize(qry, p=2, dim=-1)
        tgt_n = F.normalize(tgt, p=2, dim=-1)
    else:
        qry_n = qry
        tgt_n = tgt

    # Positive target indices: query i matches target i*tpq
    pos_idx = torch.arange(0, B * tpq, tpq, device=device)

    # --- 1. Alignment (Wang & Isola 2020) ---
    pos_tgt = tgt_n[pos_idx]
    metrics["alignment"] = (qry_n - pos_tgt).norm(p=2, dim=1).pow(2).mean().item()

    # --- 2. Uniformity (Wang & Isola 2020) ---
    if B >= 2:
        metrics["uniformity_qry"] = (
            torch.pdist(qry_n, p=2).pow(2).mul(-2).exp().mean().log().item()
        )
    if tgt_n.size(0) >= 2:
        max_pdist = 512
        tgt_sub = tgt_n if tgt_n.size(0) <= max_pdist else tgt_n[
            torch.randperm(tgt_n.size(0), device=device)[:max_pdist]
        ]
        metrics["uniformity_tgt"] = (
            torch.pdist(tgt_sub, p=2).pow(2).mul(-2).exp().mean().log().item()
        )

    # --- 3. Similarity margin ---
    sim = torch.matmul(qry_n, tgt_n.T)  # [B, B*tpq]
    pos_sim = sim[torch.arange(B), pos_idx].mean()
    neg_mask = torch.ones_like(sim, dtype=torch.bool)
    neg_mask[torch.arange(B), pos_idx] = False
    neg_sim = sim[neg_mask].mean()
    metrics["pos_sim"] = pos_sim.item()
    metrics["neg_sim"] = neg_sim.item()
    metrics["sim_margin"] = (pos_sim - neg_sim).item()

    # --- 4. Hard negative ratio ---
    metrics["hard_neg_ratio"] = (sim[neg_mask] > 0.5).float().mean().item()

    # --- 4b. Per-query hardest negative & confusion rate ---
    # neg_sim_max_per_qry: mean of each query's hardest negative sim
    # pos_neg_overlap: fraction of negs that beat the positive (sim_neg > sim_pos)
    neg_sim_per_qry = sim.clone()
    neg_sim_per_qry[torch.arange(B), pos_idx] = -2.0  # mask out positive
    hardest_neg_sim = neg_sim_per_qry.max(dim=1).values  # [B]
    metrics["neg_sim_max_per_qry"] = hardest_neg_sim.mean().item()
    # Per-query: how many negatives have sim > pos_sim?
    pos_sim_per_qry = sim[torch.arange(B), pos_idx]  # [B]
    overlap = (neg_sim_per_qry > pos_sim_per_qry.unsqueeze(1)).float().mean(dim=1)  # [B]
    metrics["pos_neg_overlap"] = overlap.mean().item()

    # --- 4c. Ranking metrics: mean rank & MRR ---
    # Rank of the positive in the sorted similarity list (1-based)
    sim_sorted_idx = sim.argsort(dim=1, descending=True)  # [B, N]
    pos_rank = (sim_sorted_idx == pos_idx.unsqueeze(1)).nonzero()[:, 1] + 1  # 1-based
    metrics["pos_rank_mean"] = pos_rank.float().mean().item()
    metrics["mrr"] = (1.0 / pos_rank.float()).mean().item()

    # --- 5. Effective rank (RankMe) ---
    max_svd = 512
    for name, emb in [("qry", qry), ("tgt", tgt)]:
        n = emb.size(0)
        if n < 2:
            continue
        emb_s = emb if n <= max_svd else emb[
            torch.randperm(n, device=device)[:max_svd]
        ]
        _, s, _ = torch.linalg.svd(emb_s, full_matrices=False)
        s = s / s.sum()
        entropy = -(s * torch.log(s + 1e-12)).sum()
        metrics[f"eff_rank_{name}"] = torch.exp(entropy).item()

    # --- 6. Participation ratio ---
    qry_var = qry.var(dim=0)
    metrics["participation_qry"] = (qry_var.sum() ** 2 / (qry_var ** 2).sum()).item()
    tgt_var = tgt.var(dim=0)
    metrics["participation_tgt"] = (tgt_var.sum() ** 2 / (tgt_var ** 2).sum()).item()

    # --- 7. Training accuracy (top-1 contrastive) ---
    logits = sim / temperature
    preds = logits.argmax(dim=-1)
    metrics["train_acc"] = (preds == pos_idx).float().mean().item()

    # --- 8. Embedding norm (if not normalized) ---
    if not normalize:
        metrics["qry_norm_mean"] = qry.norm(p=2, dim=-1).mean().item()
        metrics["tgt_norm_mean"] = tgt.norm(p=2, dim=-1).mean().item()

    return metrics


class B3DebugCallback(TrainerCallback):
    """
    Lightweight embedding quality monitor — zero extra forward pass.

    Captures embeddings via trainer._monitor_reps (set by GradCache hook)
    and computes:
      - Gradient norms by module type
      - Alignment & Uniformity (Wang & Isola 2020)
      - Similarity margin & hard negative ratio
      - Effective rank (RankMe) & participation ratio
      - Training accuracy

    Activated by env var B3_DEBUG=1. Runs every B3_DEBUG_INTERVAL steps (default 100).
    """

    def __init__(self, model, interval=None):
        self.model = model
        self.interval = interval or _B3_DEBUG_INTERVAL
        self.trainer = None
        self.normalize = getattr(model, "normalize", False)
        self.temperature = getattr(model, "temperature", 0.02)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not _B3_DEBUG:
            return
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return

        try:
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass

        trainer = kwargs.get("trainer") or self.trainer
        if trainer is None:
            return

        print(f"\n{'=' * 80}")
        print(f"[B3_DEBUG] === STEP {state.global_step} EMBEDDING MONITOR ===")
        print(f"{'=' * 80}")

        # 1. Gradient norms
        self._log_gradient_norms(logs)

        # 1b. MoE routing health, when the isolated TSEmbed architecture is in
        # use. These summaries are captured during the existing forward pass.
        self._log_moe_router()

        # 2. Embedding metrics from captured reps (no forward pass!)
        self._log_embedding_metrics(trainer, state.global_step)

        print(f"{'=' * 80}\n")

    def _log_gradient_norms(self, logs):
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gn = param.grad.norm().item()
                if gn > 0:
                    parts = name.split(".")
                    module_type = parts[-2] if len(parts) >= 2 else "other"
                    grad_norms.setdefault(module_type, []).append(gn)

        if grad_norms:
            print("\n[B3_DEBUG] Gradient norms by module type:")
            for mt, norms in sorted(grad_norms.items()):
                arr = np.array(norms)
                print(f"  {mt}: mean={arr.mean():.6f}, max={arr.max():.6f}, count={len(arr)}")
        else:
            logged_gn = logs.get("grad_norm") if logs else None
            if logged_gn:
                print(f"[B3_DEBUG] Gradients already zeroed. Logged grad_norm: {logged_gn:.4f}")

    def _log_moe_router(self):
        if not hasattr(getattr(self.model, "encoder", None), "moe_lora_config"):
            return
        try:
            from src.training.moe_lora import moe_router_diagnostics

            diagnostics = moe_router_diagnostics(self.model)
        except Exception as error:
            print(f"[B3_DEBUG] Error collecting MoE router metrics: {error}")
            return
        if not diagnostics.get("observed_modules"):
            return
        probabilities = diagnostics["mean_expert_probability"]
        entropy = diagnostics["mean_router_entropy"]
        print("[B3_DEBUG] TSEmbed MoE router (no EANS):")
        print(
            "  mean_expert_probability: "
            + "["
            + ", ".join(f"{value:.6f}" for value in probabilities)
            + "]"
        )
        print(f"  mean_router_entropy: {entropy:.6f}")
        print(
            "  normalized_entropy:  "
            f"{diagnostics['normalized_router_entropy']:.6f}"
        )
        print(
            "  max_mean_probability: "
            f"{diagnostics['max_mean_expert_probability']:.6f}"
        )
        print(
            "  expert_prob_cv:       "
            f"{diagnostics['expert_probability_cv']:.6f}"
        )
        print(f"  observed_modules:    {diagnostics['observed_modules']}")

    def _log_embedding_metrics(self, trainer, step):
        reps = getattr(trainer, "_monitor_reps", None)
        if reps is None or len(reps) < 2 or reps[0] is None or reps[1] is None:
            print("\n[B3_DEBUG] No embedding reps captured yet.")
            return

        qry_reps, tgt_reps = reps[0], reps[1]
        B, D = qry_reps.shape
        tpq = tgt_reps.size(0) // B
        print(f"\n[B3_DEBUG] Reps: qry=[{B},{D}] tgt=[{tgt_reps.size(0)},{D}] tpq={tpq}")

        try:
            metrics = compute_embedding_metrics(
                qry_reps, tgt_reps,
                temperature=self.temperature,
                normalize=self.normalize,
            )
        except Exception as e:
            print(f"[B3_DEBUG] Error computing metrics: {e}")
            return

        print("[B3_DEBUG] Alignment & Uniformity:")
        print(f"  alignment:          {metrics.get('alignment', 'N/A'):.6f}")
        print(f"  uniformity_qry:     {metrics.get('uniformity_qry', 'N/A'):.6f}")
        print(f"  uniformity_tgt:     {metrics.get('uniformity_tgt', 'N/A'):.6f}")

        print("[B3_DEBUG] Similarity:")
        # pos_sim: mean cosine similarity of positive (query, target) pairs.
        #   Should increase during training. Low pos_sim = model not learning alignment.
        print(f"  pos_sim:            {metrics.get('pos_sim', 'N/A'):.6f}")
        # neg_sim: mean cosine similarity of all negative pairs (averaged over the batch).
        #   Should decrease during training. High neg_sim = negatives too similar to queries.
        print(f"  neg_sim:            {metrics.get('neg_sim', 'N/A'):.6f}")
        # neg_sim_max_per_qry: mean of each query's hardest negative similarity.
        #   THE most important similarity metric — worst-case confusion.
        #   neg_sim (mean) can be low while a few hard negatives are very close.
        #   If this approaches pos_sim, the model is about to confuse neg with pos.
        print(f"  neg_sim_max/qry:   {metrics.get('neg_sim_max_per_qry', 'N/A'):.6f}")
        # sim_margin: pos_sim - neg_sim (mean). Higher = better separation.
        #   But this doesn't capture worst-case — check neg_sim_max_per_qry for that.
        print(f"  sim_margin:         {metrics.get('sim_margin', 'N/A'):.6f}")
        # hard_neg_ratio: fraction of negatives with cosine sim > 0.5 (fixed threshold).
        #   Coarse indicator of hard negative density. Less precise than pos_neg_overlap.
        print(f"  hard_neg_ratio:     {metrics.get('hard_neg_ratio', 'N/A'):.6f}")
        # pos_neg_overlap: fraction of negatives with sim > their query's positive sim.
        #   True confusion rate — how many negatives actually beat the positive.
        #   This is per-query (relative to each query's own pos_sim), not a fixed threshold.
        #   Should be ~0 at convergence. High = model is struggling.
        print(f"  pos_neg_overlap:   {metrics.get('pos_neg_overlap', 'N/A'):.6f}")

        print("[B3_DEBUG] Ranking:")
        # train_acc: top-1 contrastive accuracy — fraction of queries whose positive
        #   is the highest-similarity target. Binary (correct/wrong), no granularity.
        print(f"  train_acc(top1):   {metrics.get('train_acc', 'N/A'):.4f}")
        # pos_rank_mean: mean rank of the positive in the sorted similarity list (1-based).
        #   1.0 = perfect. If train_acc=0.8 but pos_rank_mean=5.0, the 20% wrong cases
        #   are ranked far down (less concerning than if pos_rank_mean=1.5).
        print(f"  pos_rank_mean:     {metrics.get('pos_rank_mean', 'N/A'):.2f}")
        # mrr: mean reciprocal rank = mean(1/rank). 1.0 = perfect.
        #   Standard IR metric, correlated with MMEB retrieval performance.
        print(f"  mrr:               {metrics.get('mrr', 'N/A'):.4f}")
        if "qry_norm_mean" in metrics:
            # Embedding L2 norms (only printed when not L2-normalized).
            #   If norms drift too far from 1.0, normalization may be fighting large pre-norm values.
            print(f"  qry_norm_mean:      {metrics['qry_norm_mean']:.4f}")
            print(f"  tgt_norm_mean:      {metrics['tgt_norm_mean']:.4f}")

        print("[B3_DEBUG] Dimensionality:")
        # eff_rank: effective rank via spectral entropy of the embedding matrix (RankMe).
        #   Measures how many dimensions are actually used. Lower = more collapsed.
        #   For a 1536-dim model, eff_rank > 800 is healthy; < 100 indicates severe collapse.
        print(f"  eff_rank_qry:       {metrics.get('eff_rank_qry', 'N/A'):.1f} / {D}")
        print(f"  eff_rank_tgt:       {metrics.get('eff_rank_tgt', 'N/A'):.1f} / {D}")
        # participation: (sum(var))^2 / sum(var^2) — how many dims have non-trivial variance.
        #   Lower = fewer dims are active. Complements eff_rank: eff_rank uses singular values,
        #   participation uses per-dimension variance directly.
        print(f"  participation_qry:  {metrics.get('participation_qry', 'N/A'):.1f} / {D}")
        print(f"  participation_tgt:  {metrics.get('participation_tgt', 'N/A'):.1f} / {D}")

        # Anisotropy check: raw (pre-subtraction) mean cosine similarity.
        # If this decreases over training, the model is learning to be less
        # anisotropic on its own, and mean subtraction can eventually be removed.
        # raw_mean_sim: mean pairwise cosine sim of embeddings BEFORE mean subtraction.
        #   High (>0.3) = severe anisotropy (all embeddings point in similar directions).
        #   If this drops below ~0.1, the space is isotropic enough to disable mean_subtraction.
        #   Only printed when model has _last_raw_reps (e.g. Qwen3.5-VL with mean_subtraction).
        inner = self.model.module if hasattr(self.model, "module") else self.model
        raw_reps = getattr(inner, "_last_raw_reps", None)
        if raw_reps is not None and raw_reps.size(0) >= 2:
            raw_sim = raw_reps @ raw_reps.T
            mask = ~torch.eye(raw_sim.size(0), dtype=torch.bool)
            raw_mean_sim = raw_sim[mask].mean().item()
            print(f"\n[B3_DEBUG] Anisotropy (pre-subtraction):")
            print(f"  raw_mean_sim:       {raw_mean_sim:.6f}")
            metrics["raw_mean_sim"] = raw_mean_sim

        # Legacy local-only EGA diagnostics are intentionally disabled.  The
        # actual EGA update uses the cross-rank global batch, so GradCache emits
        # authoritative [EGA_GLOBAL_GRADIENT_OK] metrics after all-gather.
        ega_alpha = getattr(trainer.args, 'ega_alpha', None)
        if ega_alpha is not None and getattr(trainer.args, 'use_ega', False):
            try:
                sim_matrix = qry_reps.float() @ tgt_reps.float().T  # [B, N]
                B_local = qry_reps.size(0)
                tpq_local = tgt_reps.size(0) // B_local
                pos_idx = torch.arange(0, B_local * tpq_local, tpq_local, device=sim_matrix.device)
                pos_sim_vec = sim_matrix[torch.arange(B_local), pos_idx]  # [B]
                # hardness = exp(alpha * (sim_neg - sim_pos)) for negatives only
                neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
                neg_mask[torch.arange(B_local), pos_idx] = False
                hardness = torch.exp(ega_alpha * (sim_matrix - pos_sim_vec.unsqueeze(1)))
                hardness_neg = hardness[neg_mask]

                # Percentiles for full distribution shape
                h_flat = hardness_neg.flatten()
                h_p25 = torch.quantile(h_flat, 0.25).item()
                h_p50 = torch.quantile(h_flat, 0.50).item()
                h_p75 = torch.quantile(h_flat, 0.75).item()
                h_p95 = torch.quantile(h_flat, 0.95).item()

                print(f"\n[B3_DEBUG] EGA Hardness (alpha={ega_alpha}):")
                # hardness = exp(alpha * (sim_neg - sim_pos)) for each negative.
                #   > 1: this negative is harder than the positive (sim closer to query).
                #   < 1: this negative is easier than the positive.
                # EGA amplifies gradients for hard negatives (hardness > 1) and
                # suppresses easy ones. The distribution shape tells us if EGA is
                # hitting the right negatives.
                print(f"  hardness_mean:      {hardness_neg.mean().item():.4f}")
                # Percentiles give the full distribution shape (not just mean/max).
                # p50 = median: half of negatives are above this hardness.
                # p95 = the top 5% hardest negatives — EGA's primary amplification targets.
                # If p95 >> mean, a small fraction of very hard negatives dominate the gradient.
                print(f"  hardness_p25:       {h_p25:.4f}")
                print(f"  hardness_p50:       {h_p50:.4f}")
                print(f"  hardness_p75:       {h_p75:.4f}")
                print(f"  hardness_p95:       {h_p95:.4f}")
                print(f"  hardness_max:       {hardness_neg.max().item():.4f}")
                # hardness_>1_ratio: fraction of negatives harder than the positive.
                #   = pos_neg_overlap (computed via EGA's hardness formula instead of raw sim).
                #   High = many hard negatives → EGA has a lot to amplify (good for learning).
                #   Very low = batch is too easy → EGA has little effect → consider harder batch construction.
                print(f"  hardness_>1_ratio:  {(hardness_neg > 1.0).float().mean().item():.4f}")

                # EGA gradient amplification ratio: ||EGA grad|| / ||standard grad||
                # Standard InfoNCE gradient:  G_std = (softmax(sim/tau) - one_hot) @ tgt / tau
                # EGA gradient:               G_ega = (amplified_probs - one_hot) @ tgt / tau
                tau = self.temperature
                logits = sim_matrix / tau
                probs = torch.softmax(logits, dim=1)  # [B, N]

                # Compute EGA amplified probs (same as ega.py)
                amplified_probs = probs.clone()
                amplified_probs[neg_mask] = probs[neg_mask] * hardness[neg_mask]
                amp_neg = amplified_probs[neg_mask].view(B_local, -1)
                orig_neg = probs[neg_mask].view(B_local, -1)
                amp_sum = amp_neg.sum(dim=1, keepdim=True).clamp(min=1e-12)
                orig_sum = orig_neg.sum(dim=1, keepdim=True)
                norm_neg = amp_neg / amp_sum * orig_sum
                normalized_probs = torch.zeros_like(probs)
                normalized_probs[neg_mask] = norm_neg.view(-1)
                normalized_probs[torch.arange(B_local), pos_idx] = probs[torch.arange(B_local), pos_idx]

                # Gradient matrices: W_std = probs - one_hot, W_ega = normalized_probs - one_hot
                W_std = probs.clone()
                W_std[torch.arange(B_local), pos_idx] -= 1.0
                W_ega = normalized_probs.clone()
                W_ega[torch.arange(B_local), pos_idx] -= 1.0

                # Gradient norms (query side only, [B, D])
                grad_std = torch.matmul(W_std, tgt_reps.float()) / tau  # [B, D]
                grad_ega = torch.matmul(W_ega, tgt_reps.float()) / tau  # [B, D]
                grad_std_norm = grad_std.norm(dim=-1).mean().item()  # mean per-query norm
                grad_ega_norm = grad_ega.norm(dim=-1).mean().item()
                grad_ratio = grad_ega_norm / max(grad_std_norm, 1e-12)

                print(f"\n[B3_DEBUG] EGA Gradient Amplification:")
                # grad_std_norm: mean per-query L2 norm of standard InfoNCE gradients.
                #   G_std = (softmax(sim/tau) - one_hot) @ tgt / tau
                #   This is what the model would learn WITHOUT EGA.
                print(f"  grad_std_norm:      {grad_std_norm:.6f}")
                # grad_ega_norm: mean per-query L2 norm of EGA-amplified gradients.
                #   G_ega = (amplified_probs - one_hot) @ tgt / tau
                #   This is what the model actually learns WITH EGA.
                print(f"  grad_ega_norm:      {grad_ega_norm:.6f}")
                # grad_ratio: ||G_ega|| / ||G_std|| — the actual gradient amplification factor.
                #   > 1: EGA amplifies gradients (hard negatives contribute more).
                #   ~ 1: EGA has little effect (batch is too easy or alpha is too low).
                #   < 1: EGA suppresses gradients (unusual — check if alpha is negative or
                #        batch has very few hard negatives).
                #   Rule of thumb: 1.2–2.0 is healthy. > 3.0 risks instability.
                print(f"  grad_ratio:         {grad_ratio:.4f}")

                metrics["ega_hardness_mean"] = hardness_neg.mean().item()
                metrics["ega_hardness_p95"] = h_p95
                metrics["ega_grad_ratio"] = grad_ratio
            except Exception as e:
                print(f"[B3_DEBUG] EGA hardness computation failed: {e}")

        # Log to wandb if available
        try:
            if wandb.run is not None:
                wandb.log(
                    {f"embed/{k}": v for k, v in metrics.items()},
                    step=step,
                )
        except Exception:
            pass

        # Clear reps to free GPU memory
        trainer._monitor_reps = None


def main():
    print(">>> main() entered", flush=True)
    if transformers.__version__ != "5.9.0":
        raise RuntimeError(
            "This B3 reproduction is pinned to transformers==5.9.0, but "
            f"the active environment has transformers=={transformers.__version__}. "
            "Run it from the qwen3_vl_embedding environment."
        )
    print(
        f">>> main(): transformers version verified: {transformers.__version__}",
        flush=True,
    )
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    print(">>> main(): creating HfArgumentParser", flush=True)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    print(">>> main(): parsing args", flush=True)
    try:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    except SystemExit as e:
        print(f">>> main(): SystemExit caught during parse_args! code={e.code}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    except Exception as e:
        print(f">>> main(): Exception during parse_args: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    print(">>> main(): args parsed OK", flush=True)
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if data_args.negative_strategy == "unime_v2":
        isolated_components = {
            "dataset_loader": init_concat_dataset,
            "collator": MultimodalDataCollator,
            "trainer": GradCacheLateProcessTrainer,
        }
        wrong_components = {
            name: f"{component.__module__}.{component.__name__}"
            for name, component in isolated_components.items()
            if component.__module__ != "src.training.unime_v2"
        }
        if wrong_components:
            raise RuntimeError(
                "UniME-V2 must be launched through train_unime_v2.py so its "
                "dataset expansion, collator and loss cannot be silently replaced: "
                f"{wrong_components}"
            )

    if model_args.qwen2_eval_token_pruning:
        raise ValueError(
            "--qwen2_eval_token_pruning is evaluation-only. Use "
            "eval_all_qwen2_token_pruning.sh; training behavior must remain unchanged."
        )

    if model_args.qwen2_embedding_token:
        embedding_token_errors = []
        if data_args.negative_strategy != "vlm2vec_inbatch":
            embedding_token_errors.append(
                "negative_strategy must be vlm2vec_inbatch"
            )
        if model_args.qwen2_model_impl != "b3":
            embedding_token_errors.append("qwen2_model_impl must be b3")
        if model_args.pooling not in {"last", "eos"}:
            embedding_token_errors.append("pooling must be last/eos")
        if not model_args.lora:
            embedding_token_errors.append("lora must be enabled")
        if not model_args.qwen2_embedding_token_text:
            embedding_token_errors.append("embedding token text must be non-empty")
        if embedding_token_errors:
            raise ValueError(
                "Qwen2 In-batch embedding-token contract failed: "
                + "; ".join(embedding_token_errors)
            )


    # DEBUG PRINTS for Distributed Setup
    print("Distributed init debug info:", flush=True)
    print(f"RANK: {os.environ.get('RANK')}", flush=True)
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}", flush=True)
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}", flush=True)
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}", flush=True)
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}", flush=True)

    if torch.distributed.is_available():
        print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
            print(f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")


    # Check for existing checkpoints
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        resume_checkpoint_dir = None
        logger.info("No checkpoint found. Starting fresh training.")

    from src.training.qwen3_5_prompt import (
        EMBEDDING_V2_PROFILE,
        validate_checkpoint_config as validate_qwen3_5_prompt_checkpoint,
        validate_profile as validate_qwen3_5_prompt_profile,
    )

    validate_qwen3_5_prompt_profile(model_args.qwen3_5_prompt_profile)
    if model_args.qwen3_5_prompt_profile != "legacy":
        prompt_contract_errors = []
        if model_args.qwen3_5_prompt_profile != EMBEDDING_V2_PROFILE:
            prompt_contract_errors.append("profile must be embedding_v2")
        if data_args.negative_strategy not in {"vlm2vec_inbatch", "b3"}:
            prompt_contract_errors.append(
                "negative_strategy must be vlm2vec_inbatch or b3"
            )
        if model_args.pooling not in {"last", "eos"}:
            prompt_contract_errors.append("pooling must be last/eos")
        if prompt_contract_errors:
            raise ValueError(
                "Qwen3.5 prompt-v2 experiment contract failed: "
                + "; ".join(prompt_contract_errors)
            )
        print_master(
            "[QWEN3_5_PROMPT_PROFILE_OK] "
            f"profile={model_args.qwen3_5_prompt_profile} "
            "task_role_aware=True pooling=final_active_token"
        )

    if resume_checkpoint_dir is not None and data_args.b3_ablation != "none":
        ablation_checkpoint_config_path = os.path.join(
            resume_checkpoint_dir, "b3_ablation_config.json"
        )
        if not os.path.isfile(ablation_checkpoint_config_path):
            raise RuntimeError(
                "Refusing to resume a B3 ablation from a checkpoint without "
                f"b3_ablation_config.json: {resume_checkpoint_dir}"
            )
        with open(
            ablation_checkpoint_config_path, "r", encoding="utf-8"
        ) as handle:
            checkpoint_ablation_config = json.load(handle)
        expected_ablation_config = {
            "method": "qwen2_vl_b3_ablation",
            "ablation": data_args.b3_ablation,
            "target_per_query": (
                1 if data_args.b3_ablation == "no_instance_hn" else 6
            ),
        }
        if checkpoint_ablation_config != expected_ablation_config:
            raise RuntimeError(
                "B3 ablation resume configuration differs from the checkpoint: "
                f"checkpoint={checkpoint_ablation_config} "
                f"current={expected_ablation_config}"
            )
        print_master(
            "[B3_ABLATION_RESUME_CONFIG_OK] "
            f"checkpoint={resume_checkpoint_dir} "
            f"values={checkpoint_ablation_config}"
        )

    if (
        resume_checkpoint_dir is not None
        and training_args.embedding_gradient_strategy == "ega"
    ):
        ega_checkpoint_config_path = os.path.join(
            resume_checkpoint_dir, "ega_training_config.json"
        )
        if not os.path.isfile(ega_checkpoint_config_path):
            raise RuntimeError(
                "Refusing to resume EGA from a checkpoint without "
                f"ega_training_config.json: {resume_checkpoint_dir}"
            )
        with open(ega_checkpoint_config_path, "r", encoding="utf-8") as handle:
            checkpoint_ega_config = json.load(handle)
        expected_ega_config = {
            "method": "ega",
            "alpha": float(training_args.ega_alpha),
            "temperature": float(model_args.temperature),
            "negative_strategy": data_args.negative_strategy,
            "target_per_query": 1,
        }
        if checkpoint_ega_config != expected_ega_config:
            raise RuntimeError(
                "EGA resume configuration differs from the checkpoint: "
                f"checkpoint={checkpoint_ega_config} current={expected_ega_config}"
            )
        print_master(
            "[EGA_RESUME_CONFIG_OK] "
            f"checkpoint={resume_checkpoint_dir} values={checkpoint_ega_config}"
        )
    if (
        resume_checkpoint_dir is not None
        and training_args.embedding_gradient_strategy == "b3_ega"
    ):
        b3_ega_checkpoint_config_path = os.path.join(
            resume_checkpoint_dir, "b3_ega_training_config.json"
        )
        if not os.path.isfile(b3_ega_checkpoint_config_path):
            raise RuntimeError(
                "Refusing to resume B3+EGA from a checkpoint without "
                f"b3_ega_training_config.json: {resume_checkpoint_dir}"
            )
        with open(
            b3_ega_checkpoint_config_path, "r", encoding="utf-8"
        ) as handle:
            checkpoint_b3_ega_config = json.load(handle)
        expected_b3_ega_config = {
            "method": "b3_ega",
            "alpha": float(training_args.ega_alpha),
            "temperature": float(model_args.temperature),
            "negative_strategy": data_args.negative_strategy,
            "target_per_query": 6,
            "explicit_hard_negatives_per_query": 5,
            "negative_scope": "all_global_targets_except_aligned_positive",
            "preserve_negative_mass": True,
        }
        if checkpoint_b3_ega_config != expected_b3_ega_config:
            raise RuntimeError(
                "B3+EGA resume configuration differs from the checkpoint: "
                f"checkpoint={checkpoint_b3_ega_config} "
                f"current={expected_b3_ega_config}"
            )
        print_master(
            "[B3_EGA_RESUME_CONFIG_OK] "
            f"checkpoint={resume_checkpoint_dir} "
            f"values={checkpoint_b3_ega_config}"
        )
    if (
        resume_checkpoint_dir is not None
        and model_args.moe_lora
        and data_args.negative_strategy == "b3"
    ):
        b3_moe_checkpoint_config_path = os.path.join(
            resume_checkpoint_dir, "b3_moe_training_config.json"
        )
        if not os.path.isfile(b3_moe_checkpoint_config_path):
            raise RuntimeError(
                "Refusing to resume B3+MoE from a checkpoint without "
                f"b3_moe_training_config.json: {resume_checkpoint_dir}"
            )
        with open(
            b3_moe_checkpoint_config_path, "r", encoding="utf-8"
        ) as handle:
            checkpoint_b3_moe_config = json.load(handle)
        expected_b3_moe_config = {
            "method": "b3_tsembed_moe_dora_only",
            "negative_strategy": "b3",
            "embedding_gradient_strategy": "info_nce",
            "target_per_query": 6,
            "explicit_hard_negatives_per_query": 5,
            "num_experts": int(model_args.moe_num_experts),
            "rank_per_expert": int(model_args.lora_r),
            "alpha_per_expert": float(model_args.lora_alpha),
            "router_temperature": float(model_args.moe_router_temperature),
            "eans": False,
            "two_stage_training": False,
        }
        if checkpoint_b3_moe_config != expected_b3_moe_config:
            raise RuntimeError(
                "B3+MoE resume configuration differs from the checkpoint: "
                f"checkpoint={checkpoint_b3_moe_config} "
                f"current={expected_b3_moe_config}"
            )
        print_master(
            "[B3_MOE_RESUME_CONFIG_OK] "
            f"checkpoint={resume_checkpoint_dir} "
            f"values={checkpoint_b3_moe_config}"
        )
    if (
        resume_checkpoint_dir is not None
        and training_args.embedding_gradient_strategy == "llave"
    ):
        llave_checkpoint_config_path = os.path.join(
            resume_checkpoint_dir, "llave_training_config.json"
        )
        if not os.path.isfile(llave_checkpoint_config_path):
            raise RuntimeError(
                "Refusing to resume LLaVE from a checkpoint without "
                f"llave_training_config.json: {resume_checkpoint_dir}"
            )
        with open(llave_checkpoint_config_path, "r", encoding="utf-8") as handle:
            checkpoint_llave_config = json.load(handle)
        expected_llave_config = {
            "method": "llave_hardness_weighted_contrastive_learning",
            "alpha": float(training_args.llave_alpha),
            "temperature": float(model_args.temperature),
            "negative_strategy": data_args.negative_strategy,
            "target_per_query": 1,
            "reward_model": "policy_model_synchronized_each_step",
            "reward_gradient": "stopped",
            "reward_scope": "negative_pairs_only",
            "directions": ["query_to_target", "target_to_query"],
            "direction_reduction": "mean",
            "cross_device_negatives": bool(
                int(os.environ.get("WORLD_SIZE", "1")) > 1
            ),
        }
        if checkpoint_llave_config != expected_llave_config:
            raise RuntimeError(
                "LLaVE resume configuration differs from the checkpoint: "
                f"checkpoint={checkpoint_llave_config} current={expected_llave_config}"
            )
        print_master(
            "[LLAVE_RESUME_CONFIG_OK] "
            f"checkpoint={resume_checkpoint_dir} values={checkpoint_llave_config}"
        )

    # Initialize WandB if enabled
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            wandb.login(key="37464c29a08af81181e7450dc78367e94bd2a95c")
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="disabled")
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    model = MMEBModel.build(model_args)
    log_runtime_and_model_state(model)
    install_model_hooks(model)
    model_backbone = get_backbone_name(hf_config=model.config)
    if model_args.qwen3_5_prompt_profile != "legacy" and model_backbone != "qwen3_5_vl":
        raise ValueError(
            "The role-aware prompt experiment supports only Qwen3.5-VL"
        )
    if model_backbone == "qwen3_5_vl" and resume_checkpoint_dir is not None:
        validate_qwen3_5_prompt_checkpoint(
            resume_checkpoint_dir,
            model_args.qwen3_5_prompt_profile,
        )
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    if model_args.qwen2_embedding_token and model_backbone != "qwen2_vl":
        raise ValueError(
            "--qwen2_embedding_token requires a Qwen2-VL backbone, got "
            f"{model_backbone!r}"
        )

    if model_args.moe_lora:
        from src.training.moe_lora import validate_moe_training_profile

        moe_training_profile = validate_moe_training_profile(
            model_backbone=model_backbone,
            lora_enabled=model_args.lora,
            negative_strategy=data_args.negative_strategy,
            embedding_gradient_strategy=(
                training_args.embedding_gradient_strategy
            ),
            pos_only=data_args.pos_only,
            grad_cache=training_args.grad_cache,
            sdibn=data_args.sdibn,
            odibn=data_args.odibn,
            rdibn=data_args.rdibn,
        )
        print_master(
            "[TSEMBED_MOE_TRAINING_CONTRACT_OK] "
            f"profile={moe_training_profile} backbone={model_backbone} "
            f"negative_strategy={data_args.negative_strategy} "
            "objective=info_nce eans=False two_stage=False "
            "routing=dense_softmax_per_token "
            f"experts={model_args.moe_num_experts} "
            f"rank_per_expert={model_args.lora_r} "
            f"alpha={model_args.lora_alpha} "
            f"dropout={model_args.lora_dropout:.8g} "
            f"router_temperature={model_args.moe_router_temperature:.8g}"
        )

    if model_args.qwen3_5_bidirectional_attention:
        from src.training.qwen3_5_bidirectional import (
            validate_bidirectional_mean_training_profile,
        )

        validate_bidirectional_mean_training_profile(
            model_backbone=model_backbone,
            enabled=True,
            pooling=model_args.pooling,
            lora=model_args.lora,
            moe_lora=model_args.moe_lora,
            negative_strategy=data_args.negative_strategy,
            embedding_gradient_strategy=(
                training_args.embedding_gradient_strategy
            ),
            pos_only=data_args.pos_only,
            grad_cache=training_args.grad_cache,
        )
        print_master(
            "[QWEN3_5_BIDIRECTIONAL_MEAN_TRAINING_CONTRACT_OK] "
            "backbone=qwen3_5_vl negative_strategy=vlm2vec_inbatch "
            "objective=info_nce moe=False pooling=mean_all_active_tokens "
            "full_attention=bidirectional gated_deltanet=causal"
        )
        if resume_checkpoint_dir is not None:
            from src.training.qwen3_5_bidirectional import (
                read_checkpoint_config,
            )

            checkpoint_config = read_checkpoint_config(resume_checkpoint_dir)
            runtime_config = getattr(
                model_args,
                "qwen3_5_bidirectional_runtime_config",
                None,
            )
            if checkpoint_config != runtime_config:
                raise RuntimeError(
                    "Qwen3.5 bidirectional-mean resume configuration differs "
                    f"from the checkpoint: checkpoint={checkpoint_config} "
                    f"current={runtime_config}"
                )
            print_master(
                "[QWEN3_5_BIDIRECTIONAL_MEAN_RESUME_CONFIG_OK] "
                f"checkpoint={resume_checkpoint_dir} values={checkpoint_config}"
            )

    # B3 debug: log model structure, LoRA modules, and parameter counts
    if _B3_DEBUG:
        _b3_log_model_structure(model)
        # Strategy-specific config. The default info_nce branch remains the
        # historical path.
        gradient_strategy = getattr(
            training_args, 'embedding_gradient_strategy', 'info_nce'
        )
        use_ega = gradient_strategy in {"ega", "b3_ega"}
        ega_alpha = getattr(training_args, 'ega_alpha', 20.0)
        print(f"[B3_DEBUG] Embedding gradient strategy: {gradient_strategy}")
        if use_ega:
            print(f"[B3_DEBUG] EGA alpha: {ega_alpha}")
        if gradient_strategy == "llave":
            print(f"[B3_DEBUG] LLaVE alpha: {training_args.llave_alpha}")
        print("=" * 80 + "\n")

    processor = load_processor(model_args, data_args)
    setattr(model, 'model_backbone', model_backbone)
    setattr(model, 'processor', processor)
    if model_args.qwen2_embedding_token:
        processor_token_id = getattr(
            processor, "_qwen2_embedding_token_id", None
        )
        if processor_token_id != model.qwen2_embedding_token_id:
            raise RuntimeError(
                "Qwen2 model/processor embedding-token mismatch: "
                f"model={model.qwen2_embedding_token_id}, "
                f"processor={processor_token_id}"
            )
        if resume_checkpoint_dir is not None:
            validate_qwen2_embedding_token_checkpoint(
                resume_checkpoint_dir,
                model_args,
                processor_token_id,
            )

    with open(data_args.dataset_config, 'rb') as yaml_file:
        dataset_config_bytes = yaml_file.read()
        dataset_config = yaml.safe_load(dataset_config_bytes.decode("utf-8"))
        dataset_config_sha256 = hashlib.sha256(dataset_config_bytes).hexdigest()
        train_dataset, dataset_lens = init_concat_dataset(dataset_config, model_args, data_args, training_args)
    train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args)

    valid_negative_strategies = {"b3", "vlm2vec_inbatch", "unime_v2"}
    if data_args.negative_strategy not in valid_negative_strategies:
        raise ValueError(
            f"Unknown --negative_strategy={data_args.negative_strategy!r}; "
            f"expected one of {sorted(valid_negative_strategies)}"
        )
    valid_b3_ablations = {
        "none",
        "random_batch",
        "no_fn_filter",
        "no_instance_hn",
    }
    if data_args.b3_ablation not in valid_b3_ablations:
        raise ValueError(
            f"Unknown --b3_ablation={data_args.b3_ablation!r}; expected one "
            f"of {sorted(valid_b3_ablations)}"
        )
    if data_args.b3_ablation != "none":
        ablation_errors = []
        if model_backbone != "qwen2_vl":
            ablation_errors.append(
                f"backbone must be qwen2_vl, found {model_backbone!r}"
            )
        if data_args.negative_strategy != "b3":
            ablation_errors.append("negative_strategy must be b3")
        if training_args.embedding_gradient_strategy != "info_nce":
            ablation_errors.append("embedding_gradient_strategy must be info_nce")
        if not training_args.grad_cache:
            ablation_errors.append("grad_cache must be True")
        if not model_args.normalize:
            ablation_errors.append("normalized embeddings are required")
        if model_args.qwen2_model_impl != "b3":
            ablation_errors.append("qwen2_model_impl must be b3")
        if model_args.qwen2_b3_attention_impl != "flash_attention_2":
            ablation_errors.append("qwen2_b3_attention_impl must be flash_attention_2")
        if model_args.qwen2_b3_legacy_position_ids:
            ablation_errors.append("qwen2_b3_legacy_position_ids must be False")
        if data_args.chunk_size != 32:
            ablation_errors.append("chunk_size must remain 32")

        if data_args.b3_ablation == "random_batch":
            if not data_args.sdibn or data_args.odibn or data_args.rdibn:
                ablation_errors.append(
                    "random_batch requires sdibn=True, odibn=False, rdibn=False"
                )
            if data_args.pos_only:
                ablation_errors.append("random_batch must retain five instance HNs")
            expected_target_per_query = 6
            batch_contract = "same_dataset_random"
            fn_contract = "enabled"
            instance_hn_contract = 5
        elif data_args.b3_ablation == "no_fn_filter":
            if not data_args.sdibn or not data_args.odibn or data_args.rdibn:
                ablation_errors.append(
                    "no_fn_filter requires the original sdibn+odibn sampler"
                )
            if data_args.pos_only:
                ablation_errors.append("no_fn_filter must retain five instance HNs")
            if "no_fn" not in str(data_args.dataset_config).lower():
                ablation_errors.append(
                    "no_fn_filter requires an isolated no-FN dataset config"
                )
            expected_target_per_query = 6
            batch_contract = "b3_metis_ordered"
            fn_contract = "disabled_prefix_and_class_label"
            instance_hn_contract = 5
        else:
            if not data_args.sdibn or not data_args.odibn or data_args.rdibn:
                ablation_errors.append(
                    "no_instance_hn requires the original sdibn+odibn sampler"
                )
            if not data_args.pos_only:
                ablation_errors.append("no_instance_hn requires pos_only=True")
            expected_target_per_query = 1
            batch_contract = "b3_metis_ordered"
            fn_contract = "enabled"
            instance_hn_contract = 0

        if ablation_errors:
            raise ValueError(
                "Qwen2-VL B3 ablation contract failed: "
                + "; ".join(ablation_errors)
            )
        print_master(
            "[B3_ABLATION_CONTRACT_OK] "
            f"ablation={data_args.b3_ablation} "
            f"batch={batch_contract} fn_filter={fn_contract} "
            f"explicit_instance_hn={instance_hn_contract} "
            f"target_per_query={expected_target_per_query} "
            f"dataset_config_sha256={dataset_config_sha256}"
        )
    valid_gradient_strategies = {"info_nce", "ega", "b3_ega", "llave"}
    if training_args.embedding_gradient_strategy not in valid_gradient_strategies:
        raise ValueError(
            "Unknown --embedding_gradient_strategy="
            f"{training_args.embedding_gradient_strategy!r}; expected one of "
            f"{sorted(valid_gradient_strategies)}"
        )
    if training_args.embedding_gradient_strategy == "ega":
        supported_ega_backbones = {"qwen2_vl", "qwen3_5_vl"}
        if model_backbone not in supported_ega_backbones:
            raise ValueError(
                "The isolated EGA reproduction supports only "
                f"{sorted(supported_ega_backbones)}, but detected "
                f"backbone={model_backbone!r}"
            )
        if data_args.negative_strategy != "vlm2vec_inbatch":
            raise ValueError(
                "EGA must use --negative_strategy vlm2vec_inbatch; it must not "
                "be combined with B3 mined or clustered hard negatives"
            )
        if not training_args.grad_cache:
            raise ValueError("EGA requires --grad_cache True")
        if not model_args.normalize:
            raise ValueError("EGA requires normalized query/target embeddings")
        if not math.isfinite(training_args.ega_alpha) or training_args.ega_alpha < 0:
            raise ValueError(
                "--ega_alpha must be finite and non-negative, got "
                f"{training_args.ega_alpha}"
            )
        if not math.isfinite(model.temperature) or model.temperature <= 0:
            raise ValueError(
                "EGA temperature must be finite and positive, got "
                f"{model.temperature}"
            )
        print_master(
            "[EGA_TRAINING_CONTRACT_OK] method=explicit_embedding_gradient "
            f"backbone={model_backbone} "
            f"alpha={training_args.ega_alpha:.8g} "
            "negative_strategy=vlm2vec_inbatch target_per_query=1 "
            "global_cross_device_negatives=True normalized_embeddings=True"
        )
    if training_args.embedding_gradient_strategy == "b3_ega":
        contract_errors = []
        supported_b3_ega_backbones = {"qwen2_vl", "qwen3_5_vl"}
        if model_backbone not in supported_b3_ega_backbones:
            contract_errors.append(
                "backbone must be one of "
                f"{sorted(supported_b3_ega_backbones)}, found {model_backbone!r}"
            )
        if data_args.negative_strategy != "b3":
            contract_errors.append(
                f"negative_strategy must be b3, found {data_args.negative_strategy!r}"
            )
        if not data_args.sdibn or not data_args.odibn or data_args.rdibn:
            contract_errors.append(
                "sampler must use sdibn=True, odibn=True, rdibn=False"
            )
        if data_args.pos_only:
            contract_errors.append("pos_only must be False")
        if not training_args.grad_cache:
            contract_errors.append("grad_cache must be True")
        if not model_args.normalize:
            contract_errors.append("normalized embeddings are required")
        if model_backbone == "qwen2_vl":
            if model_args.qwen2_model_impl != "b3":
                contract_errors.append("qwen2_model_impl must be b3")
            if model_args.qwen2_b3_attention_impl != "flash_attention_2":
                contract_errors.append(
                    "qwen2_b3_attention_impl must be flash_attention_2"
                )
            if model_args.qwen2_b3_legacy_position_ids:
                contract_errors.append(
                    "qwen2_b3_legacy_position_ids must be False"
                )
        if not math.isfinite(training_args.ega_alpha) or training_args.ega_alpha < 0:
            contract_errors.append(
                f"ega_alpha must be finite and non-negative, got {training_args.ega_alpha}"
            )
        if not math.isfinite(model.temperature) or model.temperature <= 0:
            contract_errors.append(
                f"temperature must be finite and positive, got {model.temperature}"
            )
        if contract_errors:
            raise ValueError(
                "B3+EGA reproduction contract failed: "
                + "; ".join(contract_errors)
            )
        checkpoint_runtime = (
            "b3_strict_reload"
            if model_backbone == "qwen2_vl"
            else "transformers_native_checkpoint_verified"
        )
        print_master(
            "[B3_EGA_TRAINING_CONTRACT_OK] "
            f"backbone={model_backbone} checkpoint_runtime={checkpoint_runtime} "
            "negative_strategy=b3 sampler=sdibn+odibn target_per_query=6 "
            "explicit_hard_negatives_per_query=5 "
            "negative_scope=all_global_targets_except_aligned_positive "
            f"alpha={training_args.ega_alpha:.8g} "
            f"temperature={model.temperature:.8g} "
            "preserve_negative_mass=True probability_dtype=float32 "
            "global_cross_device_negatives=True normalized_embeddings=True"
        )
    if training_args.embedding_gradient_strategy == "llave":
        if model_backbone != "qwen2_vl":
            raise ValueError(
                "The isolated LLaVE reproduction currently supports only "
                f"Qwen2-VL, but detected backbone={model_backbone!r}"
            )
        if data_args.negative_strategy != "vlm2vec_inbatch":
            raise ValueError(
                "LLaVE must use --negative_strategy vlm2vec_inbatch; the "
                "paper weights ordinary cross-device in-batch negatives and "
                "does not use B3 mined/clustered targets"
            )
        if not training_args.grad_cache:
            raise ValueError("The isolated LLaVE Qwen2-VL run requires --grad_cache True")
        if not model_args.normalize:
            raise ValueError("LLaVE requires normalized cosine embeddings")
        if not math.isfinite(training_args.llave_alpha) or training_args.llave_alpha < 0:
            raise ValueError(
                "--llave_alpha must be finite and non-negative, got "
                f"{training_args.llave_alpha}"
            )
        if not math.isfinite(model.temperature) or model.temperature <= 0:
            raise ValueError(
                "LLaVE temperature must be finite and positive, got "
                f"{model.temperature}"
            )
        print_master(
            "[LLAVE_TRAINING_CONTRACT_OK] "
            "method=hardness_weighted_contrastive_learning "
            f"backbone={model_backbone} alpha={training_args.llave_alpha:.8g} "
            f"temperature={model.temperature:.8g} "
            "policy_score=cosine/temperature "
            "negative_reward=alpha*stop_gradient(cosine) "
            "positive_reward=zero reward_model=policy_synchronized_each_step "
            "directions=query_to_target+target_to_query direction_reduction=mean "
            "negative_strategy=vlm2vec_inbatch target_per_query=1 "
            "global_cross_device_negatives=True normalized_embeddings=True"
        )
    if data_args.negative_strategy == "unime_v2":
        contract_errors = []
        supported_unime_backbones = {"qwen2_vl", "qwen3_5_vl"}
        if model_backbone not in supported_unime_backbones:
            contract_errors.append(
                "backbone must be one of "
                f"{sorted(supported_unime_backbones)}, found {model_backbone!r}"
            )
        if model_backbone == "qwen2_vl":
            if model_args.qwen2_model_impl != "b3":
                contract_errors.append("qwen2_model_impl must be b3")
            if model_args.qwen2_b3_attention_impl != "flash_attention_2":
                contract_errors.append(
                    "qwen2_b3_attention_impl must be flash_attention_2"
                )
            if model_args.qwen2_b3_legacy_position_ids:
                contract_errors.append(
                    "qwen2_b3_legacy_position_ids must be False"
                )
        if not model_args.lora:
            contract_errors.append("LoRA/DoRA must be enabled")
        if not model_args.normalize:
            contract_errors.append("normalized embeddings are required")
        if not training_args.grad_cache:
            contract_errors.append("GradCache must be enabled")
        if training_args.embedding_gradient_strategy != "info_nce":
            contract_errors.append(
                "embedding_gradient_strategy must remain info_nce; the dedicated "
                "trainer replaces InfoNCE with UniME-V2 symmetric KL"
            )
        if data_args.unime_v2_num_hard_negatives != 8:
            contract_errors.append("unime_v2_num_hard_negatives must be exactly 8")
        if not data_args.unime_v2_data_path:
            contract_errors.append("unime_v2_data_path is required")
        elif os.path.basename(data_args.unime_v2_data_path) != (
            "train_data_Qwen25VL_7B_scores.json"
        ):
            contract_errors.append(
                "teacher-score file must be train_data_Qwen25VL_7B_scores.json"
            )
        if not data_args.unime_v2_image_dir:
            contract_errors.append("unime_v2_image_dir is required")
        if data_args.pos_only:
            contract_errors.append("pos_only must be False")
        if data_args.rdibn or data_args.sdibn or data_args.odibn:
            contract_errors.append("B3 rdibn/sdibn/odibn must all be disabled")
        if contract_errors:
            raise ValueError(
                "UniME-V2 reproduction contract failed: "
                + "; ".join(contract_errors)
            )
        if model_backbone == "qwen2_vl":
            _validate_qwen2_vlm2vec_adapter(model, model_args)
            checkpoint_runtime = "b3_strict_reload"
        else:
            # Qwen3.5 checkpoint coverage and its architecture-aware LoRA/DoRA
            # mapping are validated inside MMEBModel.build().
            checkpoint_runtime = "transformers_native_5.9_verified_reload"
        print_master(
            "[UNIME_V2_TRAINING_CONTRACT_OK] "
            f"backbone={model_backbone} checkpoint_runtime={checkpoint_runtime} "
            "official_data_expansion=5 original_query_plus_even_hard_negatives=True "
            "positive_plus_odd_hard_negatives=True hard_negatives=8 "
            "candidate_order=positive_then_hn0_to_hn7 "
            "objective=symmetric_kl teacher=Qwen2.5-VL-7B "
            f"temperature={model.temperature:.8g} normalized_embeddings=True "
            "ordinary_inbatch_negatives=False "
            "length_aware_gradcache_packing=True "
            "restore_official_order_before_loss=True"
        )

    if data_args.negative_strategy == "vlm2vec_inbatch":
        supported_inbatch_backbones = {"qwen2_vl", "qwen3_5_vl"}
        if model_backbone not in supported_inbatch_backbones:
            raise ValueError(
                "The isolated VLM2Vec in-batch reproduction supports only "
                f"{sorted(supported_inbatch_backbones)}, but detected "
                f"backbone={model_backbone!r}"
            )
        if model_backbone == "qwen2_vl":
            qwen2_contract_errors = []
            if model_args.qwen2_model_impl != "b3":
                qwen2_contract_errors.append(
                    "qwen2_model_impl must be 'b3' (the Transformers-native "
                    "5.9 implementation is not the reproduced VLM2Vec/B3-era "
                    "Qwen2-VL runtime)"
                )
            if model_args.qwen2_b3_attention_impl != "flash_attention_2":
                qwen2_contract_errors.append(
                    "qwen2_b3_attention_impl must be 'flash_attention_2'"
                )
            if model_args.qwen2_b3_legacy_position_ids:
                qwen2_contract_errors.append(
                    "qwen2_b3_legacy_position_ids must be False under the "
                    "Transformers 5.9 replay-safe baseline"
                )
            if qwen2_contract_errors:
                raise ValueError(
                    "Qwen2-VL VLM2Vec in-batch reproduction contract failed: "
                    + "; ".join(qwen2_contract_errors)
                )
            print_master(
                "[VLM2VEC_INBATCH_QWEN2_CONTRACT_OK] "
                "model_impl=b3 attention=flash_attention_2 "
                "legacy_position_ids=False strict_checkpoint_reload=True "
                "rope_reset=True"
            )
            print_master(
                "[VLM2VEC_INBATCH_QWEN2_PROFILE] "
                "negative_and_data_strategy=vlm2vec_public "
                "optimization_profile=qwen2_b3_control "
                f"adapter_profile={'dense_moe_dora' if model_args.moe_lora else 'single_dora'}"
            )
            if model_args.moe_lora:
                _validate_qwen2_vlm2vec_moe_adapter(model, model_args)
            else:
                _validate_qwen2_vlm2vec_adapter(model, model_args)
            if training_args.embedding_gradient_strategy == "ega":
                print_master(
                    "[EGA_QWEN2_COMBINED_CONTRACT_OK] "
                    "checkpoint_runtime=b3_strict_reload "
                    "pairs=vlm2vec_public_one_positive "
                    "gradient=ega_global_cross_device"
                )
            if training_args.embedding_gradient_strategy == "llave":
                print_master(
                    "[LLAVE_QWEN2_COMBINED_CONTRACT_OK] "
                    "checkpoint_runtime=b3_strict_reload "
                    "pairs=vlm2vec_public_one_positive "
                    "objective=official_llave_symmetric_hardness_weighted "
                    "directions=query_to_target+target_to_query "
                    "global_cross_device_negatives=True"
                )
        expected_vlm2vec_sources = {
            "ImageNet_1K": 100000,
            "N24News": 50000,
            "HatefulMemes": 10000,
            "VOC2007": 10000,
            "SUN397": 20000,
            "OK-VQA": 10000,
            "A-OKVQA": 20000,
            "DocVQA": 40000,
            "InfographicsVQA": 25000,
            "ChartQA": 28000,
            "Visual7W": 70000,
            "VisDial": 130000,
            "CIRR": 30000,
            "VisualNews_t2i": 100000,
            "VisualNews_i2t": 100000,
            "MSCOCO_t2i": 100000,
            "MSCOCO_i2t": 120000,
            "NIGHTS": 20000,
            "WebQA": 20000,
            "MSCOCO": 100000,
        }
        configured_sources = list(dataset_config.keys())
        if configured_sources != list(expected_vlm2vec_sources):
            raise ValueError(
                "VLM2Vec image-only source names/order differ from the public "
                f"recipe: configured={configured_sources}, "
                f"expected={list(expected_vlm2vec_sources)}"
            )
        configured_caps = {
            name: int(config["num_sample_per_subset"])
            for name, config in dataset_config.items()
        }
        if configured_caps != expected_vlm2vec_sources:
            raise ValueError(
                "VLM2Vec image-only per-source caps differ from the public "
                f"recipe: configured={configured_caps}, "
                f"expected={expected_vlm2vec_sources}"
            )
        if model_backbone == "qwen2_vl":
            local_mirror_errors = []
            for public_name, config in dataset_config.items():
                if config.get("dataset_parser") != "mmeb":
                    local_mirror_errors.append(
                        f"{public_name}: dataset_parser={config.get('dataset_parser')!r}"
                    )
                if config.get("subset_name") != public_name:
                    local_mirror_errors.append(
                        f"{public_name}: subset_name={config.get('subset_name')!r}"
                    )
                # The offline MMEB-train mirror stores the public recipe's
                # logical `original` split as train-*.parquet files. This is a
                # storage-name adaptation, not permission to switch corpora.
                if config.get("dataset_split") != "train":
                    local_mirror_errors.append(
                        f"{public_name}: dataset_split={config.get('dataset_split')!r}"
                    )
                dataset_root = os.path.basename(
                    os.path.normpath(str(config.get("dataset_name", "")))
                )
                if dataset_root != "MMEB-train":
                    local_mirror_errors.append(
                        f"{public_name}: dataset_name={config.get('dataset_name')!r}"
                    )
            if local_mirror_errors:
                raise ValueError(
                    "Qwen2-VL VLM2Vec local MMEB mirror contract failed: "
                    + "; ".join(local_mirror_errors[:8])
                )
            print_master(
                "[VLM2VEC_INBATCH_QWEN2_DATA_MIRROR_OK] "
                "public_logical_split=original local_physical_split=train "
                "dataset_root=MMEB-train subset_names_exact=True"
            )
        if not data_args.pos_only:
            raise ValueError(
                "vlm2vec_inbatch requires --pos_only True so every query has "
                "exactly one positive target"
            )
        if data_args.rdibn or data_args.sdibn or data_args.odibn:
            raise ValueError(
                "vlm2vec_inbatch must not be combined with B3 rdibn/sdibn/odibn"
            )
        mined_or_clustered_subsets = [
            config.get("subset_name", "")
            for config in dataset_config.values()
            if any(
                marker in config.get("subset_name", "").lower()
                for marker in ("hnps", "hnips", "hnrand", "metis")
            )
        ]
        if mined_or_clustered_subsets:
            raise ValueError(
                "vlm2vec_inbatch must use the original MMEB pairs, not B3 "
                "mined/clustered subsets. Offending subset: "
                f"{mined_or_clustered_subsets[0]}"
            )
        print_master(
            "[TRAINING_STRATEGY] vlm2vec_inbatch: one positive per query, "
            "public image-only source-block interleaving, cross-device in-batch negatives"
        )
        print_master(
            "[VLM2VEC_INBATCH_DATASET_CONTRACT_OK] "
            f"sources={len(configured_sources)} order={configured_sources} "
            f"caps={configured_caps}"
        )
        # VLM2Vec's public train_image.yaml assigns weight=1 to every one of
        # these 20 MMEB sources. The local mmeb20.yaml contains B3-specific
        # 0.5 weights, so those must not leak into this control experiment.
        configured_weights = [float(config["weight"]) for config in dataset_config.values()]
        dataset_weights = [1.0] * len(dataset_config)
        weight_sum = sum(dataset_weights)
        normalized_weights = [weight / weight_sum for weight in dataset_weights]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if training_args.interleave_batch_size <= 0:
            source_block_size = 1
        elif training_args.interleave_batch_size <= 1.0:
            source_block_size = int(
                training_args.per_device_train_batch_size
                * world_size
                * training_args.interleave_batch_size
            )
        else:
            source_block_size = int(training_args.interleave_batch_size * world_size)
        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )
        if source_block_size <= 0 or global_batch_size % source_block_size != 0:
            raise ValueError(
                "VLM2Vec interleave source block must be positive and divide "
                f"the global batch: source_block={source_block_size}, "
                f"global_batch={global_batch_size}"
            )
        print_master(
            "[DATA_CONFIG] "
            f"path={os.path.abspath(data_args.dataset_config)} "
            f"sha256={dataset_config_sha256} datasets={len(dataset_config)}"
        )
        print_master(
            "[VLM2VEC_INBATCH_DATA] "
            f"names={list(dataset_config.keys())} lengths={dataset_lens} "
            f"configured_b3_weights_ignored={configured_weights} "
            f"effective_vlm2vec_weights={dataset_weights} "
            f"probabilities={normalized_weights} total_rows={sum(dataset_lens)} "
            f"source_block_size={source_block_size} "
            f"blocks_per_global_batch={global_batch_size // source_block_size} "
            f"pos_only={data_args.pos_only}"
        )
        repro_config = {
            "backbone": model_backbone,
            "model": model_args.model_name,
            "dataset_config_sha256": dataset_config_sha256,
            "seed": training_args.seed,
            "world_size": world_size,
            "per_device_batch": training_args.per_device_train_batch_size,
            "global_batch": global_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "max_steps": training_args.max_steps,
            "learning_rate": training_args.learning_rate,
            "lr_scheduler_type": str(training_args.lr_scheduler_type),
            "warmup_steps": training_args.warmup_steps,
            "optimizer": str(training_args.optim),
            "bf16": training_args.bf16,
            "temperature": model.temperature,
            "pooling": model_args.pooling,
            "normalize": model_args.normalize,
            "lora_r": model_args.lora_r,
            "lora_alpha": model_args.lora_alpha,
            "lora_target_modules": model_args.lora_target_modules,
            "gc_q_chunk_size": training_args.gc_q_chunk_size,
            "gc_p_chunk_size": training_args.gc_p_chunk_size,
            "gc_dynamic_limit": training_args.gc_dynamic_limit,
            "resize_min_pixels": data_args.resize_min_pixels,
            "resize_max_pixels": data_args.resize_max_pixels,
            "source_block_size": source_block_size,
            "effective_source_probabilities": normalized_weights,
            "ddp_broadcast_buffers": training_args.ddp_broadcast_buffers,
            "ddp_find_unused_parameters": training_args.ddp_find_unused_parameters,
        }
        if model_args.qwen3_5_prompt_profile != "legacy":
            from src.training.qwen3_5_prompt import prompt_config

            repro_config["qwen3_5_prompt"] = prompt_config(
                model_args.qwen3_5_prompt_profile
            )
        # Keep the established Qwen3.5 reproduction fingerprint byte-for-byte
        # unchanged; record these extra implementation-critical fields only for
        # the newly added Qwen2-VL control branch.
        if model_backbone == "qwen2_vl":
            repro_config.update(
                {
                    "qwen2_model_impl": model_args.qwen2_model_impl,
                    "qwen2_b3_attention_impl": (
                        model_args.qwen2_b3_attention_impl
                    ),
                    "qwen2_b3_legacy_position_ids": (
                        model_args.qwen2_b3_legacy_position_ids
                    ),
                    "lora_dropout": model_args.lora_dropout,
                    "negative_strategy": data_args.negative_strategy,
                    "embedding_gradient_strategy": (
                        training_args.embedding_gradient_strategy
                    ),
                    "pos_only": data_args.pos_only,
                    "interleave_batch_size_per_device": (
                        training_args.interleave_batch_size
                    ),
                    "distributed_source_layout": (
                        "vlm2vec_example_shard_rank_major"
                    ),
                    "resize_use_processor": data_args.resize_use_processor,
                }
            )
            if model_args.qwen2_embedding_token:
                repro_config.update(
                    {
                        "qwen2_embedding_token": True,
                        "qwen2_embedding_token_text": (
                            model_args.qwen2_embedding_token_text
                        ),
                        "qwen2_embedding_token_id": getattr(
                            model_args, "qwen2_embedding_token_id", None
                        ),
                        "qwen2_embedding_token_trainable_rows": 1,
                    }
                )
        if model_args.moe_lora:
            repro_config.update(
                {
                    "architecture_ablation": "tsembed_moe_dora_only",
                    "moe_num_experts": model_args.moe_num_experts,
                    "moe_router_temperature": model_args.moe_router_temperature,
                    "moe_routing_scope": "per_token_per_projection",
                    "moe_expert_combination": "dense_softmax",
                    "moe_use_dora": True,
                    "moe_dora_norm_accumulation_dtype": "float32",
                    "moe_dora_direction_norm_gradient": "detached",
                    "moe_dora_dropout_semantics": (
                        "peft_0.19.1_per_expert_input_and_base_recompute"
                    ),
                    "moe_eans": False,
                    "moe_two_stage_training": False,
                    "lora_dropout": model_args.lora_dropout,
                    "negative_strategy": data_args.negative_strategy,
                    "embedding_gradient_strategy": (
                        training_args.embedding_gradient_strategy
                    ),
                }
            )
        if model_args.qwen3_5_bidirectional_attention:
            repro_config.update(
                {
                    "architecture_ablation": (
                        "qwen3_5_bidirectional_full_attention_mean_pooling"
                    ),
                    "full_attention_bidirectional": True,
                    "gated_deltanet_causal": True,
                    "mean_pooling_scope": "all_active_tokens",
                    "mean_pooling_padding_excluded": True,
                    "mean_pooling_accumulation_dtype": "float32",
                    "negative_strategy": data_args.negative_strategy,
                    "embedding_gradient_strategy": (
                        training_args.embedding_gradient_strategy
                    ),
                }
            )
        repro_json = json.dumps(repro_config, sort_keys=True, separators=(",", ":"))
        print_master(
            "[VLM2VEC_INBATCH_REPRO_CONFIG] "
            f"sha256={hashlib.sha256(repro_json.encode('utf-8')).hexdigest()} "
            f"values={repro_json}"
        )
        if training_args.embedding_gradient_strategy == "ega":
            ega_repro_config = {
                "base_repro_sha256": hashlib.sha256(
                    repro_json.encode("utf-8")
                ).hexdigest(),
                "embedding_gradient_strategy": "ega",
                "ega_alpha": training_args.ega_alpha,
                "ega_probability_dtype": "float32",
                "ega_preserve_negative_mass": True,
                "ega_global_cross_device_batch": True,
            }
            ega_repro_json = json.dumps(
                ega_repro_config, sort_keys=True, separators=(",", ":")
            )
            print_master(
                "[EGA_REPRO_CONFIG] "
                f"sha256={hashlib.sha256(ega_repro_json.encode('utf-8')).hexdigest()} "
                f"values={ega_repro_json}"
            )
        if training_args.embedding_gradient_strategy == "llave":
            llave_repro_config = {
                "base_repro_sha256": hashlib.sha256(
                    repro_json.encode("utf-8")
                ).hexdigest(),
                "embedding_gradient_strategy": "llave",
                "llave_alpha": training_args.llave_alpha,
                "temperature": model.temperature,
                "policy_score": "cosine/temperature",
                "reward_score": "alpha*stop_gradient(cosine)",
                "reward_scope": "negative_pairs_only",
                "positive_reward": 0.0,
                "reward_model": "policy_model_synchronized_each_step",
                "directions": ["query_to_target", "target_to_query"],
                "direction_reduction": "mean",
                "logit_dtype": "float32",
                "global_cross_device_negatives": True,
            }
            llave_repro_json = json.dumps(
                llave_repro_config, sort_keys=True, separators=(",", ":")
            )
            print_master(
                "[LLAVE_REPRO_CONFIG] "
                f"sha256={hashlib.sha256(llave_repro_json.encode('utf-8')).hexdigest()} "
                f"values={llave_repro_json}"
            )
    elif data_args.negative_strategy == "unime_v2":
        print_master(
            "[TRAINING_STRATEGY] unime_v2: one positive and eight ordered "
            "teacher-scored hard negatives per query; symmetric distribution KL"
        )
    else:
        if training_args.embedding_gradient_strategy == "b3_ega":
            print_master(
                "[TRAINING_STRATEGY] b3_ega: verified B3 data/batch construction "
                "with rectangular Explicit Gradient Amplification"
            )
            b3_ega_repro_config = {
                "method": "b3_ega",
                "backbone": model_backbone,
                "model": model_args.model_name,
                "dataset_config_sha256": dataset_config_sha256,
                "alpha": float(training_args.ega_alpha),
                "temperature": float(model.temperature),
                "target_per_query": 6,
                "explicit_hard_negatives_per_query": 5,
                "negative_scope": "all_global_targets_except_aligned_positive",
                "preserve_negative_mass": True,
                "probability_dtype": "float32",
                "world_size": (
                    dist.get_world_size() if dist.is_initialized() else 1
                ),
                "per_device_batch": training_args.per_device_train_batch_size,
                "max_steps": training_args.max_steps,
                "seed": training_args.seed,
            }
            b3_ega_repro_json = json.dumps(
                b3_ega_repro_config, sort_keys=True, separators=(",", ":")
            )
            print_master(
                "[B3_EGA_REPRO_CONFIG] "
                f"sha256={hashlib.sha256(b3_ega_repro_json.encode('utf-8')).hexdigest()} "
                f"values={b3_ega_repro_json}"
            )
        else:
            if model_args.moe_lora:
                print_master(
                    "[TRAINING_STRATEGY] b3_moe: verified B3 data/batch "
                    "construction with TSEmbed dense MoE-DoRA architecture; "
                    "EANS and two-stage training disabled"
                )
                b3_moe_repro_config = {
                    "method": "b3_tsembed_moe_dora_only",
                    "backbone": model_backbone,
                    "model": model_args.model_name,
                    "dataset_config_sha256": dataset_config_sha256,
                    "negative_strategy": data_args.negative_strategy,
                    "embedding_gradient_strategy": (
                        training_args.embedding_gradient_strategy
                    ),
                    "sampler": "sdibn+odibn",
                    "target_per_query": 6,
                    "explicit_hard_negatives_per_query": 5,
                    "num_experts": model_args.moe_num_experts,
                    "rank_per_expert": model_args.lora_r,
                    "alpha_per_expert": model_args.lora_alpha,
                    "dropout": model_args.lora_dropout,
                    "router_temperature": (
                        model_args.moe_router_temperature
                    ),
                    "routing_scope": "per_token_per_projection",
                    "expert_combination": "dense_softmax",
                    "use_dora": True,
                    "eans": False,
                    "two_stage_training": False,
                    "world_size": (
                        dist.get_world_size() if dist.is_initialized() else 1
                    ),
                    "per_device_batch": (
                        training_args.per_device_train_batch_size
                    ),
                    "max_steps": training_args.max_steps,
                    "seed": training_args.seed,
                }
                b3_moe_repro_json = json.dumps(
                    b3_moe_repro_config,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                print_master(
                    "[B3_MOE_REPRO_CONFIG] "
                    "sha256="
                    f"{hashlib.sha256(b3_moe_repro_json.encode('utf-8')).hexdigest()} "
                    f"values={b3_moe_repro_json}"
                )
            else:
                print_master("[TRAINING_STRATEGY] b3")

    if (
        data_args.negative_strategy in {"vlm2vec_inbatch", "unime_v2"}
        or data_args.rdibn
        or data_args.sdibn
        or data_args.odibn
    ) and len(dataset_lens) >= 1:
        training_args.accelerator_config.use_seedable_sampler=False


    assert data_args.max_len is None
    assert training_args.gradient_accumulation_steps==1
    assert data_args.image_resolution is None
    assert data_args.resize_use_processor


    trainer_cls = GradCacheLateProcessTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=processor,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        max_length=data_args.max_len,
        eval_dataset=data_args.eval_dataset_name,
        data_args=data_args,
        training_args=training_args,
    )
    train_dataset.trainer = trainer
    print(f">>>>>>>>>dataset lengths: {dataset_lens}", flush=True)
    if data_args.negative_strategy == "vlm2vec_inbatch":
        total_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        total_bs = total_bs * dist.get_world_size() if dist.is_initialized() else total_bs
        dataset_weights = [1.0] * len(dataset_config)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if training_args.interleave_batch_size <= 0:
            source_block_size = 1
        elif training_args.interleave_batch_size <= 1.0:
            source_block_size = int(
                training_args.per_device_train_batch_size
                * world_size
                * training_args.interleave_batch_size
            )
        else:
            source_block_size = int(training_args.interleave_batch_size * world_size)
        trainer._get_train_sampler = lambda dataset: VLM2VecInBatchSampler(
            total_batch_size=total_bs,
            ds_lens=dataset_lens,
            probabilities=dataset_weights,
            source_block_size=source_block_size,
            num_steps=training_args.max_steps,
            data_source=train_dataset,
            seed=training_args.seed,
            # The public VLM2Vec Qwen2 recipe splits its interleaved iterable
            # stream by individual examples across ranks. Enable the matching
            # map-style layout only for this new Qwen2 branch, leaving the
            # established Qwen3.5 in-batch sampling behavior unchanged.
            match_vlm2vec_rank_sharding=(model_backbone == "qwen2_vl"),
        )
    elif (data_args.rdibn or data_args.sdibn or data_args.odibn) and len(dataset_lens)>=1:
        # Multiple embedding datasets & we want to make sure each batch mostly comes from one dataset
        # Set custom sampler, see https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/trainer.py#L785
        total_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        total_bs = total_bs * dist.get_world_size() if dist.is_initialized() else total_bs

        num_samples = np.sum(dataset_lens)
        num_repeats = math.ceil((total_bs*training_args.max_steps)/num_samples)

        print(f">>>>>>>>>>Embedding dataset lengths inside: {dataset_lens}; training args {training_args}; num_repeats {num_repeats}", flush=True)
        trainer._get_train_sampler = lambda dataset: CustomRandomSampler(
            total_batch_size=total_bs, ds_lens=dataset_lens,
            _num_samples=num_samples, data_source=train_dataset, ordered_datasets=data_args.odibn, same_datasets=data_args.sdibn, random_datasets=data_args.rdibn, num_repeats=num_repeats, chunk_size=data_args.chunk_size)

    # import ipdb; ipdb.set_trace()

    # B3 debug: add embedding monitor callback
    if _B3_DEBUG:
        debug_cb = B3DebugCallback(model=model, interval=_B3_DEBUG_INTERVAL)
        debug_cb.trainer = trainer
        trainer.add_callback(debug_cb)
        print_master(f"[B3_DEBUG] Added B3DebugCallback (interval={_B3_DEBUG_INTERVAL} steps)")

    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
