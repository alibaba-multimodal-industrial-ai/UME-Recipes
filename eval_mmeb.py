"""B3-compatible MMEB image evaluation for checkpoints trained in this repo.

The evaluator shares dataset, prompt-prefix, pooling, scoring and load-once
semantics across supported backbones.  Model and processor selection still
follows the checkpoint's exact backbone so Qwen2 and Qwen3.5 adapters cannot be
silently mixed.
"""

import json
import os
import pickle
import random
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, HfArgumentParser

from src.training.arguments import DataArguments, ModelArguments, TrainingArguments
from src.training.data.collator import EvalCollator
from src.training.data.dataset import EvalDataset
from src.training.model import (
    MMEBModel,
    get_backbone_name,
    load_processor,
    validate_qwen2_embedding_token_checkpoint,
)
from src.training.utils import batch_to_device, print_master, print_rank


POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
    "ImageNet-1K": POS_MOD_CLASS_LABEL,
    "HatefulMemes": POS_MOD_CLASS_LABEL,
    "SUN397": POS_MOD_CLASS_LABEL,
    "N24News": POS_MOD_CLASS_LABEL,
    "VOC2007": POS_MOD_CLASS_LABEL,
    "Place365": POS_MOD_CLASS_LABEL,
    "ImageNet-A": POS_MOD_CLASS_LABEL,
    "ImageNet-R": POS_MOD_CLASS_LABEL,
    "ObjectNet": POS_MOD_CLASS_LABEL,
    "Country211": POS_MOD_CLASS_LABEL,
    "OK-VQA": POS_MOD_ANSWER,
    "A-OKVQA": POS_MOD_ANSWER,
    "DocVQA": POS_MOD_ANSWER,
    "InfographicsVQA": POS_MOD_ANSWER,
    "ChartQA": POS_MOD_ANSWER,
    "Visual7W": POS_MOD_ANSWER,
    "ScienceQA": POS_MOD_ANSWER,
    "GQA": POS_MOD_ANSWER,
    "TextVQA": POS_MOD_ANSWER,
    "VizWiz": POS_MOD_ANSWER,
    "MSCOCO_i2t": POS_MOD_IMAGE_CAPTION,
    "VisualNews_i2t": POS_MOD_IMAGE_CAPTION,
}

QWEN2_EVALUATOR_FORMAT = "b3-native-qwen2-legacy-position-v2"
QWEN3_5_EVALUATOR_FORMAT = "b3-native-qwen3.5-transformers5.9-v1"


def _atomic_pickle_dump(payload, path):
    """Publish a pickle only after its complete contents are durable."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _embedding_cache_is_valid(path):
    """Reject empty/truncated caches left by an interrupted evaluation."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "rb") as handle:
            tensor, pairs = pickle.load(handle)
        return (
            isinstance(tensor, np.ndarray)
            and tensor.ndim == 2
            and len(pairs) == tensor.shape[0]
        )
    except Exception:
        return False


def _evaluator_format(model_args):
    if model_args.model_backbone == "qwen2_vl":
        return QWEN2_EVALUATOR_FORMAT
    if model_args.model_backbone == "qwen3_5_vl":
        return QWEN3_5_EVALUATOR_FORMAT
    raise ValueError(
        f"Unsupported B3 evaluation backbone: {model_args.model_backbone}"
    )


def _load_eval_data(data_args, subset):
    local_path = os.path.join(data_args.dataset_name, subset)
    if os.path.exists(local_path):
        return load_dataset(local_path, split=data_args.dataset_split)
    return load_dataset(
        data_args.dataset_name,
        subset,
        split=data_args.dataset_split,
    )


def _checkpoint_signature(model_args):
    checkpoint = os.path.realpath(model_args.checkpoint_path)
    weight_names = (
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    weight_info = {}
    for name in weight_names:
        path = os.path.join(checkpoint, name)
        if os.path.isfile(path):
            stat = os.stat(path)
            weight_info[name] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    if not weight_info:
        raise FileNotFoundError(
            f"No adapter weights found in checkpoint: {checkpoint}"
        )
    signature = {
        "evaluator_format": _evaluator_format(model_args),
        "base_model": os.path.realpath(model_args.model_name),
        "checkpoint": checkpoint,
        "adapter_weights": weight_info,
        "pooling": model_args.pooling,
        "normalize": model_args.normalize,
        "qwen2_b3_legacy_position_ids": (
            model_args.qwen2_b3_legacy_position_ids
        ),
    }
    # Keep the already-released Qwen2 manifest byte-for-byte compatible.
    if model_args.model_backbone == "qwen3_5_vl":
        signature["model_backbone"] = model_args.model_backbone
        if model_args.qwen3_5_prompt_profile != "legacy":
            from src.training.qwen3_5_prompt import prompt_config

            signature["qwen3_5_prompt"] = prompt_config(
                model_args.qwen3_5_prompt_profile
            )
    if model_args.moe_lora:
        moe_config_path = os.path.join(checkpoint, "moe_lora_config.json")
        if not os.path.isfile(moe_config_path):
            raise FileNotFoundError(
                f"MoE-DoRA evaluation requires {moe_config_path}"
            )
        with open(moe_config_path, "r", encoding="utf-8") as handle:
            signature["tsembed_moe_lora"] = json.load(handle)
    if model_args.qwen2_eval_token_pruning:
        signature["qwen2_eval_token_pruning"] = {
            "implementation": "qwen2-layer0-last-row-prune-v1",
            "selection_layer": 0,
            "importance": "mean_heads_last_query",
            "keep_ratio": model_args.qwen2_eval_token_keep_ratio,
            "short_text_threshold": (
                model_args.qwen2_eval_token_prune_short_text_threshold
            ),
            "min_keep_tokens": model_args.qwen2_eval_token_prune_min_tokens,
            "force_final_token": True,
            "preserve_original_mrope": True,
            "decoder_attention": "flash_attention_2",
        }
    if model_args.qwen2_embedding_token:
        token_config_path = os.path.join(
            checkpoint, "qwen2_embedding_token_config.json"
        )
        if not os.path.isfile(token_config_path):
            raise FileNotFoundError(
                "Embedding-token evaluation requires "
                f"{token_config_path}"
            )
        with open(token_config_path, "r", encoding="utf-8") as handle:
            signature["qwen2_embedding_token"] = json.load(handle)
    return signature


def _verify_output_provenance(output_dir, signature):
    """Prevent stale embeddings/scores from another checkpoint being reused."""
    manifest_path = os.path.join(output_dir, "evaluation_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != signature:
            raise RuntimeError(
                f"{output_dir} already contains results from a different "
                "model/checkpoint. Use a new output directory instead of "
                "mixing cached embeddings."
            )
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(signature, handle, indent=2, ensure_ascii=False)


def _gather_embeddings(
    tensor,
    paired_data,
    subset,
    side,
    output_dir,
    rank,
    world_size,
):
    if world_size == 1:
        return tensor.cpu().float().numpy(), paired_data

    # OSS/FUSE may expose a newly opened file before its contents become
    # visible to another process.  eval_all_qwen3_5.sh supplies a node-local
    # directory shared by all torchrun ranks, avoiding that consistency race.
    shard_root = os.environ.get("B3_EVAL_SHARD_ROOT", output_dir)
    shard_dir = os.path.join(shard_root, "_distributed_shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f"{subset}_{side}_rank{rank}.pkl")
    _atomic_pickle_dump((tensor.cpu(), paired_data), shard_path)

    dist.barrier()
    if rank != 0:
        dist.barrier()
        return None, None

    tensors = []
    all_pairs = []
    try:
        for shard_rank in range(world_size):
            path = os.path.join(
                shard_dir,
                f"{subset}_{side}_rank{shard_rank}.pkl",
            )
            with open(path, "rb") as handle:
                shard_tensor, shard_pairs = pickle.load(handle)
            if shard_tensor.shape[0]:
                tensors.append(shard_tensor)
                all_pairs.extend(shard_pairs)
        if not tensors:
            raise RuntimeError(
                f"All distributed shards are empty for {subset}/{side}. "
                "The evaluation dataset or candidate construction is empty."
            )
        embeddings = torch.cat(tensors, dim=0).float().numpy()
    finally:
        for shard_rank in range(world_size):
            path = os.path.join(
                shard_dir,
                f"{subset}_{side}_rank{shard_rank}.pkl",
            )
            if os.path.exists(path):
                os.remove(path)
        try:
            os.rmdir(shard_dir)
        except OSError:
            pass

    dist.barrier()
    return embeddings, all_pairs


@torch.no_grad()
def _encode(model, loader, side, device, description):
    chunks = []
    for batch in tqdm(loader, desc=description, disable=dist.is_initialized() and dist.get_rank() != 0):
        batch = batch_to_device(batch, device)
        with torch.autocast(
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
            device_type=device.type,
        ):
            output = model(qry=batch) if side == "qry" else model(tgt=batch)
        chunks.append(output[f"{side}_reps"].detach())
    if chunks:
        return torch.cat(chunks, dim=0).contiguous()
    config = model.encoder.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None and hasattr(config, "text_config"):
        hidden_size = getattr(config.text_config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError(
            f"Cannot determine embedding dimension from "
            f"{type(config).__name__}; neither config.hidden_size nor "
            "config.text_config.hidden_size is defined."
        )
    return torch.empty((0, hidden_size), device=device)


def _score_subset(data_args, model_args, subset, qry_cache, tgt_cache):
    with open(qry_cache, "rb") as handle:
        qry_tensor, qry_index = pickle.load(handle)
    with open(tgt_cache, "rb") as handle:
        tgt_tensor, tgt_index = pickle.load(handle)

    qry_key_to_emb = {
        (item["text"], item["img_path"]): embedding
        for embedding, item in zip(qry_tensor, qry_index)
    }
    tgt_key_to_emb = {
        (item["text"], item["img_path"]): embedding
        for embedding, item in zip(tgt_tensor, tgt_index)
    }

    eval_data = _load_eval_data(data_args, subset)
    correct = 0
    predictions = []
    skipped_empty_candidates = 0
    for row in tqdm(eval_data, desc=f"Score - {subset}"):
        qry_key = (row["qry_text"], row["qry_img_path"])
        if qry_key not in qry_key_to_emb:
            raise KeyError(f"Missing query embedding: {qry_key}")
        original_candidate_keys = list(
            zip(row["tgt_text"], row["tgt_img_path"])
        )
        if not original_candidate_keys:
            raise ValueError(f"Query has no candidates: {qry_key}")
        positive_key = original_candidate_keys[0]
        if positive_key == ("", ""):
            raise ValueError(
                f"Query has an empty positive target: {qry_key}"
            )
        candidate_keys = [
            key for key in original_candidate_keys if key != ("", "")
        ]
        skipped_empty_candidates += (
            len(original_candidate_keys) - len(candidate_keys)
        )
        missing = [
            key for key in candidate_keys if key not in tgt_key_to_emb
        ]
        if missing:
            raise KeyError(f"Missing target embedding: {missing[0]}")
        candidates = np.stack(
            [tgt_key_to_emb[key] for key in candidate_keys],
            axis=0,
        )
        query = np.asarray(qry_key_to_emb[qry_key])
        if model_args.normalize:
            query_norm = np.linalg.norm(query)
            candidate_norms = np.linalg.norm(candidates, axis=1)
            scores = candidates @ query / (candidate_norms * query_norm)
        else:
            scores = candidates @ query
        prediction = int(np.argmax(scores))
        predicted_key = candidate_keys[prediction]
        correct += int(predicted_key == positive_key)
        predictions.append(predicted_key)

    if skipped_empty_candidates:
        print_master(
            f"{subset}: ignored {skipped_empty_candidates} empty "
            "('', '') candidate placeholders during scoring"
        )
    return eval_data, correct, predictions


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not model_args.checkpoint_path:
        raise ValueError(
            "--checkpoint_path is required. --model_name must point to the "
            "unchanged base model used to train that adapter."
        )

    base_config = AutoConfig.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
    )
    model_args.model_backbone = get_backbone_name(base_config)
    training_args.model_backbone = model_args.model_backbone
    from src.training.qwen3_5_prompt import (
        validate_checkpoint_config as validate_qwen3_5_prompt_checkpoint,
        validate_profile as validate_qwen3_5_prompt_profile,
    )

    validate_qwen3_5_prompt_profile(model_args.qwen3_5_prompt_profile)
    if model_args.qwen3_5_prompt_profile != "legacy":
        if model_args.model_backbone != "qwen3_5_vl":
            raise ValueError(
                "The role-aware prompt experiment supports only Qwen3.5-VL"
            )
        if model_args.pooling not in {"last", "eos"}:
            raise ValueError(
                "The role-aware prompt experiment requires final-token pooling"
            )
    if model_args.model_backbone == "qwen3_5_vl":
        validate_qwen3_5_prompt_checkpoint(
            model_args.checkpoint_path,
            model_args.qwen3_5_prompt_profile,
        )
    if model_args.qwen2_eval_token_pruning:
        pruning_contract_errors = []
        if model_args.model_backbone != "qwen2_vl":
            pruning_contract_errors.append("model_backbone must be qwen2_vl")
        if model_args.qwen2_model_impl != "b3":
            pruning_contract_errors.append("qwen2_model_impl must be b3")
        if model_args.qwen2_b3_attention_impl != "flash_attention_2":
            pruning_contract_errors.append(
                "qwen2_b3_attention_impl must be flash_attention_2"
            )
        if not (0.0 < model_args.qwen2_eval_token_keep_ratio <= 1.0):
            pruning_contract_errors.append("token keep ratio must be in (0, 1]")
        if model_args.qwen2_eval_token_prune_short_text_threshold < 0:
            pruning_contract_errors.append(
                "short-text threshold must be non-negative"
            )
        if model_args.qwen2_eval_token_prune_min_tokens < 1:
            pruning_contract_errors.append("minimum retained tokens must be positive")
        if model_args.qwen2_eval_token_prune_debug_interval < 1:
            pruning_contract_errors.append("debug interval must be positive")
        if pruning_contract_errors:
            raise ValueError(
                "Qwen2 evaluation token-pruning contract failed: "
                + "; ".join(pruning_contract_errors)
            )
    if model_args.qwen2_embedding_token:
        embedding_token_errors = []
        if model_args.model_backbone != "qwen2_vl":
            embedding_token_errors.append("model_backbone must be qwen2_vl")
        if model_args.qwen2_model_impl != "b3":
            embedding_token_errors.append("qwen2_model_impl must be b3")
        if model_args.pooling not in {"last", "eos"}:
            embedding_token_errors.append("pooling must be last/eos")
        if not model_args.lora:
            embedding_token_errors.append("lora must be enabled")
        if model_args.qwen2_eval_token_pruning:
            embedding_token_errors.append(
                "token pruning must be disabled for this isolated evaluation"
            )
        if embedding_token_errors:
            raise ValueError(
                "Qwen2 embedding-token evaluation contract failed: "
                + "; ".join(embedding_token_errors)
            )
    subsets = data_args.subset_name
    if isinstance(subsets, str):
        subsets = [subsets]

    if rank == 0:
        os.makedirs(data_args.encode_output_path, exist_ok=True)
        _verify_output_provenance(
            data_args.encode_output_path,
            _checkpoint_signature(model_args),
        )
    if dist.is_initialized():
        dist.barrier()

    processor = load_processor(model_args, data_args)
    if model_args.qwen2_embedding_token:
        validate_qwen2_embedding_token_checkpoint(
            model_args.checkpoint_path,
            model_args,
            processor._qwen2_embedding_token_id,
        )
    # Preserve B3's staggered local loading, but use the same native model
    # implementation and the same checkpoint on every rank.
    if rank:
        time.sleep(random.randint(1, min(3 * rank, 12)))
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(device=device, dtype=torch.bfloat16)
    token_pruning_controller = None
    token_pruning_stats = None
    token_pruning_stats_path = None
    if model_args.qwen2_eval_token_pruning:
        from src.evaluation.qwen2_token_pruning import (
            install_qwen2_eval_token_pruning,
        )

        token_pruning_controller = install_qwen2_eval_token_pruning(
            model, model_args
        )
        token_pruning_stats_path = os.path.join(
            data_args.encode_output_path, "qwen2_token_pruning_stats.json"
        )
        if rank == 0 and os.path.isfile(token_pruning_stats_path):
            with open(token_pruning_stats_path, "r", encoding="utf-8") as handle:
                token_pruning_stats = json.load(handle)
        elif rank == 0:
            token_pruning_stats = {
                "configuration": _checkpoint_signature(model_args)[
                    "qwen2_eval_token_pruning"
                ],
                "subsets": {},
            }
    print_rank(
        f"B3-consistent evaluation: rank={rank}/{world_size}, "
        f"backbone={model_args.model_backbone}, "
        f"checkpoint={model_args.checkpoint_path}, "
        f"qwen3_5_prompt_profile={model_args.qwen3_5_prompt_profile}"
    )

    for subset in subsets:
        score_path = os.path.join(
            data_args.encode_output_path,
            f"{subset}_score.json",
        )
        qry_cache = os.path.join(
            data_args.encode_output_path,
            f"{subset}_qry",
        )
        tgt_cache = os.path.join(
            data_args.encode_output_path,
            f"{subset}_tgt",
        )
        skip_encoding = False
        if rank == 0:
            qry_cache_valid = _embedding_cache_is_valid(qry_cache)
            tgt_cache_valid = _embedding_cache_is_valid(tgt_cache)
            skip_encoding = qry_cache_valid and tgt_cache_valid
            if not skip_encoding and (
                os.path.exists(qry_cache) or os.path.exists(tgt_cache)
            ):
                print_master(
                    f"{subset}: cached embeddings are incomplete or invalid; "
                    "re-encoding both query and target sides"
                )
        if dist.is_initialized():
            flag = torch.tensor(
                [int(skip_encoding)],
                dtype=torch.int64,
                device=device,
            )
            dist.broadcast(flag, src=0)
            skip_encoding = bool(flag.item())

        if not skip_encoding:
            qry_dataset = EvalDataset(
                data_args,
                model_args,
                subset,
                "qry_text",
                "qry_img_path",
            )
            tgt_dataset = EvalDataset(
                data_args,
                model_args,
                subset,
                "tgt_text",
                "tgt_img_path",
                mod_instruction=(
                    POS_MOD_DICT.get(subset)
                    if data_args.tgt_prefix_mod
                    else None
                ),
            )
            qry_collator = EvalCollator(
                data_args=data_args,
                model_args=model_args,
                processor=processor,
                prompt_dataset_name=subset,
                prompt_side="query",
            )
            tgt_collator = EvalCollator(
                data_args=data_args,
                model_args=model_args,
                processor=processor,
                prompt_dataset_name=subset,
                prompt_side="target",
            )
            if world_size > 1:
                qry_indices = list(range(rank, len(qry_dataset), world_size))
                tgt_indices = list(range(rank, len(tgt_dataset), world_size))
                qry_dataset.paired_dataset = qry_dataset.paired_dataset.select(qry_indices)
                tgt_dataset.paired_dataset = tgt_dataset.paired_dataset.select(tgt_indices)
                qry_dataset.paired_data = [
                    qry_dataset.paired_data[index] for index in qry_indices
                ]
                tgt_dataset.paired_data = [
                    tgt_dataset.paired_data[index] for index in tgt_indices
                ]

            qry_loader = DataLoader(
                qry_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=qry_collator,
                shuffle=False,
                num_workers=0,
            )
            tgt_loader = DataLoader(
                tgt_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=tgt_collator,
                shuffle=False,
                num_workers=0,
            )
            qry_embeddings = _encode(
                model,
                qry_loader,
                "qry",
                device,
                f"Query - {subset}",
            )
            qry_embeddings, qry_pairs = _gather_embeddings(
                qry_embeddings,
                qry_dataset.paired_data,
                subset,
                "qry",
                data_args.encode_output_path,
                rank,
                world_size,
            )
            if token_pruning_controller is not None:
                pruning_summary = token_pruning_controller.log_and_reset_summary(
                    f"{subset}/query"
                )
                if rank == 0:
                    token_pruning_stats["subsets"].setdefault(subset, {})[
                        "query"
                    ] = pruning_summary
                    with open(
                        token_pruning_stats_path, "w", encoding="utf-8"
                    ) as handle:
                        json.dump(
                            token_pruning_stats,
                            handle,
                            indent=2,
                            ensure_ascii=False,
                        )
            if rank == 0:
                _atomic_pickle_dump((qry_embeddings, qry_pairs), qry_cache)

            tgt_embeddings = _encode(
                model,
                tgt_loader,
                "tgt",
                device,
                f"Target - {subset}",
            )
            tgt_embeddings, tgt_pairs = _gather_embeddings(
                tgt_embeddings,
                tgt_dataset.paired_data,
                subset,
                "tgt",
                data_args.encode_output_path,
                rank,
                world_size,
            )
            if token_pruning_controller is not None:
                pruning_summary = token_pruning_controller.log_and_reset_summary(
                    f"{subset}/target"
                )
                if rank == 0:
                    token_pruning_stats["subsets"].setdefault(subset, {})[
                        "target"
                    ] = pruning_summary
                    with open(
                        token_pruning_stats_path, "w", encoding="utf-8"
                    ) as handle:
                        json.dump(
                            token_pruning_stats,
                            handle,
                            indent=2,
                            ensure_ascii=False,
                        )
            if rank == 0:
                _atomic_pickle_dump((tgt_embeddings, tgt_pairs), tgt_cache)
            if dist.is_initialized():
                dist.barrier()

        if rank == 0 and not os.path.isfile(score_path):
            eval_data, correct, predictions = _score_subset(
                data_args,
                model_args,
                subset,
                qry_cache,
                tgt_cache,
            )
            score = {
                "acc": correct / len(eval_data),
                "hit@1": correct / len(eval_data),
                "num_correct": correct,
                "num_pred": len(predictions),
                "num_data": len(eval_data),
                "checkpoint": os.path.realpath(model_args.checkpoint_path),
                "evaluator_format": _evaluator_format(model_args),
            }
            if model_args.qwen2_eval_token_pruning:
                score["qwen2_eval_token_pruning"] = (
                    _checkpoint_signature(model_args)[
                        "qwen2_eval_token_pruning"
                    ]
                )
            if model_args.qwen3_5_prompt_profile != "legacy":
                from src.training.qwen3_5_prompt import prompt_config

                score["qwen3_5_prompt"] = prompt_config(
                    model_args.qwen3_5_prompt_profile
                )
            with open(score_path, "w", encoding="utf-8") as handle:
                json.dump(score, handle, indent=2, ensure_ascii=False)
            with open(
                os.path.join(
                    data_args.encode_output_path,
                    f"{subset}_pred.txt",
                ),
                "w",
                encoding="utf-8",
            ) as handle:
                for prediction in predictions:
                    handle.write(f"{prediction}\n")
            print_master(f"{subset}: {score}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
