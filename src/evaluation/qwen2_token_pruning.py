"""Evaluation-only first-layer attention token pruning for B3 Qwen2-VL.

All decoder blocks remain on FlashAttention 2.  Hooks capture layer 0's
already-computed Q/K projections, and this module recomputes only the final
query row of attention.  After layer 0, selected tokens are repacked with left
padding while their original multimodal RoPE coordinates are preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn

from src.training.vlm_backbone.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLFlashAttention2,
    repeat_kv,
)


def _rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def select_keep_indices(
    importance: Tensor,
    valid_mask: Tensor,
    visual_rows: Tensor,
    keep_ratio: float,
    short_text_threshold: int,
    min_keep_tokens: int,
) -> Tuple[List[Tensor], List[bool]]:
    """Select sorted padded-sequence indices independently for every row."""
    if importance.ndim != 2 or valid_mask.shape != importance.shape:
        raise ValueError(
            "importance and valid_mask must share shape [batch, sequence], got "
            f"{tuple(importance.shape)} and {tuple(valid_mask.shape)}"
        )
    if visual_rows.shape != (importance.shape[0],):
        raise ValueError(
            f"visual_rows must have shape ({importance.shape[0]},), got "
            f"{tuple(visual_rows.shape)}"
        )
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if short_text_threshold < 0:
        raise ValueError("short_text_threshold must be non-negative")
    if min_keep_tokens < 1:
        raise ValueError("min_keep_tokens must be positive")

    selections: List[Tensor] = []
    bypassed: List[bool] = []
    for row in range(importance.shape[0]):
        active = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        if active.numel() == 0:
            raise ValueError(f"Cannot token-prune empty evaluation row {row}")
        bypass = bool(
            not bool(visual_rows[row].item())
            and active.numel() <= short_text_threshold
        )
        if bypass:
            selected = active
        else:
            keep_count = min(
                active.numel(),
                max(min_keep_tokens, math.ceil(active.numel() * keep_ratio)),
            )
            final_index = active[-1]
            historical = active[:-1]
            other_count = keep_count - 1
            if other_count:
                ranked = torch.topk(
                    importance[row, historical],
                    k=other_count,
                    largest=True,
                    sorted=False,
                ).indices
                selected = torch.cat((historical[ranked], final_index.view(1)))
                selected = torch.sort(selected).values
            else:
                selected = final_index.view(1)
        if selected[-1] != active[-1]:
            raise RuntimeError("Token pruning failed to preserve the final token")
        if selected.numel() > 1 and not torch.all(selected[1:] > selected[:-1]):
            raise RuntimeError("Selected token indices are not strictly ordered")
        selections.append(selected)
        bypassed.append(bypass)
    return selections, bypassed


@dataclass
class _Counters:
    batches: int = 0
    rows: int = 0
    visual_rows: int = 0
    short_text_bypassed: int = 0
    original_tokens: int = 0
    retained_tokens: int = 0
    original_visual_tokens: int = 0
    retained_visual_tokens: int = 0

    def values(self) -> List[int]:
        return [
            self.batches,
            self.rows,
            self.visual_rows,
            self.short_text_bypassed,
            self.original_tokens,
            self.retained_tokens,
            self.original_visual_tokens,
            self.retained_visual_tokens,
        ]

    def clear(self) -> None:
        self.batches = 0
        self.rows = 0
        self.visual_rows = 0
        self.short_text_bypassed = 0
        self.original_tokens = 0
        self.retained_tokens = 0
        self.original_visual_tokens = 0
        self.retained_visual_tokens = 0


class Qwen2FirstLayerTokenPruner:
    def __init__(
        self,
        outer_model: nn.Module,
        decoder_model: nn.Module,
        keep_ratio: float,
        short_text_threshold: int,
        min_keep_tokens: int,
        debug_interval: int,
    ) -> None:
        if decoder_model.training or outer_model.training:
            raise RuntimeError("Qwen2 token pruning may only be installed in eval mode")
        if getattr(decoder_model.config, "_attn_implementation", None) != "flash_attention_2":
            raise RuntimeError(
                "Qwen2 token pruning requires every decoder layer to use "
                "flash_attention_2"
            )
        if not decoder_model.layers:
            raise RuntimeError("Qwen2 decoder contains no layers")
        non_flash_layers = [
            index
            for index, layer in enumerate(decoder_model.layers)
            if not isinstance(layer.self_attn, Qwen2VLFlashAttention2)
        ]
        if non_flash_layers:
            raise RuntimeError(
                "Qwen2 token pruning requires FlashAttention 2 in every decoder "
                f"layer; non-flash layers={non_flash_layers}"
            )
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if short_text_threshold < 0 or min_keep_tokens < 1 or debug_interval < 1:
            raise ValueError("Invalid Qwen2 token-pruning integer configuration")

        self.outer_model = outer_model
        self.decoder_model = decoder_model
        self.keep_ratio = float(keep_ratio)
        self.short_text_threshold = int(short_text_threshold)
        self.min_keep_tokens = int(min_keep_tokens)
        self.debug_interval = int(debug_interval)
        self._query_projection: Optional[Tensor] = None
        self._key_projection: Optional[Tensor] = None
        self._visual_rows: Optional[Tensor] = None
        self._visual_token_mask: Optional[Tensor] = None
        self._input_attention_mask: Optional[Tensor] = None
        self._counters = _Counters()

        first_attention = decoder_model.layers[0].self_attn
        self._handles = [
            outer_model.register_forward_pre_hook(
                self._capture_outer_inputs, with_kwargs=True
            ),
            first_attention.q_proj.register_forward_hook(
                self._capture_query_projection
            ),
            first_attention.k_proj.register_forward_hook(
                self._capture_key_projection
            ),
        ]

    def _capture_outer_inputs(self, _module, args, kwargs):
        if args:
            raise RuntimeError(
                "Qwen2 token-pruning evaluator expects keyword-only model inputs"
            )
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
            raise TypeError(
                "Qwen2 token pruning requires tensor input_ids and attention_mask"
            )
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids/attention_mask mismatch before pruning: "
                f"{tuple(input_ids.shape)} != {tuple(attention_mask.shape)}"
            )
        if not bool(attention_mask[:, -1].bool().all().item()):
            raise ValueError(
                "Qwen2 token pruning requires left padding so every final "
                "pooling token occupies the last sequence column"
            )
        image_id = getattr(self.outer_model.config, "image_token_id", None)
        video_id = getattr(self.outer_model.config, "video_token_id", None)
        visual_tokens = torch.zeros_like(input_ids, dtype=torch.bool)
        if image_id is not None:
            visual_tokens |= input_ids == image_id
        if video_id is not None:
            visual_tokens |= input_ids == video_id
        visual_tokens &= attention_mask.bool()
        self._visual_token_mask = visual_tokens
        self._visual_rows = visual_tokens.any(dim=-1)
        self._input_attention_mask = attention_mask.bool()
        self._query_projection = None
        self._key_projection = None

    def _capture_query_projection(self, _module, _args, output):
        self._query_projection = output

    def _capture_key_projection(self, _module, _args, output):
        self._key_projection = output

    def _last_query_importance(
        self,
        position_embeddings: Tuple[Tensor, Tensor],
        valid_mask: Tensor,
    ) -> Tensor:
        query_projection = self._query_projection
        key_projection = self._key_projection
        if query_projection is None or key_projection is None:
            raise RuntimeError("Layer-0 Q/K projections were not captured")
        batch, sequence, _ = query_projection.shape
        attention = self.decoder_model.layers[0].self_attn
        if key_projection.shape[:2] != (batch, sequence):
            raise ValueError("Captured Q/K projection shapes do not align")
        if valid_mask.shape != (batch, sequence):
            raise ValueError(
                f"Captured attention mask is {tuple(valid_mask.shape)}, "
                f"expected {(batch, sequence)}"
            )

        query = query_projection.view(
            batch, sequence, attention.num_heads, attention.head_dim
        ).transpose(1, 2)
        key = key_projection.view(
            batch, sequence, attention.num_key_value_heads, attention.head_dim
        ).transpose(1, 2)
        cos, sin = position_embeddings
        if cos.shape[:3] != (3, batch, sequence) or sin.shape != cos.shape:
            raise ValueError(
                "Unexpected Qwen2 multimodal RoPE shape: "
                f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
            )
        sections = list(attention.rope_scaling["mrope_section"]) * 2
        composed_cos = torch.cat(
            [chunk[index % 3] for index, chunk in enumerate(cos.split(sections, dim=-1))],
            dim=-1,
        )
        composed_sin = torch.cat(
            [chunk[index % 3] for index, chunk in enumerate(sin.split(sections, dim=-1))],
            dim=-1,
        )
        positions = torch.arange(sequence, device=valid_mask.device).expand(batch, -1)
        last_positions = positions.masked_fill(~valid_mask, -1).max(dim=-1).values
        if torch.any(last_positions < 0):
            raise ValueError("Qwen2 token pruning encountered an empty row")
        batch_indices = torch.arange(batch, device=query.device)
        query_indices = last_positions[:, None, None, None].expand(
            batch, attention.num_heads, 1, attention.head_dim
        )
        last_query = query.gather(2, query_indices)
        last_cos = composed_cos[batch_indices, last_positions].unsqueeze(1).unsqueeze(2)
        last_sin = composed_sin[batch_indices, last_positions].unsqueeze(1).unsqueeze(2)
        last_query = (last_query * last_cos) + (_rotate_half(last_query) * last_sin)
        key = (key * composed_cos.unsqueeze(1)) + (
            _rotate_half(key) * composed_sin.unsqueeze(1)
        )
        key = repeat_kv(key, attention.num_key_value_groups)
        # Mirror Qwen2VLFlashAttention2's PEFT/autocast dtype correction so the
        # ranking uses the same Q/K precision as the actual layer-0 attention.
        if last_query.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(attention.config, "_pre_quantization_dtype"):
                target_dtype = attention.config._pre_quantization_dtype
            else:
                target_dtype = attention.q_proj.weight.dtype
            last_query = last_query.to(target_dtype)
            key = key.to(target_dtype)
        scores = torch.matmul(last_query, key.transpose(2, 3)).squeeze(2)
        scores = scores / math.sqrt(attention.head_dim)
        score_mask = valid_mask
        if (
            attention.config.use_sliding_window
            and getattr(attention.config, "sliding_window", None) is not None
            and attention.layer_idx >= attention.config.max_window_layers
        ):
            positions = torch.arange(sequence, device=valid_mask.device)
            window_start = (last_positions - attention.config.sliding_window + 1).clamp_min(0)
            score_mask = score_mask & (positions[None, :] >= window_start[:, None])
        scores = scores.masked_fill(
            ~score_mask[:, None, :], torch.finfo(scores.dtype).min
        )
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        importance = probabilities.mean(dim=1)
        if not torch.isfinite(importance).all():
            raise FloatingPointError("Non-finite layer-0 last-token attention importance")
        return importance

    @staticmethod
    def _repack(
        hidden_states: Tensor,
        position_ids: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        selections: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        batch, _, hidden = hidden_states.shape
        max_keep = max(int(indices.numel()) for indices in selections)
        new_hidden = hidden_states.new_zeros((batch, max_keep, hidden))
        new_mask = torch.zeros(
            (batch, max_keep), dtype=torch.long, device=hidden_states.device
        )
        new_position_ids = position_ids.new_zeros((3, batch, max_keep))
        cos, sin = position_embeddings
        new_cos = cos.new_zeros((3, batch, max_keep, cos.shape[-1]))
        new_sin = sin.new_zeros((3, batch, max_keep, sin.shape[-1]))
        for row, indices in enumerate(selections):
            count = int(indices.numel())
            destination = slice(max_keep - count, max_keep)
            new_hidden[row, destination] = hidden_states[row, indices]
            new_mask[row, destination] = 1
            new_position_ids[:, row, destination] = position_ids[:, row, indices]
            new_cos[:, row, destination] = cos[:, row, indices]
            new_sin[:, row, destination] = sin[:, row, indices]
        return new_hidden, new_mask, new_position_ids, (new_cos, new_sin)

    def _record_batch(
        self,
        hidden_states: Tensor,
        original_lengths: Tensor,
        retained_lengths: Tensor,
        bypassed: Sequence[bool],
        selections: Optional[Sequence[Tensor]],
    ) -> None:
        if self._visual_rows is None or self._visual_token_mask is None:
            raise RuntimeError("Visual-row state was cleared before accounting")
        original_visual = int(self._visual_token_mask.sum().item())
        if selections is None:
            retained_visual = original_visual
            retained_visual_by_row = self._visual_token_mask.sum(dim=-1).tolist()
        else:
            retained_visual_by_row = [
                int(self._visual_token_mask[row, indices].sum().item())
                for row, indices in enumerate(selections)
            ]
            retained_visual = sum(retained_visual_by_row)
        self._counters.batches += 1
        self._counters.rows += hidden_states.shape[0]
        self._counters.visual_rows += int(self._visual_rows.sum().item())
        self._counters.short_text_bypassed += sum(bypassed)
        self._counters.original_tokens += int(original_lengths.sum().item())
        self._counters.retained_tokens += int(retained_lengths.sum().item())
        self._counters.original_visual_tokens += original_visual
        self._counters.retained_visual_tokens += retained_visual
        if self._counters.batches <= 3 or self._counters.batches % self.debug_interval == 0:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                removed_lengths = (original_lengths - retained_lengths).tolist()
                row_keep_ratios = (
                    retained_lengths.float() / original_lengths.clamp_min(1).float()
                ).tolist()
                print(
                    f"[QWEN2_TOKEN_PRUNE_BATCH] rank={rank} "
                    f"batch={self._counters.batches} rows={hidden_states.shape[0]} "
                    f"original={original_lengths.tolist()} "
                    f"retained={retained_lengths.tolist()} "
                    f"removed={removed_lengths} "
                    f"effective_keep_ratio={row_keep_ratios} "
                    f"visual={self._visual_rows.tolist()} "
                    f"retained_visual_tokens={retained_visual_by_row} "
                    f"short_text_bypass={list(bypassed)}",
                    flush=True,
                )

    def _clear_batch_state(self) -> None:
        self._query_projection = None
        self._key_projection = None
        self._visual_rows = None
        self._visual_token_mask = None
        self._input_attention_mask = None

    def prune_after_first_layer(
        self,
        *,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Tensor,
        cache_position: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        use_cache: bool,
        output_attentions: bool,
    ):
        if self.decoder_model.training or torch.is_grad_enabled():
            raise RuntimeError("Qwen2 token pruning is inference-only")
        if use_cache or output_attentions:
            raise RuntimeError(
                "Qwen2 token pruning requires use_cache=False and output_attentions=False"
            )
        if self._visual_rows is None or self._input_attention_mask is None:
            raise RuntimeError("Outer Qwen2 inputs were not captured for token pruning")
        valid_mask = self._input_attention_mask
        if valid_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "Layer-0 hidden/input-mask mismatch: "
                f"{tuple(hidden_states.shape[:2])} != {tuple(valid_mask.shape)}"
            )
        original_lengths = valid_mask.sum(dim=-1)
        all_short_text = (~self._visual_rows) & (
            original_lengths <= self.short_text_threshold
        )
        if bool(all_short_text.all().item()):
            bypassed = [True] * hidden_states.shape[0]
            self._record_batch(
                hidden_states,
                original_lengths,
                original_lengths,
                bypassed,
                selections=None,
            )
            self._clear_batch_state()
            return (
                hidden_states,
                attention_mask,
                position_ids,
                cache_position,
                position_embeddings,
            )
        importance = self._last_query_importance(position_embeddings, valid_mask)
        selections, bypassed = select_keep_indices(
            importance,
            valid_mask,
            self._visual_rows,
            self.keep_ratio,
            self.short_text_threshold,
            self.min_keep_tokens,
        )
        retained_lengths = torch.tensor(
            [indices.numel() for indices in selections],
            device=hidden_states.device,
            dtype=original_lengths.dtype,
        )
        self._record_batch(
            hidden_states,
            original_lengths,
            retained_lengths,
            bypassed,
            selections=selections,
        )

        new_hidden, new_mask, new_position_ids, new_position_embeddings = self._repack(
            hidden_states, position_ids, position_embeddings, selections
        )
        if new_hidden.shape[1] >= hidden_states.shape[1] and torch.equal(
            new_mask.bool(), valid_mask
        ):
            new_hidden = hidden_states
            new_mask = attention_mask if attention_mask is not None else valid_mask.long()
            new_position_ids = position_ids
            new_position_embeddings = position_embeddings
        new_cache_position = torch.arange(
            new_hidden.shape[1], device=cache_position.device, dtype=cache_position.dtype
        )
        if not torch.isfinite(new_hidden).all():
            raise FloatingPointError("Non-finite hidden state after Qwen2 token pruning")

        self._clear_batch_state()
        return (
            new_hidden,
            new_mask,
            new_position_ids,
            new_cache_position,
            new_position_embeddings,
        )

    def log_and_reset_summary(self, label: str) -> Dict[str, object]:
        values = torch.tensor(
            self._counters.values(),
            dtype=torch.int64,
            device=next(self.decoder_model.parameters()).device,
        )
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        names = (
            "batches",
            "rows",
            "visual_rows",
            "short_text_bypassed",
            "original_tokens",
            "retained_tokens",
            "original_visual_tokens",
            "retained_visual_tokens",
        )
        summary = dict(zip(names, [int(value) for value in values.tolist()]))
        summary["removed_tokens"] = (
            summary["original_tokens"] - summary["retained_tokens"]
        )
        summary["configured_keep_ratio"] = self.keep_ratio
        summary["effective_keep_ratio"] = (
            summary["retained_tokens"] / summary["original_tokens"]
            if summary["original_tokens"]
            else 0.0
        )
        summary["average_original_tokens_per_row"] = (
            summary["original_tokens"] / summary["rows"]
            if summary["rows"]
            else 0.0
        )
        summary["average_retained_tokens_per_row"] = (
            summary["retained_tokens"] / summary["rows"]
            if summary["rows"]
            else 0.0
        )
        summary["visual_token_keep_ratio"] = (
            summary["retained_visual_tokens"]
            / summary["original_visual_tokens"]
            if summary["original_visual_tokens"]
            else 0.0
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[QWEN2_TOKEN_PRUNE_SUMMARY] label={label} "
                f"batches={summary['batches']} rows={summary['rows']} "
                f"visual_rows={summary['visual_rows']} "
                f"short_text_bypassed={summary['short_text_bypassed']} "
                f"original_tokens={summary['original_tokens']} "
                f"retained_tokens={summary['retained_tokens']} "
                f"removed_tokens={summary['removed_tokens']} "
                f"visual_tokens={summary['original_visual_tokens']}->"
                f"{summary['retained_visual_tokens']} "
                f"configured_keep_ratio={summary['configured_keep_ratio']:.6f} "
                f"effective_keep_ratio={summary['effective_keep_ratio']:.6f} "
                f"avg_tokens={summary['average_original_tokens_per_row']:.2f}->"
                f"{summary['average_retained_tokens_per_row']:.2f}",
                flush=True,
            )
        self._counters.clear()
        return summary


def install_qwen2_eval_token_pruning(model, model_args) -> Qwen2FirstLayerTokenPruner:
    if model.training:
        raise RuntimeError("Install Qwen2 evaluation token pruning only after model.eval()")
    if getattr(model_args, "model_backbone", None) != "qwen2_vl":
        raise ValueError("Qwen2 token pruning cannot be enabled for another backbone")
    if getattr(model_args, "qwen2_model_impl", None) != "b3":
        raise ValueError("Qwen2 token pruning currently requires qwen2_model_impl=b3")
    if getattr(model_args, "qwen2_b3_attention_impl", None) != "flash_attention_2":
        raise ValueError("Qwen2 token pruning requires qwen2_b3_attention_impl=flash_attention_2")
    outer_model = model.encoder
    decoder_model = getattr(outer_model, "model", None)
    if decoder_model is None or not hasattr(decoder_model, "layers"):
        raise TypeError(
            f"Cannot locate Qwen2 decoder under {type(outer_model).__name__}"
        )
    if hasattr(decoder_model, "_eval_token_pruning_controller"):
        raise RuntimeError("Qwen2 token pruning was already installed")
    controller = Qwen2FirstLayerTokenPruner(
        outer_model=outer_model,
        decoder_model=decoder_model,
        keep_ratio=model_args.qwen2_eval_token_keep_ratio,
        short_text_threshold=model_args.qwen2_eval_token_prune_short_text_threshold,
        min_keep_tokens=model_args.qwen2_eval_token_prune_min_tokens,
        debug_interval=model_args.qwen2_eval_token_prune_debug_interval,
    )
    decoder_model._eval_token_pruning_controller = controller
    print(
        "[QWEN2_TOKEN_PRUNE_OK] layer=0 importance=mean_heads_last_query "
        f"decoder_layers={len(decoder_model.layers)} "
        f"keep_ratio={controller.keep_ratio} "
        f"short_text_threshold={controller.short_text_threshold} "
        f"min_keep_tokens={controller.min_keep_tokens} "
        "decoder_backend=flash_attention_2 preserve_original_mrope=true",
        flush=True,
    )
    return controller
