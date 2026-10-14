"""Isolated UniME-V2 soft-distribution training for verified VLM backbones.

The data expansion and objective follow the released UniME-V2 implementation:
one source record becomes five encoder pairs and represents one query, one
positive, and the first eight teacher-ranked hard negatives.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import RandomSampler

from src.training.data.collator import MultimodalDataCollator
from src.training.data.collator import process_vlm_inputs_fns
from src.training.model import PHI3V, VLM_IMAGE_TOKENS
from src.training.trainer import GradCacheLateProcessTrainer
from src.training.utils import find_latest_checkpoint, print_master


_METHOD = "unime_v2"
_CONFIG_FILE = "unime_v2_training_config.json"
_EXPANSION = 5
_OFFICIAL_SHUFFLE_SEED = 42


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_source_record(row: Dict[str, Any], index: int, hard_count: int) -> None:
    required = {
        "query_text",
        "query_image",
        "pos_text",
        "pos_image",
        "query_pos_scores",
        "hard_negatives",
        "hard_negatives_scores",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise KeyError(f"UniME-V2 row {index} lacks fields: {missing}")
    negatives = row["hard_negatives"]
    scores = row["hard_negatives_scores"]
    if len(negatives) < hard_count or len(scores) < hard_count:
        raise ValueError(
            f"UniME-V2 row {index} has only {len(negatives)} negatives and "
            f"{len(scores)} scores; expected at least {hard_count}"
        )
    for negative_index, pair in enumerate(negatives[:hard_count]):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"UniME-V2 row {index} hard negative {negative_index} must "
                f"be [text, image_path], got {pair!r}"
            )
    numeric_scores = [row["query_pos_scores"], *scores[:hard_count]]
    if not all(math.isfinite(float(score)) for score in numeric_scores):
        raise ValueError(f"UniME-V2 row {index} contains a non-finite teacher score")


def _official_pair_or_dummy(x_item, y_item):
    """Apply the released UniME-v2 invalid-pair fallback.

    The official dataset skips an invalid expanded pair while constructing its
    singleton lists.  Its collator then observes empty lists and substitutes a
    two-space, text-only input and zero scores on *both* sides.  Materializing
    that result directly preserves the fixed five-pair layout required by the
    loss and, unlike filtering, keeps every DDP rank at the same batch size.
    """
    x_text, x_image, x_score = x_item
    y_text, y_image, y_score = y_item
    x_valid = bool(x_text) or bool(x_image)
    y_valid = bool(y_text) or bool(y_image)
    if x_valid and y_valid:
        return x_item, y_item, False
    dummy = ("  ", "", 0.0)
    return dummy, dummy, True


def expand_unime_v2_row(row: Dict[str, Any], hard_count: int = 8):
    """Return official x/y/score sequences for one source row."""
    _validate_source_record(row, int(row.get("idx", -1)), hard_count)
    x_items = [(row["query_text"], row["query_image"], 0.0)]
    y_items = [(row["pos_text"], row["pos_image"], row["query_pos_scores"])]
    for hard_index in range(0, hard_count, 2):
        even = row["hard_negatives"][hard_index]
        odd = row["hard_negatives"][hard_index + 1]
        x_items.append((even[0], even[1], row["hard_negatives_scores"][hard_index]))
        y_items.append((odd[0], odd[1], row["hard_negatives_scores"][hard_index + 1]))
    return x_items, y_items


class UniMEV2Dataset(torch.utils.data.Dataset):
    """The official JSON dataset with its exact fixed Python shuffle."""

    def __init__(self, data_path: str, hard_count: int = 8):
        if hard_count != 8:
            raise ValueError(
                "The released UniME-V2 Qwen training objective requires exactly "
                f"eight hard negatives, got {hard_count}"
            )
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"UniME-V2 training JSON not found: {path}")
        print(
            f"[UNIME_V2_DATA_LOAD_BEGIN] rank={dist.get_rank() if dist.is_initialized() else 0} "
            f"path={path} bytes={path.stat().st_size}",
            flush=True,
        )
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or not rows:
            raise ValueError("UniME-V2 training JSON must be a non-empty list")

        # This deliberately matches dataset.py from the release, which calls
        # random.seed(42) and random.shuffle(train_data), instead of using an
        # Arrow/NumPy shuffle with a different permutation.
        # Preserve the release's global Python-RNG side effect as well as its
        # permutation; using Random(42).shuffle would match only the order.
        random.seed(_OFFICIAL_SHUFFLE_SEED)
        random.shuffle(rows)
        self.rows = rows
        self.num_rows = len(rows)
        self.hard_count = hard_count
        self.trainer = None

        # Validate the complete JSON before training.  Empty expanded pairs are
        # legal in the released code and become fixed-size dummy pairs; other
        # schema/score errors remain fatal here instead of surfacing hundreds of
        # optimizer steps later.
        dummy_pair_count = 0
        dummy_row_count = 0
        dummy_examples = []
        for index, row in enumerate(rows):
            _validate_source_record(row, index, hard_count)
            x_items, y_items = expand_unime_v2_row(row, hard_count)
            row_dummy_positions = []
            for position, (x_item, y_item) in enumerate(zip(x_items, y_items)):
                _, _, is_dummy = _official_pair_or_dummy(x_item, y_item)
                if is_dummy:
                    dummy_pair_count += 1
                    row_dummy_positions.append(position)
            if row_dummy_positions:
                dummy_row_count += 1
                if len(dummy_examples) < 16:
                    dummy_examples.append((index, row_dummy_positions))

        sentinel_indices = sorted({0, len(rows) // 2, len(rows) - 1})
        sentinels = [rows[index] for index in sentinel_indices]
        self.fingerprint = {
            "file_name": path.name,
            "file_bytes": path.stat().st_size,
            "rows": len(rows),
            "shuffle": f"python_random_seed_{_OFFICIAL_SHUFFLE_SEED}",
            "sentinel_indices": sentinel_indices,
            "sentinel_sha256": _sha256_json(sentinels),
        }
        print_master(
            "[UNIME_V2_DATA_OK] "
            f"rows={len(rows)} hard_negatives_per_query={hard_count} "
            f"shuffle=python_random_seed_{_OFFICIAL_SHUFFLE_SEED} "
            f"sentinel_indices={sentinel_indices} "
            f"sentinel_sha256={self.fingerprint['sentinel_sha256']}"
        )
        print_master(
            "[UNIME_V2_DUMMY_AUDIT] "
            f"rows={len(rows)} dummy_rows={dummy_row_count} "
            f"dummy_pairs={dummy_pair_count} expansion_pairs={len(rows) * _EXPANSION} "
            f"first_examples={dummy_examples} "
            "policy=official_fixed_layout_blank_text_and_zero_scores"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        _validate_source_record(row, index, self.hard_count)
        # Keep the shuffled dataset position so batch fingerprints and resume
        # comparisons identify the exact examples consumed.
        return {**row, "idx": index}


def init_unime_v2_dataset(dataset_config, model_args, data_args, training_args):
    """Drop-in loader used only by train_unime_v2.py."""
    if list(dataset_config) != ["UniME-V2"]:
        raise ValueError(
            "The isolated UniME-V2 config must contain exactly one 'UniME-V2' entry"
        )
    dataset = UniMEV2Dataset(
        data_path=data_args.unime_v2_data_path,
        hard_count=data_args.unime_v2_num_hard_negatives,
    )
    data_args._unime_v2_dataset_fingerprint = dataset.fingerprint
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    global_batch = training_args.per_device_train_batch_size * world_size
    if len(dataset) < global_batch:
        raise ValueError(
            f"UniME-V2 rows ({len(dataset)}) are fewer than global batch ({global_batch})"
        )
    print_master(
        "[UNIME_V2_BATCH_SCHEDULE] "
        f"source_rows={len(dataset)} original_queries_per_global_batch={global_batch} "
        f"expanded_pairs_per_global_batch={global_batch * _EXPANSION} "
        f"candidate_targets_per_query=9 steps={training_args.max_steps}"
    )
    return dataset, [len(dataset)]


def _image_descriptor(image_root: str, image_path: Any) -> Dict[str, List[Any]]:
    if image_path is None:
        image_path = ""
    if not isinstance(image_path, str):
        raise TypeError(f"UniME-V2 image path must be a string, got {type(image_path)}")
    full_path = os.path.join(image_root, image_path) if image_path else ""
    return {"bytes": [None], "paths": [full_path], "resolutions": [None]}


def _qwen_text(text: Any, model_backbone: str) -> str:
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise TypeError(f"UniME-V2 text must be a string, got {type(text)}")
    if model_backbone != PHI3V:
        text = text.replace(
            VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]
        )
    return text


def sort_processed_inputs_by_length(processed: Dict[str, Any]):
    """Sort one encoded side by active length and return its inverse order.

    The loss uses the inverse permutation to restore the released UniME-V2
    row-major layout.  Sorting affects only execution order, allowing
    GradCache to pack similarly sized sequences without changing candidates,
    scores, or the objective.
    """
    attention_mask = processed.get("attention_mask")
    if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
        raise TypeError("UniME-V2 length packing requires a rank-2 attention_mask")
    batch_size = int(attention_mask.shape[0])
    lengths = attention_mask.long().sum(dim=1)
    if batch_size == 0 or (lengths <= 0).any():
        raise ValueError("UniME-V2 length packing found an empty sequence batch")
    order = torch.argsort(lengths, descending=True, stable=True)
    restore = torch.argsort(order, stable=True)
    order_list = order.tolist()

    sorted_processed = {}
    for key, value in processed.items():
        if isinstance(value, Tensor):
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(
                    f"UniME-V2 cannot reorder tensor {key!r} with shape "
                    f"{tuple(value.shape)} for batch size {batch_size}"
                )
            sorted_processed[key] = value.index_select(0, order)
        else:
            if len(value) != batch_size:
                raise ValueError(
                    f"UniME-V2 cannot reorder field {key!r} with length "
                    f"{len(value)} for batch size {batch_size}"
                )
            sorted_processed[key] = [value[index] for index in order_list]

    if not torch.equal(
        sorted_processed["attention_mask"].sum(dim=1),
        lengths.index_select(0, order),
    ):
        raise RuntimeError("UniME-V2 length sort corrupted the attention mask")
    return sorted_processed, restore


@dataclass
class UniMEV2DataCollator(MultimodalDataCollator):
    """Expand B original rows to the release's row-major 5B x/y layout."""

    def __call__(self, examples: Sequence[Dict[str, Any]]):
        if self.data_args.unime_v2_num_hard_negatives != 8:
            raise ValueError("UniME-V2 collator requires eight hard negatives")
        image_root = self.data_args.unime_v2_image_dir
        if not image_root:
            raise ValueError("--unime_v2_image_dir is required")

        expanded = []
        score0: List[float] = []
        score1: List[float] = []
        dummy_pairs = 0
        dummy_records = []
        for row in examples:
            row_index = int(row["idx"])
            _validate_source_record(row, row_index, 8)
            x_items, y_items = expand_unime_v2_row(row, 8)

            for position, (x_item, y_item) in enumerate(zip(x_items, y_items)):
                x_item, y_item, is_dummy = _official_pair_or_dummy(x_item, y_item)
                dummy_pairs += int(is_dummy)
                if is_dummy:
                    dummy_records.append((row_index, position))
                x_text, x_image, x_score = x_item
                y_text, y_image, y_score = y_item
                expanded.append(
                    {
                        "idx": row_index * _EXPANSION + position,
                        "query_text": [_qwen_text(x_text, self.model_args.model_backbone)],
                        "query_image": [_image_descriptor(image_root, x_image)],
                        "pos_text": [_qwen_text(y_text, self.model_args.model_backbone)],
                        "pos_image": [_image_descriptor(image_root, y_image)],
                    }
                )
                score0.append(float(x_score))
                score1.append(float(y_score))

        expected = len(examples) * _EXPANSION
        if len(expanded) != expected or len(score0) != expected or len(score1) != expected:
            raise RuntimeError("UniME-V2 row expansion produced an invalid layout")
        if dummy_records:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                "[UNIME_V2_DUMMY_BATCH] "
                f"rank={rank} records={dummy_records} "
                "replacement=both_sides_blank_text scores=0,0",
                flush=True,
            )

        qry_inputs = self._get_batch_inputs(expanded, "query_text", "query_image")
        tgt_inputs = self._get_batch_inputs(expanded, "pos_text", "pos_image")
        process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
        processed_qry = process_fn(
            qry_inputs, processor=self.processor, max_length=self.data_args.max_len
        )
        processed_tgt = process_fn(
            tgt_inputs, processor=self.processor, max_length=self.data_args.max_len
        )
        processed_qry["global_dataset_name"] = ["UniME-V2"] * expected
        processed_tgt["global_dataset_name"] = ["UniME-V2"] * expected

        processed_qry, qry_restore_order = sort_processed_inputs_by_length(
            processed_qry
        )
        processed_tgt, tgt_restore_order = sort_processed_inputs_by_length(
            processed_tgt
        )

        if not getattr(self, "_unime_layout_logged", False):
            raw_fingerprint = [
                {
                    "idx": int(row["idx"]),
                    "query": row["query_text"],
                    "positive": row["pos_text"],
                    "negative_scores": row["hard_negatives_scores"][:8],
                }
                for row in examples
            ]
            query_image_count = sum(bool(item["query_image"][0]["paths"][0]) for item in expanded)
            target_image_count = sum(bool(item["pos_image"][0]["paths"][0]) for item in expanded)
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                "[UNIME_V2_COLLATOR_OK] "
                f"rank={rank} original_rows={len(examples)} expanded_x={len(expanded)} "
                f"expanded_y={len(expanded)} interval=5 candidates_per_query=9 "
                "candidate_order=positive,hn0,hn1,hn2,hn3,hn4,hn5,hn6,hn7 "
                f"query_image_pairs={query_image_count}/{expected} "
                f"target_image_pairs={target_image_count}/{expected} "
                f"dummy_pairs={dummy_pairs}/{expected} "
                f"rows_sha256={_sha256_json(raw_fingerprint)} "
                f"first_indices={[int(row['idx']) for row in examples[:16]]} "
                f"qry_shape={tuple(processed_qry['input_ids'].shape)} "
                f"tgt_shape={tuple(processed_tgt['input_ids'].shape)}",
                flush=True,
            )
            self._unime_layout_logged = True

        return (
            processed_qry,
            processed_tgt,
            torch.tensor(score0, dtype=torch.float32),
            torch.tensor(score1, dtype=torch.float32),
            qry_restore_order,
            tgt_restore_order,
        )


def unime_v2_symmetric_kl(
    model_scores: Tensor, teacher_scores: Tensor, temperature: float
):
    """The release's symmetric KL, kept as a separately testable primitive."""
    if model_scores.shape != teacher_scores.shape:
        raise ValueError(
            f"UniME-V2 score shapes differ: model={model_scores.shape}, teacher={teacher_scores.shape}"
        )
    teacher_prob = F.softmax(teacher_scores / temperature, dim=-1)
    model_prob = F.softmax(model_scores / temperature, dim=-1)
    log_teacher = torch.log(teacher_prob + 1e-8)
    log_model = torch.log(model_prob + 1e-8)
    teacher_to_model = F.kl_div(
        log_model, teacher_prob, reduction="batchmean", log_target=False
    )
    model_to_teacher = F.kl_div(
        log_teacher, model_prob, reduction="batchmean", log_target=False
    )
    return 0.5 * (teacher_to_model + model_to_teacher), teacher_to_model, model_to_teacher


class UniMEV2SoftDistributionLoss:
    def __init__(self, temperature: float, scale_loss: bool):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"UniME-V2 temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.scale_loss = bool(scale_loss)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.calls = 0

    def _gather(self, tensor: Tensor) -> Tensor:
        if self.world_size == 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        # Restore the local graph exactly as UniME-V2's released loss does.
        gathered[self.rank] = tensor
        return torch.cat(gathered, dim=0)

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        score0: Tensor,
        score1: Tensor,
        qry_restore_order: Tensor = None,
        tgt_restore_order: Tensor = None,
        **kwargs,
    ):
        self.calls += 1
        if qry_restore_order is not None or tgt_restore_order is not None:
            if qry_restore_order is None or tgt_restore_order is None:
                raise ValueError("UniME-V2 requires both inverse permutations")
            qry_restore_order = qry_restore_order.to(device=x.device, dtype=torch.long)
            tgt_restore_order = tgt_restore_order.to(device=y.device, dtype=torch.long)
            expected_order = torch.arange(x.size(0), device=x.device)
            if (
                qry_restore_order.shape != expected_order.shape
                or tgt_restore_order.shape != expected_order.shape
                or not torch.equal(torch.sort(qry_restore_order).values, expected_order)
                or not torch.equal(torch.sort(tgt_restore_order).values, expected_order)
            ):
                raise ValueError("UniME-V2 received an invalid inverse permutation")
            x = x.index_select(0, qry_restore_order)
            y = y.index_select(0, tgt_restore_order)
        x = self._gather(x)
        y = self._gather(y)
        score0 = self._gather(score0.to(device=x.device, dtype=torch.float32))
        score1 = self._gather(score1.to(device=x.device, dtype=torch.float32))

        if x.ndim != 2 or y.shape != x.shape or x.size(0) % _EXPANSION:
            raise RuntimeError(
                f"Invalid UniME-V2 representation layout: x={tuple(x.shape)} y={tuple(y.shape)}"
            )
        if score0.numel() != x.size(0) or score1.numel() != x.size(0):
            raise RuntimeError(
                "UniME-V2 score/representation counts differ: "
                f"x={x.size(0)} score0={score0.numel()} score1={score1.numel()}"
            )

        query = x[::_EXPANSION]
        positive = y[::_EXPANSION]
        hard = []
        for position in range(1, _EXPANSION):
            hard.extend((x[position::_EXPANSION], y[position::_EXPANSION]))
        candidates = torch.cat((positive.unsqueeze(1), torch.stack(hard, dim=1)), dim=1)
        model_scores = torch.bmm(query.unsqueeze(1), candidates.transpose(1, 2)).squeeze(1)

        teacher_columns = []
        for position in range(_EXPANSION):
            teacher_columns.extend(
                (score0[position::_EXPANSION], score1[position::_EXPANSION])
            )
        teacher_scores = torch.stack(teacher_columns, dim=1)[:, 1:]
        loss, kl_teacher_model, kl_model_teacher = unime_v2_symmetric_kl(
            model_scores, teacher_scores, self.temperature
        )

        debug_interval = max(1, int(os.environ.get("B3_DEBUG_INTERVAL", "100")))
        should_log = self.rank == 0 and (
            self.calls == 1
            or (
                os.environ.get("B3_DEBUG", "0") == "1"
                and self.calls % debug_interval == 0
            )
        )
        if should_log:
            with torch.no_grad():
                model_prob = F.softmax(model_scores.float() / self.temperature, dim=-1)
                teacher_prob = F.softmax(teacher_scores.float() / self.temperature, dim=-1)
                positive_sim = model_scores[:, 0].float()
                negative_sim = model_scores[:, 1:].float()
                positive_top1 = (model_scores.argmax(dim=1) == 0).float().mean()
                teacher_top1_agreement = (
                    model_scores.argmax(dim=1) == teacher_scores.argmax(dim=1)
                ).float().mean()
                teacher_positive_top1 = (teacher_scores.argmax(dim=1) == 0).float().mean()
                model_entropy = -(model_prob * torch.log(model_prob + 1e-8)).sum(dim=1).mean()
                teacher_entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean()
                print(
                    "[UNIME_V2_LOSS] "
                    f"step={self.calls} global_queries={query.size(0)} "
                    f"global_expanded_pairs={x.size(0)} candidates_per_query=9 "
                    f"rep_dtype={x.dtype} score_dtype={teacher_scores.dtype} "
                    f"temperature={self.temperature:.8g} "
                    f"pos_sim={positive_sim.mean().item():.8g} "
                    f"hard_sim_mean={negative_sim.mean().item():.8g} "
                    f"hard_sim_max_mean={negative_sim.max(dim=1).values.mean().item():.8g} "
                    f"teacher_pos_score={teacher_scores[:, 0].mean().item():.8g} "
                    f"teacher_hard_score={teacher_scores[:, 1:].mean().item():.8g} "
                    f"positive_top1={positive_top1.item():.8g} "
                    f"teacher_positive_top1={teacher_positive_top1.item():.8g} "
                    f"teacher_top1_agreement={teacher_top1_agreement.item():.8g} "
                    f"model_entropy={model_entropy.item():.8g} "
                    f"teacher_entropy={teacher_entropy.item():.8g} "
                    f"kl_teacher_model={kl_teacher_model.detach().float().item():.8g} "
                    f"kl_model_teacher={kl_model_teacher.detach().float().item():.8g} "
                    f"loss={loss.detach().float().item():.8g}",
                    flush=True,
                )
        return loss * self.world_size if self.scale_loss else loss


class UniMEV2Trainer(GradCacheLateProcessTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_args.negative_strategy != _METHOD:
            raise ValueError("UniMEV2Trainer requires --negative_strategy unime_v2")
        loss_fn = UniMEV2SoftDistributionLoss(
            temperature=self.model.temperature,
            scale_loss=self.is_ddp,
        )
        self.gc.loss_fn = loss_fn
        self.gc.length_aware_packing = True
        self._unime_config = {
            "method": _METHOD,
            "objective": "symmetric_kl_teacher_and_model_distributions",
            "teacher_file": os.path.basename(self.data_args.unime_v2_data_path),
            "teacher_checkpoint_family": "Qwen25VL_7B",
            "dataset_fingerprint": getattr(
                self.data_args, "_unime_v2_dataset_fingerprint", None
            ),
            "image_root": os.path.abspath(self.data_args.unime_v2_image_dir),
            "hard_negatives": 8,
            "expansion_interval": 5,
            "candidates_per_query": 9,
            "candidate_order": ["positive", *[f"hard_negative_{i}" for i in range(8)]],
            "temperature": float(self.model.temperature),
            "pooling": self.model_args.pooling,
            "normalize": bool(self.model_args.normalize),
            "model": self.model_args.model_name,
            "qwen2_model_impl": self.model_args.qwen2_model_impl,
            "qwen2_attention_impl": self.model_args.qwen2_b3_attention_impl,
            "qwen2_legacy_position_ids": bool(
                self.model_args.qwen2_b3_legacy_position_ids
            ),
            "per_device_original_rows": int(self.args.per_device_train_batch_size),
            "per_device_expanded_pairs": int(self.args.per_device_train_batch_size) * _EXPANSION,
            "length_aware_gradcache_packing": True,
            "restore_official_order_before_loss": True,
            "world_size": self._dist_loss_scale_factor,
            "learning_rate": float(self.args.learning_rate),
            "optimizer": str(self.args.optim),
            "lr_scheduler_type": str(self.args.lr_scheduler_type),
            "warmup_steps": int(self.args.warmup_steps),
            "max_steps": int(self.args.max_steps),
            "seed": int(self.args.seed),
            "bf16": bool(self.args.bf16),
            "gradient_accumulation_steps": int(
                self.args.gradient_accumulation_steps
            ),
            "lora_r": int(self.model_args.lora_r),
            "lora_alpha": int(self.model_args.lora_alpha),
            "lora_dropout": float(self.model_args.lora_dropout),
            "lora_target_modules": self.model_args.lora_target_modules,
            "gc_q_chunk_size": int(self.args.gc_q_chunk_size),
            "gc_p_chunk_size": int(self.args.gc_p_chunk_size),
            "gc_dynamic_limit": int(self.args.gc_dynamic_limit),
            "resize_use_processor": bool(self.data_args.resize_use_processor),
            "resize_min_pixels": int(self.data_args.resize_min_pixels),
            "resize_max_pixels": int(self.data_args.resize_max_pixels),
            "ddp_broadcast_buffers": bool(self.args.ddp_broadcast_buffers),
            "ddp_find_unused_parameters": bool(
                self.args.ddp_find_unused_parameters
            ),
        }
        # Preserve the serialized Qwen2-VL contract byte-for-byte so existing
        # Qwen2 UniME-V2 checkpoints remain resumable. Qwen3.5 records its
        # native runtime instead of meaningless qwen2_* compatibility fields.
        if self.model_args.model_backbone != "qwen2_vl":
            self._unime_config.pop("qwen2_model_impl")
            self._unime_config.pop("qwen2_attention_impl")
            self._unime_config.pop("qwen2_legacy_position_ids")
            self._unime_config.update(
                {
                    "model_backbone": self.model_args.model_backbone,
                    "checkpoint_runtime": "transformers_native_5.9_verified_reload",
                }
            )
        print_master(
            "[UNIME_V2_REPRO_CONFIG] "
            f"sha256={_sha256_json(self._unime_config)} "
            f"values={json.dumps(self._unime_config, sort_keys=True, separators=(',', ':'))}"
        )
        self._validate_resume_config()

    def _get_train_sampler(self, train_dataset=None):
        # MMEBTrainer's historical override has an unreachable RandomSampler
        # return. UniME-V2's released Trainer uses Transformers' ordinary
        # RandomSampler on top of the fixed seed-42 dataset shuffle.
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        return RandomSampler(dataset)

    def _on_reps_captured(self, all_reps):
        """Expose a truthful 1-query/9-target view to existing diagnostics."""
        x, y = all_reps
        if getattr(self, "_current_qry_restore_order", None) is not None:
            x = x.index_select(
                0, self._current_qry_restore_order.to(device=x.device, dtype=torch.long)
            )
            y = y.index_select(
                0, self._current_tgt_restore_order.to(device=y.device, dtype=torch.long)
            )
        if x.shape != y.shape or x.size(0) % _EXPANSION:
            raise RuntimeError(
                f"Invalid UniME-V2 captured reps: x={tuple(x.shape)} y={tuple(y.shape)}"
            )
        query = x[::_EXPANSION]
        hard = []
        for position in range(1, _EXPANSION):
            hard.extend((x[position::_EXPANSION], y[position::_EXPANSION]))
        candidates = torch.cat(
            (y[::_EXPANSION].unsqueeze(1), torch.stack(hard, dim=1)), dim=1
        )
        query_finite = int(torch.isfinite(query).sum().item())
        candidate_finite = int(torch.isfinite(candidates).sum().item())
        if query_finite != query.numel() or candidate_finite != candidates.numel():
            raise FloatingPointError(
                "UniME-V2 captured non-finite representations: "
                f"query={query_finite}/{query.numel()} "
                f"candidates={candidate_finite}/{candidates.numel()}"
            )
        # The established callback now sees target_per_query=9 with the aligned
        # positive first. Its cross-query negatives are a comparability-only
        # diagnostic; UNIME_V2_LOSS reports the actual per-query 9-way objective.
        self._monitor_reps = [
            query.detach(), candidates.reshape(-1, candidates.size(-1)).detach()
        ]
        if not getattr(self, "_unime_reps_logged", False):
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                "[UNIME_V2_REPS_OK] "
                f"rank={rank} local_original_queries={query.size(0)} "
                f"local_candidates={candidates.size(0) * candidates.size(1)} "
                "target_per_query=9 aligned_positive_offset=0 "
                f"hidden_size={query.size(1)} "
                f"query_norm_mean={query.float().norm(dim=-1).mean().item():.8g} "
                f"candidate_norm_mean={candidates.float().norm(dim=-1).mean().item():.8g}",
                flush=True,
            )
            self._unime_reps_logged = True

    def _resume_dir(self):
        value = str(self.args.resume_from)
        if value == "auto":
            return find_latest_checkpoint(self.args.output_dir)
        if value.isdigit():
            path = os.path.join(self.args.output_dir, f"checkpoint-{value}")
            return path if os.path.isdir(path) else None
        return None

    def _validate_resume_config(self):
        checkpoint = self._resume_dir()
        if checkpoint is None:
            return
        config_path = Path(checkpoint) / _CONFIG_FILE
        if not config_path.is_file():
            raise RuntimeError(
                f"Refusing to resume UniME-V2 without {_CONFIG_FILE}: {checkpoint}"
            )
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != self._unime_config:
            raise RuntimeError(
                f"UniME-V2 resume configuration differs: saved={saved} current={self._unime_config}"
            )
        print_master(f"[UNIME_V2_RESUME_CONFIG_OK] checkpoint={checkpoint}")

    def training_step(self, model, inputs, *args, **kwargs) -> Tensor:
        model.train()
        (
            queries,
            targets,
            score0,
            score1,
            qry_restore_order,
            tgt_restore_order,
        ) = inputs
        self._current_qry_restore_order = qry_restore_order
        self._current_tgt_restore_order = tgt_restore_order
        distributed = dist.is_initialized() and dist.get_world_size() > 1
        self.gc.models = [model, model]
        loss = self.gc(
            {"qry": queries},
            {"tgt": targets},
            score0=score0,
            score1=score1,
            qry_restore_order=qry_restore_order,
            tgt_restore_order=tgt_restore_order,
            no_sync_except_last=distributed,
        )
        return loss / self._dist_loss_scale_factor

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / _CONFIG_FILE).write_text(
            json.dumps(self._unime_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if _is_main_process():
            print(f"[UNIME_V2_CONFIG_SAVED] path={output / _CONFIG_FILE}", flush=True)
