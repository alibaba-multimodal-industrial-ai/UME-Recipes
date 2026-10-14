"""Isolated trainer for the 200-step positive-reliability probe.

Nothing in the normal training or evaluation entry points imports this file.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.training.data.collator import MultimodalDataCollator
from src.training.loss import DistributedContrastiveLoss
from src.training.positive_reliability import (
    positive_reliability_weighted_info_nce,
)
from src.training.trainer import GradCacheLateProcessTrainer


_CASE_IDS_KEY = "_positive_reliability_case_ids"


class PositiveReliabilityDataCollator(MultimodalDataCollator):
    """Carry stable dataset row IDs to the probe loss."""

    def __call__(self, examples):
        case_ids = [
            f"{example['global_dataset_name']}:{int(example['idx'])}"
            for example in examples
        ]
        queries, targets = super().__call__(examples)
        queries[_CASE_IDS_KEY] = case_ids
        return queries, targets


class _PositiveReliabilityLossBase:
    def _configure_probe(self, *, temperature, min_weight, scale, log_path):
        self.temperature = float(temperature)
        self.min_weight = float(min_weight)
        self.scale = float(scale)
        self.log_path = Path(log_path)
        self.debug_enabled = os.environ.get(
            "POS_REL_DEBUG", os.environ.get("B3_DEBUG", "1")
        ) == "1"
        self.step = None
        self.local_case_ids = None

    def set_batch_context(self, *, step, local_case_ids):
        self.step = int(step)
        self.local_case_ids = list(local_case_ids)

    def _global_case_ids(self):
        if self.local_case_ids is None:
            raise RuntimeError("positive-reliability batch context was not set")
        return self.local_case_ids

    def _write_diagnostics(self, case_ids, diagnostics):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if len(case_ids) != diagnostics["weight"].numel():
            raise RuntimeError(
                "case-ID/representation count mismatch: "
                f"ids={len(case_ids)} reps={diagnostics['weight'].numel()}"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        values = {key: value.detach().cpu() for key, value in diagnostics.items()}
        negative_count = len(case_ids) - 1
        with self.log_path.open("a", encoding="utf-8") as handle:
            for row, case_id in enumerate(case_ids):
                dataset, idx = case_id.rsplit(":", 1)
                record = {
                    "step": self.step,
                    "case_id": case_id,
                    "dataset": dataset,
                    "idx": int(idx),
                    "positive_similarity": float(values["positive_similarity"][row]),
                    "negative_count": negative_count,
                    "violation_count": int(values["violation_count"][row]),
                    "violation_ratio": float(values["violation_ratio"][row]),
                    "violation_magnitude": float(values["violation_magnitude"][row]),
                    "anomaly_score": float(values["score"][row]),
                    "reliability_weight": float(values["weight"][row]),
                    "per_case_loss": float(values["per_case_loss"][row]),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.debug_enabled:
            self._print_step_debug(case_ids, values)

    @staticmethod
    def _quantile(values, q):
        return torch.quantile(values.float(), q).item()

    def _print_step_debug(self, case_ids, values):
        ratio = values["violation_ratio"].float()
        magnitude = values["violation_magnitude"].float()
        score = values["score"].float()
        weight = values["weight"].float()
        floor_mask = weight <= self.min_weight + 1e-7

        print(
            "[B3_DEBUG][POS_REL] "
            f"step={self.step} global_cases={len(case_ids)} "
            f"ratio_mean/p50/p90/p99={ratio.mean().item():.8g}/"
            f"{self._quantile(ratio, 0.50):.8g}/"
            f"{self._quantile(ratio, 0.90):.8g}/"
            f"{self._quantile(ratio, 0.99):.8g} "
            f"magnitude_mean/p50/p90/p99={magnitude.mean().item():.8g}/"
            f"{self._quantile(magnitude, 0.50):.8g}/"
            f"{self._quantile(magnitude, 0.90):.8g}/"
            f"{self._quantile(magnitude, 0.99):.8g} "
            f"score_mean/p50/p90/p99={score.mean().item():.8g}/"
            f"{self._quantile(score, 0.50):.8g}/"
            f"{self._quantile(score, 0.90):.8g}/"
            f"{self._quantile(score, 0.99):.8g} "
            f"weight_mean/p50/p10/p01={weight.mean().item():.8g}/"
            f"{self._quantile(weight, 0.50):.8g}/"
            f"{self._quantile(weight, 0.10):.8g}/"
            f"{self._quantile(weight, 0.01):.8g} "
            f"floor_cases={int(floor_mask.sum().item())}/{len(case_ids)}",
            flush=True,
        )

    def _weighted_loss(self, query, target, case_ids):
        loss, diagnostics = positive_reliability_weighted_info_nce(
            query,
            target,
            temperature=self.temperature,
            min_weight=self.min_weight,
            scale=self.scale,
        )
        self._write_diagnostics(case_ids, diagnostics)
        return loss


class PositiveReliabilitySimpleLoss(_PositiveReliabilityLossBase):
    def __init__(self, *, temperature, min_weight, scale, log_path):
        self._configure_probe(
            temperature=temperature,
            min_weight=min_weight,
            scale=scale,
            log_path=log_path,
        )

    def __call__(self, query, target, **kwargs):
        return self._weighted_loss(query, target, self._global_case_ids())


class PositiveReliabilityDistributedLoss(
    _PositiveReliabilityLossBase, DistributedContrastiveLoss
):
    def __init__(self, *, temperature, min_weight, scale, log_path):
        DistributedContrastiveLoss.__init__(self, temperature=temperature)
        self._configure_probe(
            temperature=temperature,
            min_weight=min_weight,
            scale=scale,
            log_path=log_path,
        )

    def _global_case_ids(self):
        if self.local_case_ids is None:
            raise RuntimeError("positive-reliability batch context was not set")
        gathered = [None for _ in range(self.word_size)]
        dist.all_gather_object(gathered, self.local_case_ids)
        return [case_id for rank_ids in gathered for case_id in rank_ids]

    def __call__(self, query, target, **kwargs):
        global_query = self.gather_tensor(query)
        global_target = self.gather_tensor(target)
        loss = self._weighted_loss(
            global_query, global_target, self._global_case_ids()
        )
        return loss * self.word_size if self.scale_loss else loss


class PositiveReliabilityTrainer(GradCacheLateProcessTrainer):
    """GradCache trainer whose only changed component is the loss callable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_args.negative_strategy != "vlm2vec_inbatch":
            raise ValueError("positive reliability requires vlm2vec_inbatch")
        if self.args.embedding_gradient_strategy != "info_nce":
            raise ValueError("positive reliability requires info_nce GradCache")

        scale = float(os.environ.get("POS_REL_SCALE", "4.0"))
        min_weight = float(os.environ.get("POS_REL_MIN_WEIGHT", "0.1"))
        log_path = os.environ.get(
            "POS_REL_LOG_PATH",
            os.path.join(self.args.output_dir, "positive_reliability_cases.jsonl"),
        )
        loss_class = (
            PositiveReliabilityDistributedLoss
            if self.is_ddp
            else PositiveReliabilitySimpleLoss
        )
        self.positive_reliability_loss = loss_class(
            temperature=self.model.temperature,
            min_weight=min_weight,
            scale=scale,
            log_path=log_path,
        )
        self.gc.loss_fn = self.positive_reliability_loss
        self._positive_reliability_config = {
            "method": "detached_positive_reliability_weighted_info_nce",
            "formula": "weight=max(exp(-scale*ratio*magnitude/temperature), min_weight)",
            "scale": scale,
            "min_weight": min_weight,
            "temperature": float(self.model.temperature),
            "normalization": "sum(weight*loss)/sum(weight)",
            "weight_gradient": "detached",
            "source_checkpoint": os.environ.get("POS_REL_SOURCE_CHECKPOINT"),
        }

    def training_step(self, model, inputs, *args, **kwargs):
        queries, _ = inputs
        case_ids = queries.pop(_CASE_IDS_KEY)
        self.positive_reliability_loss.set_batch_context(
            step=self.state.global_step + 1,
            local_case_ids=case_ids,
        )
        return super().training_step(model, inputs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        config_path = Path(output_dir) / "positive_reliability_config.json"
        config_path.write_text(
            json.dumps(self._positive_reliability_config, indent=2) + "\n",
            encoding="utf-8",
        )
