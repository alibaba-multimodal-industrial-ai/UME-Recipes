"""Isolated Qwen2-VL in-batch training with GloFND filtering."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from src.training.data.collator import MultimodalDataCollator
from src.training.glofnd_core import (
    GloFNDHyperparameters,
    PerAnchorGloFNDState,
    glofnd_filtered_info_nce,
)
from src.training.loss import DistributedContrastiveLoss
from src.training.trainer import GradCacheLateProcessTrainer


_ANCHOR_IDS_KEY = "_glofnd_anchor_ids"
_STATE_FILE = "glofnd_state.pt"
_CONFIG_FILE = "glofnd_training_config.json"


def build_anchor_id_layout(dataset_config_path: str) -> dict:
    """Build stable IDs from dataset name plus its configured row index."""
    config_path = Path(dataset_config_path)
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    if not isinstance(config, dict) or not config:
        raise RuntimeError(f"Invalid dataset config: {config_path}")

    offset = 0
    datasets = []
    seen_names = set()
    for config_key, spec in config.items():
        if not isinstance(spec, dict):
            raise RuntimeError(f"Invalid dataset entry {config_key!r}")
        parser = str(spec.get("dataset_parser", "mmeb"))
        subset = str(spec.get("subset_name", config_key))
        global_name = f"{parser}/{subset}"
        capacity = int(spec.get("num_sample_per_subset", 0))
        if capacity <= 0:
            raise RuntimeError(
                f"GloFND requires a positive num_sample_per_subset for {config_key}"
            )
        if global_name in seen_names:
            raise RuntimeError(f"Duplicate GloFND dataset identity: {global_name}")
        seen_names.add(global_name)
        datasets.append(
            {
                "config_key": str(config_key),
                "global_dataset_name": global_name,
                "offset": offset,
                "capacity": capacity,
            }
        )
        offset += capacity

    layout = {
        "version": 1,
        "dataset_config": str(config_path.resolve()),
        "dataset_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "datasets": datasets,
        "num_anchors": offset,
    }
    canonical = json.dumps(layout, sort_keys=True, separators=(",", ":"))
    layout["layout_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return layout


class GloFNDDataCollator(MultimodalDataCollator):
    """Attach stable per-query IDs while preserving the proven collator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.glofnd_layout = build_anchor_id_layout(self.data_args.dataset_config)
        self._dataset_ranges = {
            item["global_dataset_name"]: (item["offset"], item["capacity"])
            for item in self.glofnd_layout["datasets"]
        }
        self._glofnd_ids_logged = False

    def __call__(self, examples):
        anchor_ids = []
        for example in examples:
            dataset_name = str(example["global_dataset_name"])
            if dataset_name not in self._dataset_ranges:
                raise KeyError(
                    f"Dataset {dataset_name!r} is absent from the GloFND ID layout"
                )
            offset, capacity = self._dataset_ranges[dataset_name]
            row_index = int(example["idx"])
            if row_index < 0 or row_index >= capacity:
                raise IndexError(
                    f"GloFND row index outside configured capacity for {dataset_name}: "
                    f"idx={row_index}, capacity={capacity}"
                )
            anchor_ids.append(offset + row_index)

        queries, targets = super().__call__(examples)
        queries[_ANCHOR_IDS_KEY] = anchor_ids
        if not self._glofnd_ids_logged:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                "[GLOFND_ANCHOR_IDS_OK] "
                f"rank={rank} local_pairs={len(anchor_ids)} "
                f"unique={len(set(anchor_ids))}/{len(anchor_ids)} "
                f"capacity={self.glofnd_layout['num_anchors']} "
                f"layout_sha256={self.glofnd_layout['layout_sha256']} "
                f"first_ids={anchor_ids[:16]}",
                flush=True,
            )
            self._glofnd_ids_logged = True
        return queries, targets


def _sample_quantiles(values: torch.Tensor, probabilities=(0.1, 0.5, 0.9, 0.99)):
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return [float("nan") for _ in probabilities]
    stride = max(1, math.ceil(values.numel() / 65536))
    sample = values[::stride]
    return [float(torch.quantile(sample, p).item()) for p in probabilities]


class _GloFNDLossBase:
    def _configure(self, *, temperature, state, log_path, debug_interval):
        self.temperature = float(temperature)
        self.glofnd_state = state
        self.log_path = Path(log_path)
        self.debug_interval = max(1, int(debug_interval))
        self.step = None
        self.local_anchor_ids = None
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._first_contract_logged = False
        self._embedding_norms = None

    def set_batch_context(self, step, local_anchor_ids):
        self.step = int(step)
        self.local_anchor_ids = torch.as_tensor(local_anchor_ids, dtype=torch.long)

    def _global_anchor_ids(self, device):
        if self.local_anchor_ids is None:
            raise RuntimeError("GloFND batch context was not set before the loss")
        local_ids = self.local_anchor_ids.to(device=device)
        if not dist.is_initialized():
            return local_ids
        sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(dist.get_world_size())]
        local_size = torch.tensor([local_ids.numel()], device=device, dtype=torch.long)
        dist.all_gather(sizes, local_size)
        if len({int(size.item()) for size in sizes}) != 1:
            raise RuntimeError(f"GloFND requires equal local batch sizes, got {[int(x.item()) for x in sizes]}")
        gathered = [torch.empty_like(local_ids) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_ids)
        return torch.cat(gathered, dim=0)

    def _state_sync_error(self, device, anchor_ids):
        initialized = self.glofnd_state.initialized
        global_signature = torch.tensor(
            [
                float(self.glofnd_state.threshold[initialized].sum().item()),
                float(self.glofnd_state.first_moment[initialized].sum().item()),
                float(self.glofnd_state.second_moment[initialized].sum().item()),
                float(self.glofnd_state.visit_count.sum().item()),
                float(self.glofnd_state.initialized_total),
            ],
            dtype=torch.float64,
            device=device,
        )
        if not dist.is_initialized():
            return 0.0
        ids_cpu = anchor_ids.detach().to(device="cpu", dtype=torch.long)
        batch_signature = torch.stack(
            [
                self.glofnd_state.threshold[ids_cpu].double(),
                self.glofnd_state.first_moment[ids_cpu].double(),
                self.glofnd_state.second_moment[ids_cpu].double(),
                self.glofnd_state.visit_count[ids_cpu].double(),
            ],
            dim=1,
        ).to(device)
        signature = torch.cat([global_signature, batch_signature.reshape(-1)])
        gathered = [
            torch.empty_like(signature) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered, signature)
        stacked = torch.stack(gathered)
        error = float((stacked - stacked[0]).abs().max().item())
        if error != 0.0:
            raise RuntimeError(
                f"GloFND per-anchor state diverged across ranks; max_delta={error}"
            )
        return error

    def _loss(self, query, target, anchor_ids):
        if self.step is None:
            raise RuntimeError("GloFND step was not set before the loss")
        if anchor_ids.numel() != query.shape[0]:
            raise RuntimeError(
                f"GloFND global ID/representation mismatch: {anchor_ids.numel()} != {query.shape[0]}"
            )
        if self._embedding_norms is None:
            query_norm = query.detach().float().norm(dim=1)
            target_norm = target.detach().float().norm(dim=1)
            query_error = float((query_norm - 1.0).abs().max().item())
            target_error = float((target_norm - 1.0).abs().max().item())
            if query_error > 1e-2 or target_error > 1e-2:
                raise RuntimeError(
                    "GloFND requires actually normalized embeddings, not only "
                    f"normalize=True; max errors are query={query_error:.8g}, "
                    f"target={target_error:.8g}"
                )
            self._embedding_norms = {
                "query_norm_mean": float(query_norm.mean().item()),
                "target_norm_mean": float(target_norm.mean().item()),
                "query_norm_max_error": query_error,
                "target_norm_max_error": target_error,
            }
        loss, diagnostics = glofnd_filtered_info_nce(
            query, target, anchor_ids, self.glofnd_state, self.temperature
        )
        fallback_count = int(diagnostics["empty_filter_fallback"].sum().item())
        if fallback_count and self.rank == 0:
            fallback_rows = torch.nonzero(
                diagnostics["empty_filter_fallback"], as_tuple=False
            ).flatten()
            print(
                "[GLOFND_EMPTY_NEGATIVE_SAFETY] "
                f"step={self.step} anchors={fallback_count} "
                f"global_rows={fallback_rows[:16].tolist()} "
                f"proposed_threshold_min="
                f"{float(diagnostics['proposed_threshold'][fallback_rows].min().item())} "
                f"recalibrated_threshold_min="
                f"{float(diagnostics['threshold'][fallback_rows].min().item())}",
                flush=True,
            )
        should_log = self.step == 1 or (
            os.environ.get("GLOFND_DEBUG", os.environ.get("B3_DEBUG", "1")) == "1"
            and self.step % self.debug_interval == 0
        )
        if should_log:
            sync_error = self._state_sync_error(query.device, anchor_ids)
            if self.rank == 0:
                similarity = diagnostics["similarity"]
                negative_values = similarity.masked_select(diagnostics["negative_mask"])
                filtered_values = similarity.masked_select(diagnostics["filtered_mask"])
                retained_values = similarity.masked_select(
                    diagnostics["negative_mask"] & ~diagnostics["filtered_mask"]
                )
                threshold_q = _sample_quantiles(diagnostics["threshold"])
                filtered_q = _sample_quantiles(diagnostics["filtered_count"].float())
                visit_q = _sample_quantiles(diagnostics["visit_count"].float())
                removed_mass_q = _sample_quantiles(
                    diagnostics["removed_negative_probability_mass"]
                )
                record = {
                    "step": self.step,
                    "global_pairs": int(query.shape[0]),
                    "temperature": self.temperature,
                    "alpha": self.glofnd_state.hyperparameters.alpha,
                    "new_anchors": int(diagnostics["new_anchor"].sum().item()),
                    "initialized_anchors": self.glofnd_state.initialized_total,
                    "anchor_capacity": self.glofnd_state.num_anchors,
                    "initialized_coverage": self.glofnd_state.initialized_total / self.glofnd_state.num_anchors,
                    "threshold_mean": float(diagnostics["threshold"].mean().item()),
                    "threshold_p10_p50_p90_p99": threshold_q,
                    "batch_quantile_mean": float(diagnostics["batch_quantile"].mean().item()),
                    "batch_quantile_min": float(diagnostics["batch_quantile"].min().item()),
                    "batch_quantile_max": float(diagnostics["batch_quantile"].max().item()),
                    "visit_mean_p10_p50_p90_p99": [float(diagnostics["visit_count"].float().mean().item()), *visit_q],
                    "filtered_per_anchor_mean": float(diagnostics["filtered_count"].float().mean().item()),
                    "filtered_per_anchor_p10_p50_p90_p99": filtered_q,
                    "anchors_filtering_zero": int((diagnostics["filtered_count"] == 0).sum().item()),
                    "empty_filter_fallback_anchors": fallback_count,
                    "empty_filter_fallback_ratio": float(diagnostics["empty_filter_fallback"].float().mean().item()),
                    "proposed_threshold_min": float(diagnostics["proposed_threshold"].min().item()),
                    "filtered_negative_pairs": int(diagnostics["filtered_count"].sum().item()),
                    "total_negative_pairs": int(diagnostics["negative_mask"].sum().item()),
                    "target_filtered_pairs_from_alpha": float(self.glofnd_state.hyperparameters.alpha * diagnostics["negative_mask"].sum().item()),
                    "empirical_filtered_ratio": float(diagnostics["filtered_count"].sum().item() / diagnostics["negative_mask"].sum().item()),
                    "positive_similarity_mean": float(diagnostics["positive_similarity"].mean().item()),
                    "positive_similarity_min": float(diagnostics["positive_similarity"].min().item()),
                    "positive_similarity_max": float(diagnostics["positive_similarity"].max().item()),
                    "negative_similarity_mean": float(negative_values.mean().item()),
                    "negative_similarity_max": float(negative_values.max().item()),
                    "hardest_retained_similarity": float(retained_values.max().item()),
                    "weakest_filtered_similarity": float(filtered_values.min().item()) if filtered_values.numel() else None,
                    "loss": float(loss.detach().item()),
                    "unfiltered_infonce_loss": float(diagnostics["unfiltered_loss"].item()),
                    "loss_reduction_from_filter": float(diagnostics["unfiltered_loss"].item() - loss.detach().item()),
                    "unfiltered_train_accuracy": float(diagnostics["unfiltered_accuracy"].item()),
                    "filtered_train_accuracy": float(diagnostics["filtered_accuracy"].item()),
                    "removed_negative_probability_mass_mean": float(diagnostics["removed_negative_probability_mass"].mean().item()),
                    "removed_negative_probability_mass_p10_p50_p90_p99": removed_mass_q,
                    "state_sync_max_delta": sync_error,
                    **self._embedding_norms,
                }
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    "[B3_DEBUG][GLOFND] "
                    + " ".join(f"{key}={value}" for key, value in record.items()),
                    flush=True,
                )
        if not self._first_contract_logged:
            print(
                "[GLOFND_LOSS_OK] "
                f"rank={self.rank} global_pairs={query.shape[0]} labels=diagonal "
                f"direction=query_to_target similarity=raw_cosine "
                f"filter_gradient=detached loss_dtype={loss.dtype}",
                flush=True,
            )
            self._first_contract_logged = True
        return loss


class GloFNDSimpleLoss(_GloFNDLossBase):
    def __init__(self, **kwargs):
        self._configure(**kwargs)

    def __call__(self, query, target, **kwargs):
        return self._loss(query, target, self._global_anchor_ids(query.device))


class GloFNDDistributedLoss(_GloFNDLossBase, DistributedContrastiveLoss):
    def __init__(self, **kwargs):
        DistributedContrastiveLoss.__init__(self, temperature=kwargs["temperature"])
        self._configure(**kwargs)

    def __call__(self, query, target, **kwargs):
        global_query = self.gather_tensor(query)
        global_target = self.gather_tensor(target)
        global_ids = self._global_anchor_ids(query.device)
        loss = self._loss(global_query, global_target, global_ids)
        return loss * self.word_size if self.scale_loss else loss


def _state_digest(state: PerAnchorGloFNDState) -> str:
    digest = hashlib.sha256()
    for name in ("threshold", "first_moment", "second_moment", "visit_count", "initialized"):
        tensor = getattr(state, name).contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    staged = destination.with_name(destination.name + ".tmp")
    try:
        with tempfile.NamedTemporaryFile(prefix="glofnd_", suffix=".pt", dir="/tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        local_digest = hashlib.sha256()
        with temporary_path.open("rb") as source, staged.open("wb") as target:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                local_digest.update(chunk)
            target.flush()
        published_digest = hashlib.sha256()
        with staged.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                published_digest.update(chunk)
        if published_digest.digest() != local_digest.digest():
            raise RuntimeError(
                f"GloFND checkpoint publish checksum mismatch: {destination}"
            )
        os.replace(staged, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if staged.exists():
            staged.unlink()


class GloFNDTrainer(GradCacheLateProcessTrainer):
    """GradCache trainer whose only algorithmic change is GloFND filtering."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.model, "model_backbone", None) != "qwen2_vl":
            raise ValueError("This isolated GloFND reproduction requires Qwen2-VL")
        if self.data_args.negative_strategy != "vlm2vec_inbatch":
            raise ValueError("GloFND requires ordinary vlm2vec_inbatch pairs")
        if self.args.embedding_gradient_strategy != "info_nce":
            raise ValueError("GloFND keeps InfoNCE and requires embedding_gradient_strategy=info_nce")
        if not bool(self.model.normalize):
            raise ValueError("GloFND thresholds require normalized cosine embeddings")
        if not isinstance(self.data_collator, GloFNDDataCollator):
            raise TypeError("GloFNDTrainer requires GloFNDDataCollator")

        hyperparameters = GloFNDHyperparameters(
            alpha=float(os.environ.get("GLOFND_ALPHA", "0.0005")),
            threshold_lr=float(os.environ.get("GLOFND_THRESHOLD_LR", "0.05")),
            beta1=float(os.environ.get("GLOFND_BETA1", "0.9")),
            beta2=float(os.environ.get("GLOFND_BETA2", "0.98")),
            eps=float(os.environ.get("GLOFND_EPS", "1e-8")),
        )
        self.glofnd_state = PerAnchorGloFNDState(
            self.data_collator.glofnd_layout["num_anchors"], hyperparameters
        )
        loss_class = GloFNDDistributedLoss if self.is_ddp else GloFNDSimpleLoss
        self.glofnd_loss = loss_class(
            temperature=self.model.temperature,
            state=self.glofnd_state,
            log_path=os.environ.get(
                "GLOFND_LOG_PATH",
                os.path.join(self.args.output_dir, "glofnd_steps.jsonl"),
            ),
            debug_interval=int(os.environ.get("GLOFND_DEBUG_INTERVAL", os.environ.get("B3_DEBUG_INTERVAL", "100"))),
        )
        self.gc.loss_fn = self.glofnd_loss
        self._glofnd_loaded_checkpoint = None
        self._glofnd_config = {
            "method": "glofnd_false_negative_detection",
            "reference_repository": "https://github.com/vibalcam/GloFND.git",
            "reference_commit": "b7f1dd1ccec7bb3006f3ac7cd62ffab6bf9e8a7e",
            "objective": "one_direction_query_to_target_infonce",
            "threshold_update": "released_glofnd_stochastic_subgradient_adam",
            "threshold_initialization": "per_anchor_lazy_current_global_batch_1_minus_alpha_quantile_applied_immediately",
            "bias_correction_clock": "per_anchor_visit_count_initialization_is_visit_1",
            "similarity": "raw_cosine_from_normalized_embeddings",
            "mask_rule": "filter_negative_if_similarity_gt_lambda",
            "mask_gradient": "detached",
            "positive_filtering": False,
            "threshold_warmup_steps": 0,
            "adam_updated_threshold_domain": [-1.0, 1.0],
            "lazy_quantile_initialization_clamped": False,
            "empty_negative_safety": "recalibrate_only_degenerate_rows_to_current_global_batch_1_minus_alpha_quantile",
            "empty_negative_safety_scope": "only_if_adam_proposal_would_filter_all_observed_negatives",
            "temperature": float(self.model.temperature),
            "hyperparameters": vars(hyperparameters),
            "anchor_layout": self.data_collator.glofnd_layout,
        }
        if self.is_world_process_zero():
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            config_json = json.dumps(
                self._glofnd_config, sort_keys=True, separators=(",", ":")
            )
            (output_dir / _CONFIG_FILE).write_text(
                json.dumps(self._glofnd_config, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                "[GLOFND_REPRO_CONFIG] "
                f"sha256={hashlib.sha256(config_json.encode('utf-8')).hexdigest()} "
                f"reference_commit={self._glofnd_config['reference_commit']} "
                f"objective={self._glofnd_config['objective']} "
                f"initialization={self._glofnd_config['threshold_initialization']} "
                f"layout_sha256={self.data_collator.glofnd_layout['layout_sha256']}",
                flush=True,
            )
        print(
            "[GLOFND_CONFIG_OK] "
            f"rank={dist.get_rank() if dist.is_initialized() else 0} "
            f"alpha={hyperparameters.alpha} threshold_lr={hyperparameters.threshold_lr} "
            f"betas=({hyperparameters.beta1},{hyperparameters.beta2}) "
            f"anchors={self.glofnd_state.num_anchors} "
            f"layout_sha256={self.data_collator.glofnd_layout['layout_sha256']}",
            flush=True,
        )

    def training_step(self, model, inputs, *args, **kwargs):
        queries, _ = inputs
        anchor_ids = queries.pop(_ANCHOR_IDS_KEY)
        self.glofnd_loss.set_batch_context(self.state.global_step + 1, anchor_ids)
        return super().training_step(model, inputs, *args, **kwargs)

    def _checkpoint_payload(self):
        return {
            "format_version": 1,
            "trainer_global_step": int(self.state.global_step),
            "training_config": self._glofnd_config,
            "state": self.glofnd_state.state_dict(),
            "state_sha256": _state_digest(self.glofnd_state),
        }

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        output_path = Path(output_dir)
        (output_path / _CONFIG_FILE).write_text(
            json.dumps(self._glofnd_config, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _atomic_torch_save(self._checkpoint_payload(), output_path / _STATE_FILE)
        print(
            "[GLOFND_STATE_SAVED] "
            f"step={self.state.global_step} path={output_path / _STATE_FILE} "
            f"initialized={self.glofnd_state.initialized_total} "
            f"sha256={_state_digest(self.glofnd_state)}",
            flush=True,
        )

    def _load_glofnd_state(self, checkpoint_dir):
        checkpoint_dir = str(Path(checkpoint_dir).resolve())
        if self._glofnd_loaded_checkpoint == checkpoint_dir:
            return
        state_path = Path(checkpoint_dir) / _STATE_FILE
        if not state_path.is_file():
            raise RuntimeError(
                f"Refusing to resume GloFND without {_STATE_FILE}: {checkpoint_dir}"
            )
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != 1:
            raise RuntimeError(f"Unsupported GloFND checkpoint payload: {payload.get('format_version')}")
        saved_config = payload.get("training_config")
        config_matches = saved_config == self._glofnd_config
        legacy_safety_keys = {
            "empty_negative_safety",
            "empty_negative_safety_scope",
        }
        if isinstance(saved_config, dict) and not config_matches:
            current_without_safety = {
                key: value
                for key, value in self._glofnd_config.items()
                if key not in legacy_safety_keys
            }
            config_matches = saved_config == current_without_safety
            if config_matches:
                print(
                    "[GLOFND_CHECKPOINT_SAFETY_MIGRATION] "
                    f"rank={dist.get_rank() if dist.is_initialized() else 0} "
                    "resuming legacy state with the empty-negative safety guard",
                    flush=True,
                )
        if not config_matches:
            raise RuntimeError("GloFND checkpoint training configuration differs from this run")
        trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise RuntimeError(f"GloFND checkpoint lacks trainer_state.json: {checkpoint_dir}")
        trainer_step = int(json.loads(trainer_state_path.read_text(encoding="utf-8"))["global_step"])
        if int(payload.get("trainer_global_step", -1)) != trainer_step:
            raise RuntimeError(
                "GloFND threshold/model checkpoint steps differ: "
                f"threshold={payload.get('trainer_global_step')} trainer={trainer_step}"
            )
        self.glofnd_state.load_state_dict(payload["state"])
        digest = _state_digest(self.glofnd_state)
        if digest != payload.get("state_sha256"):
            raise RuntimeError(
                f"GloFND checkpoint digest mismatch: computed={digest} saved={payload.get('state_sha256')}"
            )
        self._glofnd_loaded_checkpoint = checkpoint_dir
        print(
            "[GLOFND_STATE_RESUMED] "
            f"rank={dist.get_rank() if dist.is_initialized() else 0} "
            f"step={payload.get('trainer_global_step')} initialized={self.glofnd_state.initialized_total} "
            f"sha256={digest}",
            flush=True,
        )

    def _inner_training_loop(self, *args, **kwargs):
        checkpoint_dir = kwargs.get("resume_from_checkpoint")
        if checkpoint_dir is None and len(args) >= 3:
            checkpoint_dir = args[2]
        if checkpoint_dir is not None:
            self._load_glofnd_state(checkpoint_dir)
        return super()._inner_training_loop(*args, **kwargs)
