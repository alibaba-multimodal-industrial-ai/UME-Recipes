from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F
import logging
import os
from src.debug_fingerprint import log_fingerprint
from src.training.llave import llave_loss_from_similarity

logger = logging.getLogger(__name__)


class SimpleContrastiveLoss:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature
        self.expected_target_per_query = None
        self.expected_layout_name = None
        self._inbatch_contract_logged = False

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean') -> Tensor:
        if target is None:
            if x.size(0) <= 0 or y.size(0) % x.size(0) != 0:
                raise RuntimeError(
                    "Contrastive target/query counts are incompatible: "
                    f"query={tuple(x.shape)} target={tuple(y.shape)}"
                )
            target_per_qry = y.size(0) // x.size(0)
            if (
                self.expected_target_per_query is not None
                and target_per_qry != self.expected_target_per_query
            ):
                layout_name = self.expected_layout_name or "ordinary VLM2Vec in-batch"
                raise RuntimeError(
                    f"{layout_name} loss requires exactly "
                    f"{self.expected_target_per_query} target(s) per query, "
                    f"found {target_per_qry}: "
                    f"query={tuple(x.shape)} target={tuple(y.shape)}"
                )
            logger.info(f"Using {target_per_qry-1} hard negatives")
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        loss = self._compute_contrastive_loss(logits, target, reduction)
        if self.expected_target_per_query is not None and not self._inbatch_contract_logged:
            if not torch.isfinite(loss):
                if self.expected_layout_name:
                    raise FloatingPointError(
                        f"{self.expected_layout_name} loss is non-finite: {loss.item()}"
                    )
                # Preserve the established VLM2Vec diagnostic byte-for-byte.
                raise FloatingPointError(
                    f"Ordinary VLM2Vec in-batch loss is non-finite: {loss.item()}"
                )
            positive = logits.gather(1, target[:, None]).squeeze(1).float()
            negative_mask = torch.ones_like(logits, dtype=torch.bool)
            negative_mask.scatter_(1, target[:, None], False)
            negatives = logits.masked_select(negative_mask).float()
            rank = dist.get_rank() if dist.is_initialized() else 0
            if self.expected_layout_name == "b3_moe":
                marker = "[B3_MOE_LOSS_OK]"
            elif self.expected_layout_name:
                marker = "[B3_ABLATION_LOSS_OK]"
            else:
                marker = "[VLM2VEC_INBATCH_LOSS_OK]"
            print(
                f"{marker} "
                + (
                    f"layout={self.expected_layout_name} "
                    if self.expected_layout_name
                    else ""
                )
                + f"rank={rank} global_queries={x.size(0)} "
                f"global_targets={y.size(0)} target_per_query={target_per_qry} "
                f"label_first={target[:16].tolist()} "
                f"positive_mean={positive.mean().item():.8g} "
                f"positive_min={positive.min().item():.8g} "
                f"positive_max={positive.max().item():.8g} "
                f"negative_mean={negatives.mean().item():.8g} "
                f"negative_max={negatives.max().item():.8g} "
                f"temperature={self.temperature:.8g} loss={loss.detach().float().item():.8g}",
                flush=True,
            )
            self._inbatch_contract_logged = True
        return loss

    def _compute_contrastive_loss(
        self, similarity: Tensor, target: Tensor, reduction: str
    ) -> Tensor:
        # Keep the established InfoNCE path byte-for-byte equivalent by making
        # the new LLaVE behavior an override rather than a branch here.
        return F.cross_entropy(
            similarity / self.temperature, target, reduction=reduction
        )


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature
        self._trace_call = 0
        self._trace_calls = int(os.environ.get("B3_DDP_TRACE_STEPS", "0"))
        self._numeric_debug_calls = int(os.environ.get("B3_NUMERIC_DEBUG_STEPS", "0"))

    def _trace(self, stage: str):
        if self._trace_call > self._trace_calls or self._trace_calls <= 0:
            return
        print(
            f"[GC_DDP_TRACE] rank={self.rank} step={self._trace_call} stage={stage}",
            flush=True,
        )

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        self._trace_call += 1
        log_fingerprint("loss_local_reps", query=x, target=y)
        self._trace("gather_query_begin")
        dist_x = self.gather_tensor(x)
        self._trace("gather_query_end")
        self._trace("gather_target_begin")
        dist_y = self.gather_tensor(y)
        self._trace("gather_target_end")
        target_per_query = dist_y.size(0) // dist_x.size(0)
        labels = torch.arange(
            0,
            dist_x.size(0) * target_per_query,
            target_per_query,
            device=dist_x.device,
            dtype=torch.long,
        )
        log_fingerprint(
            "loss_global_reps",
            query=dist_x,
            target=dist_y,
            labels=labels,
            target_per_query=target_per_query,
        )
        if self._trace_call <= self._numeric_debug_calls:
            target_per_qry = dist_y.size(0) // dist_x.size(0)
            similarity = torch.matmul(dist_x.float(), dist_y.float().transpose(0, 1))
            target = torch.arange(dist_x.size(0), device=dist_x.device) * target_per_qry
            positive = similarity.gather(1, target[:, None]).squeeze(1)
            similarity.scatter_(1, target[:, None], float("-inf"))
            negative_max = similarity.max(dim=1).values
            finite_x = int(torch.isfinite(dist_x).sum().item())
            finite_y = int(torch.isfinite(dist_y).sum().item())
            if self.rank == 0:
                print(
                    "[B3_NUMERIC] "
                    f"step={self._trace_call} "
                    f"qry_dtype={dist_x.dtype} tgt_dtype={dist_y.dtype} "
                    f"qry_finite={finite_x}/{dist_x.numel()} "
                    f"tgt_finite={finite_y}/{dist_y.numel()} "
                    f"qry_norm_mean={dist_x.float().norm(dim=-1).mean().item():.8g} "
                    f"tgt_norm_mean={dist_y.float().norm(dim=-1).mean().item():.8g} "
                    f"positive_mean={positive.mean().item():.8g} "
                    f"negative_max_mean={negative_max.mean().item():.8g}",
                    flush=True,
                )
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self._trace_call <= self._numeric_debug_calls and self.rank == 0:
            print(
                f"[B3_NUMERIC] step={self._trace_call} "
                f"raw_distributed_loss={loss.detach().float().item():.8g}",
                flush=True,
            )
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class _LLaVEHardnessWeightedMixin:
    """Override only the objective; batching and DDP gathering stay unchanged."""

    def _init_llave(self, alpha: float):
        self.llave_alpha = float(alpha)
        self._llave_loss_calls = 0

    def _compute_contrastive_loss(
        self, similarity: Tensor, target: Tensor, reduction: str
    ) -> Tensor:
        self._llave_loss_calls += 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        debug_interval = max(1, int(os.environ.get("B3_DEBUG_INTERVAL", "100")))
        should_log = rank == 0 and (
            self._llave_loss_calls == 1
            or (
                os.environ.get("B3_DEBUG", "0") == "1"
                and self._llave_loss_calls % debug_interval == 0
            )
        )
        loss, diagnostics = llave_loss_from_similarity(
            similarity,
            target,
            temperature=self.temperature,
            alpha=self.llave_alpha,
            reduction=reduction,
            return_diagnostics=should_log,
        )
        if should_log:
            formatted = " ".join(
                f"{name}={value:.8g}" for name, value in diagnostics.items()
            )
            print(
                "[LLAVE_HARDNESS_WEIGHTED_LOSS_OK] "
                f"step={self._llave_loss_calls} alpha={self.llave_alpha:.8g} "
                f"temperature={self.temperature:.8g} "
                "directions=query_to_target+target_to_query mean=True "
                "policy_score=similarity/temperature "
                "negative_reward=alpha*stop_gradient(similarity) "
                f"positive_reward=zero {formatted}",
                flush=True,
            )
        return loss


class LLaVESimpleContrastiveLoss(
    _LLaVEHardnessWeightedMixin, SimpleContrastiveLoss
):
    def __init__(self, temperature: float = 0.02, alpha: float = 9.0):
        super().__init__(temperature=temperature)
        self._init_llave(alpha)


class LLaVEDistributedContrastiveLoss(
    _LLaVEHardnessWeightedMixin, DistributedContrastiveLoss
):
    def __init__(
        self,
        n_target: int = 0,
        scale_loss: bool = True,
        temperature: float = 0.02,
        alpha: float = 9.0,
    ):
        super().__init__(
            n_target=n_target,
            scale_loss=scale_loss,
            temperature=temperature,
        )
        self._init_llave(alpha)
