from typing import List, Union, Callable, Any
from contextlib import nullcontext
from itertools import repeat
from collections import UserDict
import logging
import os

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast
from src.training.grad_cache.context_managers import RandContext
import numpy as np
import bisect

logger = logging.getLogger(__name__)


class GradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    def __init__(
        self,
        models: List[nn.Module],
        chunk_sizes: Union[int, List[int]],
        loss_fn: Callable[..., Tensor],
        split_input_fn: Callable[[Any, int], Any] = None,
        get_rep_fn: Callable[..., Tensor] = None,
        fp16: bool = False,
        scaler: GradScaler = None,
        process_fn: Callable = None,
        dynamic_limit=None,
        embedding_gradient_strategy: str = "info_nce",
        ega_alpha: float = 20.0,
    ):
        """
        Initialize the Gradient Cache class instance.
        :param models: A list of all encoder models to be updated by the current cache.
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss_fn: A loss function that takes arbitrary numbers of representation tensors and
        arbitrary numbers of keyword arguments as input. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param fp16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        """
        self.models = models
        if embedding_gradient_strategy not in {
            "info_nce",
            "ega",
            "b3_ega",
            "llave",
        }:
            raise ValueError(
                "Unknown embedding gradient strategy: "
                f"{embedding_gradient_strategy!r}"
            )
        self.embedding_gradient_strategy = embedding_gradient_strategy
        self.ega_alpha = float(ega_alpha)
        self._ega_config_logged = False

        if isinstance(chunk_sizes, int):
            self.chunk_sizes = [chunk_sizes for _ in range(len(models))]
        else:
            self.chunk_sizes = chunk_sizes

        self.split_input_fn = split_input_fn
        self.process_fn = process_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fn = loss_fn

        if fp16:
            assert scaler is not None, "mixed precision training requires a gradient scaler passed in"

        self.fp16 = fp16
        self.scaler = scaler

        self._get_input_tensors_strict = False
        self.dynamic_limit = dynamic_limit

        # Hook for capturing embeddings during training (no extra forward pass)
        self.reps_callback = None
        self._trace_step = 0
        self._trace_steps = int(os.environ.get("B3_DDP_TRACE_STEPS", "0"))
        self._numeric_debug_steps = int(os.environ.get("B3_NUMERIC_DEBUG_STEPS", "0"))

        # Opt-in only.  UniME-V2 mixes text and high-resolution visual items in
        # the same expanded batch.  Its trainer enables this flag so samples
        # that have already been sorted by active token length can use a
        # different microbatch size at each point in the batch.  Every other
        # training/evaluation path retains the historical fixed-size splitter.
        self.length_aware_packing = False

        self.multipliers_qry = [2**i for i in range(int(np.log2(self.chunk_sizes[0])) + 1)]
        self.thresholds_qry = [self.dynamic_limit * m for m in self.multipliers_qry]

        self.multipliers_pos = [2**i for i in range(int(np.log2(self.chunk_sizes[1])) + 1)]
        self.thresholds_pos = [self.dynamic_limit * m for m in self.multipliers_pos]

    def _trace(self, stage: str):
        if self._trace_step > self._trace_steps or self._trace_steps <= 0:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(
            f"[GC_DDP_TRACE] rank={rank} step={self._trace_step} stage={stage}",
            flush=True,
        )

    def _check_tensor_finite(self, tensor: Tensor, stage: str):
        if self._trace_step > self._numeric_debug_steps or self._numeric_debug_steps <= 0:
            return
        finite = torch.isfinite(tensor)
        local_failed = torch.tensor(
            0 if finite.all() else 1,
            device=tensor.device,
            dtype=torch.int32,
        )
        global_failed = local_failed.clone()
        if dist.is_initialized():
            dist.all_reduce(global_failed, op=dist.ReduceOp.MAX)
        if global_failed.item() == 0:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        finite_count = int(finite.sum().item())
        raise FloatingPointError(
            f"[B3_NUMERIC_ERROR] rank={rank} step={self._trace_step} "
            f"stage={stage} dtype={tensor.dtype} shape={tuple(tensor.shape)} "
            f"local_finite={finite_count}/{tensor.numel()} "
            f"local_failed={local_failed.item()} global_failed=1"
        )

    def _report_nonfinite_chunk(
        self,
        reps: Tensor,
        model_input,
        chunk_index: int,
        batch_offset: int,
    ):
        """Print rank-local evidence without introducing a collective."""
        if (
            self._trace_step > self._numeric_debug_steps
            or self._numeric_debug_steps <= 0
            or torch.isfinite(reps).all()
        ):
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        finite_by_row = torch.isfinite(reps).reshape(reps.shape[0], -1).all(dim=1)
        bad_local_rows = torch.nonzero(~finite_by_row, as_tuple=False).flatten().tolist()

        payload_name = "unknown"
        payload = None
        if isinstance(model_input, (dict, UserDict)) and len(model_input) == 1:
            payload_name, payload = next(iter(model_input.items()))
        if not isinstance(payload, (dict, UserDict)):
            payload = {}

        attention_mask = payload.get("attention_mask")
        input_ids = payload.get("input_ids")
        image_grids = payload.get("image_grid_thw")
        video_grids = payload.get("video_grid_thw")
        pixel_values = payload.get("pixel_values")
        pixel_values_videos = payload.get("pixel_values_videos")

        def _row_value(values, row):
            if isinstance(values, torch.Tensor):
                return values[row]
            if isinstance(values, (list, tuple)) and row < len(values):
                return values[row]
            return None

        def _present(value):
            if value is None:
                return False
            try:
                value_tensor = torch.as_tensor(value)
                return value_tensor.numel() > 0 and bool(value_tensor.sum().item() > 0)
            except (TypeError, ValueError):
                return True

        def _finite(value):
            if value is None:
                return None
            try:
                return bool(torch.isfinite(torch.as_tensor(value)).all().item())
            except (TypeError, ValueError):
                return False

        row_details = []
        image_token_id = 151655
        video_token_id = 151656
        for row in bad_local_rows:
            mask_row = _row_value(attention_mask, row)
            ids_row = _row_value(input_ids, row)
            valid_tokens = (
                int(mask_row.sum().item())
                if isinstance(mask_row, torch.Tensor)
                else None
            )
            last_mask = (
                int(mask_row[-1].item())
                if isinstance(mask_row, torch.Tensor) and mask_row.numel() > 0
                else None
            )
            image_tokens = (
                int((ids_row == image_token_id).sum().item())
                if isinstance(ids_row, torch.Tensor)
                else None
            )
            video_tokens = (
                int((ids_row == video_token_id).sum().item())
                if isinstance(ids_row, torch.Tensor)
                else None
            )
            image_grid = _row_value(image_grids, row)
            video_grid = _row_value(video_grids, row)
            image_pixels = _row_value(pixel_values, row)
            video_pixels = _row_value(pixel_values_videos, row)
            row_details.append(
                {
                    "batch_row": batch_offset + row,
                    "valid_tokens": valid_tokens,
                    "last_mask": last_mask,
                    "image_tokens": image_tokens,
                    "video_tokens": video_tokens,
                    "has_image": _present(image_grid),
                    "has_video": _present(video_grid),
                    "image_pixels_finite": _finite(image_pixels),
                    "video_pixels_finite": _finite(video_pixels),
                }
            )

        print(
            f"[B3_CHUNK_NUMERIC] rank={rank} step={self._trace_step} "
            f"payload={payload_name} chunk={chunk_index} "
            f"bad_rows={bad_local_rows} details={row_details}",
            flush=True,
        )

    def _check_model_gradients(self, model: nn.Module, stage: str):
        if self._trace_step > self._numeric_debug_steps or self._numeric_debug_steps <= 0:
            return
        first_bad = None
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                finite = torch.isfinite(parameter.grad)
                if not finite.all():
                    first_bad = (
                        name,
                        parameter.grad.dtype,
                        tuple(parameter.grad.shape),
                        int(finite.sum().item()),
                        parameter.grad.numel(),
                    )
                    break

        device = next(model.parameters()).device
        local_failed = torch.tensor(
            1 if first_bad is not None else 0,
            device=device,
            dtype=torch.int32,
        )
        global_failed = local_failed.clone()
        if dist.is_initialized():
            dist.all_reduce(global_failed, op=dist.ReduceOp.MAX)
        if global_failed.item() == 0:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if first_bad is None:
            detail = "local_gradients_finite=True"
        else:
            name, dtype, shape, finite_count, numel = first_bad
            detail = (
                f"parameter={name} dtype={dtype} shape={shape} "
                f"local_finite={finite_count}/{numel}"
            )
        raise FloatingPointError(
            f"[B3_NUMERIC_ERROR] rank={rank} step={self._trace_step} "
            f"stage={stage} {detail} "
            f"local_failed={local_failed.item()} global_failed=1"
        )

    def __call__(self, *args, **kwargs):
        """
        Call the cache_step function.
        :return: Current step loss.
        """
        return self.cache_step(*args, **kwargs)

    def split_inputs(self, model_input, chunk_size: int) -> List:
        """
        Split input into chunks. Will call user provided `split_input_fn` if specified. Otherwise,
        it can handle input types of tensor, list of tensors and dictionary of tensors.
        :param model_input: Generic model input.
        :param chunk_size:  Size of each chunk.
        :return: A list of chunked model input.
        """
        # delegate splitting to user provided function
        if self.split_input_fn is not None:
            return self.split_input_fn(model_input, chunk_size)

        if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, Tensor) for x in model_input.values()):
            keys = list(model_input.keys())
            chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
            return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

        elif isinstance(model_input, list) and all(isinstance(x, Tensor) for x in model_input):
            chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
            return [list(s) for s in zip(*chunked_x)]

        elif isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
            args_chunks = self.split_inputs(model_input[0], chunk_size)
            kwargs_chunks = self.split_inputs(model_input[1], chunk_size)
            return list(zip(args_chunks, kwargs_chunks))

        else:
            raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

    @staticmethod
    def _slice_wrapped_batch(model_input, start: int, stop: int):
        """Slice the one-key MMEB input wrapper without changing its schema."""
        if not isinstance(model_input, (dict, UserDict)) or len(model_input) != 1:
            raise TypeError(
                "Length-aware packing requires a one-key MMEB model-input wrapper"
            )
        arg_key = next(iter(model_input))
        payload = model_input[arg_key]
        if not isinstance(payload, (dict, UserDict)):
            raise TypeError("Length-aware packing requires a dictionary payload")
        batch_size = int(payload["attention_mask"].shape[0])
        sliced = {}
        for key, value in payload.items():
            if isinstance(value, Tensor):
                if value.ndim == 0 or value.shape[0] != batch_size:
                    raise ValueError(
                        f"Cannot length-pack tensor {key!r} with shape "
                        f"{tuple(value.shape)} for batch size {batch_size}"
                    )
                sliced[key] = value[start:stop]
            else:
                if len(value) != batch_size:
                    raise ValueError(
                        f"Cannot length-pack field {key!r} with length "
                        f"{len(value)} for batch size {batch_size}"
                    )
                sliced[key] = value[start:stop]
        return {arg_key: sliced}

    @staticmethod
    def _trim_chunk_padding(model_input):
        """Crop token-aligned tensors to the active width of one microbatch."""
        arg_key = next(iter(model_input))
        payload = model_input[arg_key]
        attention_mask = payload["attention_mask"]
        if attention_mask.ndim != 2 or attention_mask.shape[0] == 0:
            raise ValueError(
                "Length-aware packing requires a non-empty rank-2 attention mask"
            )
        active_lengths = attention_mask.long().sum(dim=1)
        if (active_lengths <= 0).any():
            raise ValueError("Length-aware packing found an empty token sequence")
        active_width = int(active_lengths.max().item())
        padded_width = int(attention_mask.shape[1])
        if active_width > padded_width:
            raise RuntimeError("Active token count exceeds the padded sequence width")

        # Qwen2-VL training uses left padding, but supporting a consistently
        # right-padded input makes this helper safe to unit test and reuse.
        left_padded = bool(attention_mask[:, -1].bool().all().item())
        right_padded = bool(attention_mask[:, 0].bool().all().item())
        if not left_padded and not right_padded:
            raise ValueError(
                "Length-aware packing requires consistently left- or right-padded rows"
            )
        if left_padded:
            token_slice = slice(padded_width - active_width, padded_width)
            trimmed_mask = attention_mask[:, token_slice]
            if (trimmed_mask[:, 1:] < trimmed_mask[:, :-1]).any():
                raise ValueError("Attention mask is not contiguous left padding")
        else:
            token_slice = slice(0, active_width)
            trimmed_mask = attention_mask[:, token_slice]
            if (trimmed_mask[:, 1:] > trimmed_mask[:, :-1]).any():
                raise ValueError("Attention mask is not contiguous right padding")

        for key in ("input_ids", "attention_mask", "mm_token_type_ids", "position_ids"):
            value = payload.get(key)
            if not isinstance(value, Tensor):
                continue
            if value.ndim == 2 and value.shape[1] == padded_width:
                payload[key] = value[:, token_slice]
            elif value.ndim == 3 and value.shape[-1] == padded_width:
                payload[key] = value[..., token_slice]
        return model_input, active_width

    def _split_inputs_length_aware(
        self,
        model_input,
        base_chunk_size: int,
        thresholds: List[int],
    ):
        """Pack a length-sorted batch into token-budgeted microbatches."""
        arg_key = next(iter(model_input))
        attention_mask = model_input[arg_key]["attention_mask"]
        lengths = attention_mask.long().sum(dim=1)
        if lengths.numel() == 0:
            raise ValueError("Cannot length-pack an empty batch")
        if (lengths[:-1] < lengths[1:]).any():
            raise ValueError(
                "Length-aware packing expects rows sorted by descending active length"
            )

        chunks = []
        chunk_lengths = []
        chunk_widths = []
        cursor = 0
        batch_size = int(lengths.numel())
        while cursor < batch_size:
            max_tokens = int(lengths[cursor].item())
            multiplier_index = min(
                bisect.bisect_left(thresholds, max_tokens),
                len(thresholds) - 1,
            )
            capacity = max(1, base_chunk_size // (2 ** multiplier_index))
            stop = min(batch_size, cursor + capacity)
            chunk = self._slice_wrapped_batch(model_input, cursor, stop)
            chunk, active_width = self._trim_chunk_padding(chunk)
            chunks.append(chunk)
            chunk_lengths.append(stop - cursor)
            chunk_widths.append(active_width)
            cursor = stop
        return chunks, chunk_lengths, chunk_widths

    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def model_call(self, model: nn.Module, model_input):
        """
        Literally call the model's __call__ method.
        :param model: model to be called
        :param model_input: input to the model call
        :return: model output
        """
        with autocast() if self.fp16 else nullcontext():
            if isinstance(model_input, Tensor):
                return model(model_input)
            elif isinstance(model_input, list):
                return model(*model_input)
            elif isinstance(model_input, (dict, UserDict)):
                return model(**model_input)
            elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
                model_args, model_kwargs = model_input
                return model(*model_args, **model_kwargs)
            else:
                raise NotImplementedError

    def get_reps(self, model_out) -> Tensor:
        """
        Return representation tensor from generic model output
        :param model_out: generic model output
        :return: a single tensor corresponding to the model representation output
        """
        if self.get_rep_fn is not None:
            return self.get_rep_fn(model_out)
        else:
            return model_out

    def compute_loss(self, *reps: Tensor, **loss_kwargs) -> Tensor:
        """
        Compute the loss based on the representation tensors. The tensors should be ordered same as the list of models
        registered in this GradCache class instance.
        :param reps: Representations for computing the loss.
        :param loss_kwargs: Keyword arguments input to the loss function.
        :return: the loss tensor.
        """
        loss = self.loss_fn(*reps, **loss_kwargs)
        return loss

    def forward_no_grad(
        self,
        model: nn.Module,
        model_inputs,
    ) -> [Tensor, List[RandContext]]:
        """
        The first forward pass without gradient computation.
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks.
        :return: A tuple of a) representations and b) recorded random states.
        """
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            batch_offset = 0
            for chunk_index, x in enumerate(model_inputs):
                self._trace(f"no_grad_chunk_{chunk_index}_begin")
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                y = self.model_call(model, x)
                reps = self.get_reps(y)
                self._report_nonfinite_chunk(
                    reps,
                    x,
                    chunk_index,
                    batch_offset,
                )
                model_reps.append(reps)
                batch_offset += reps.shape[0]
                self._trace(f"no_grad_chunk_{chunk_index}_end")

        # concatenate all sub-batch representations
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    def build_cache(self, *reps: Tensor, **loss_kwargs) -> [List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        if self.embedding_gradient_strategy == "ega":
            return self._build_ega_cache(*reps, **loss_kwargs)
        if self.embedding_gradient_strategy == "b3_ega":
            return self._build_b3_ega_cache(*reps, **loss_kwargs)

        reps = [r.detach().requires_grad_() for r in reps]
        with autocast() if self.fp16 else nullcontext():
            loss = self.compute_loss(*reps, **loss_kwargs)

        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        cache = [r.grad for r in reps]

        return cache, loss.detach()

    def _gather_detached_equal_shape(self, tensor: Tensor, name: str) -> Tensor:
        """Gather rank-ordered representations without creating an autograd graph."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor.detach()

        # all_gather requires equal shapes. Training uses drop-last full local
        # batches, but fail coherently before the tensor collective if that
        # contract ever changes.
        local_shape = torch.tensor(
            tensor.shape, device=tensor.device, dtype=torch.int64
        )
        gathered_shapes = [torch.empty_like(local_shape) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_shapes, local_shape)
        shapes = [tuple(int(value) for value in shape.tolist()) for shape in gathered_shapes]
        if any(shape != shapes[0] for shape in shapes):
            raise RuntimeError(
                f"EGA requires equal {name} representation shapes on every rank; "
                f"found {shapes}"
            )

        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.detach())
        return torch.cat(gathered, dim=0)

    def _build_ega_cache(self, *reps: Tensor, **loss_kwargs):
        """Build the GradCache from EGA's explicit global embedding gradients."""
        from src.training.ega import ega_gradients_and_loss

        if len(reps) != 2:
            raise RuntimeError(f"EGA expects query and target reps, got {len(reps)} tensors")
        local_query, local_target = (rep.detach() for rep in reps)
        local_contract_failed = torch.tensor(
            int(local_query.shape != local_target.shape),
            device=local_query.device,
            dtype=torch.int32,
        )
        global_contract_failed = local_contract_failed.clone()
        if dist.is_initialized():
            dist.all_reduce(global_contract_failed, op=dist.ReduceOp.MAX)
        if global_contract_failed.item() != 0:
            raise RuntimeError(
                "EGA requires one local target per query on every rank; "
                f"local_contract_failed={local_contract_failed.item()} "
                f"query={tuple(local_query.shape)} target={tuple(local_target.shape)}"
            )

        self._trace("ega_gather_query_begin")
        global_query = self._gather_detached_equal_shape(local_query, "query")
        self._trace("ega_gather_query_end")
        self._trace("ega_gather_target_begin")
        global_target = self._gather_detached_equal_shape(local_target, "target")
        self._trace("ega_gather_target_end")

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_batch = local_query.size(0)
        if not self._ega_config_logged:
            query_norm_error = (
                global_query.float().norm(dim=1) - 1.0
            ).abs().max()
            target_norm_error = (
                global_target.float().norm(dim=1) - 1.0
            ).abs().max()
            if query_norm_error.item() > 1e-2 or target_norm_error.item() > 1e-2:
                raise RuntimeError(
                    "EGA requires L2-normalized embeddings; maximum norm errors "
                    f"are query={query_norm_error.item():.8g} "
                    f"target={target_norm_error.item():.8g}"
                )
        should_log = (
            not self._ega_config_logged
            or (
                os.environ.get("B3_DEBUG", "0") == "1"
                and self._trace_step
                % max(1, int(os.environ.get("B3_DEBUG_INTERVAL", "100")))
                == 0
            )
        )
        grad_query_global, grad_target_global, mean_loss, diagnostics = (
            ega_gradients_and_loss(
                global_query,
                global_target,
                temperature=self.loss_fn.temperature,
                alpha=self.ega_alpha,
                return_diagnostics=should_log and rank == 0,
            )
        )

        # DistributedContrastiveLoss multiplies global mean CE by world_size
        # before taking derivatives. Consequently the cache gradient for each
        # local representation is world_size/global_batch == 1/local_batch.
        # DDP later averages parameter gradients and recovers the global-mean
        # update. This scaling also makes alpha=0 match the existing path.
        cache_scale = 1.0 / local_batch
        start = rank * local_batch
        stop = start + local_batch
        grad_query = grad_query_global[start:stop] * cache_scale
        grad_target = grad_target_global[start:stop] * cache_scale

        # Match the existing GradScaler contract if somebody explicitly runs
        # EGA under FP16. Qwen3.5's supported script uses BF16 and scale=1.
        if self.fp16:
            grad_scale = float(self.scaler.get_scale())
            grad_query = grad_query * grad_scale
            grad_target = grad_target * grad_scale

        grad_query = grad_query.to(dtype=local_query.dtype)
        grad_target = grad_target.to(dtype=local_target.dtype)
        reported_loss = mean_loss * world_size

        if should_log and rank == 0:
            diagnostics["local_query_cache_norm_mean"] = float(
                grad_query.float().norm(dim=1).mean().item()
            )
            diagnostics["local_target_cache_norm_mean"] = float(
                grad_target.float().norm(dim=1).mean().item()
            )
            formatted = " ".join(
                f"{name}={value:.8g}" for name, value in diagnostics.items()
            )
            print(
                "[EGA_GLOBAL_GRADIENT_OK] "
                f"step={self._trace_step} alpha={self.ega_alpha:.8g} "
                f"temperature={self.loss_fn.temperature:.8g} "
                f"world_size={world_size} local_pairs={local_batch} "
                f"global_pairs={global_query.size(0)} cache_scale={cache_scale:.8g} "
                f"cache_dtype={grad_query.dtype} {formatted}",
                flush=True,
            )
        self._ega_config_logged = True
        return [grad_query, grad_target], reported_loss.detach()

    def _build_b3_ega_cache(self, *reps: Tensor, **loss_kwargs):
        """Build rectangular EGA caches for B3's 1-positive + 5-negative cases."""
        from src.training.ega import b3_ega_gradients_and_loss

        if len(reps) != 2:
            raise RuntimeError(
                f"B3+EGA expects query and target reps, got {len(reps)} tensors"
            )
        local_query, local_target = (rep.detach() for rep in reps)
        expected_target_per_query = 6
        local_contract_ok = (
            local_query.ndim == 2
            and local_target.ndim == 2
            and local_query.size(0) > 0
            and local_query.size(1) == local_target.size(1)
            and local_target.size(0)
            == local_query.size(0) * expected_target_per_query
        )
        local_contract_failed = torch.tensor(
            int(not local_contract_ok),
            device=local_query.device,
            dtype=torch.int32,
        )
        global_contract_failed = local_contract_failed.clone()
        if dist.is_initialized():
            dist.all_reduce(global_contract_failed, op=dist.ReduceOp.MAX)
        if global_contract_failed.item() != 0:
            raise RuntimeError(
                "B3+EGA requires exactly one positive followed by five hard "
                "negative targets per local query on every rank; "
                f"local_contract_failed={local_contract_failed.item()} "
                f"query={tuple(local_query.shape)} target={tuple(local_target.shape)} "
                f"expected_target_per_query={expected_target_per_query}"
            )

        self._trace("b3_ega_gather_query_begin")
        global_query = self._gather_detached_equal_shape(local_query, "B3+EGA query")
        self._trace("b3_ega_gather_query_end")
        self._trace("b3_ega_gather_target_begin")
        global_target = self._gather_detached_equal_shape(local_target, "B3+EGA target")
        self._trace("b3_ega_gather_target_end")

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_query_count = local_query.size(0)
        local_target_count = local_target.size(0)
        if not self._ega_config_logged:
            query_norm_error = (
                global_query.float().norm(dim=1) - 1.0
            ).abs().max()
            target_norm_error = (
                global_target.float().norm(dim=1) - 1.0
            ).abs().max()
            if query_norm_error.item() > 1e-2 or target_norm_error.item() > 1e-2:
                raise RuntimeError(
                    "B3+EGA requires L2-normalized embeddings; maximum norm "
                    f"errors are query={query_norm_error.item():.8g} "
                    f"target={target_norm_error.item():.8g}"
                )

        should_log = (
            not self._ega_config_logged
            or (
                os.environ.get("B3_DEBUG", "0") == "1"
                and self._trace_step
                % max(1, int(os.environ.get("B3_DEBUG_INTERVAL", "100")))
                == 0
            )
        )
        grad_query_global, grad_target_global, mean_loss, diagnostics = (
            b3_ega_gradients_and_loss(
                global_query,
                global_target,
                target_per_query=expected_target_per_query,
                temperature=self.loss_fn.temperature,
                alpha=self.ega_alpha,
                return_diagnostics=should_log and rank == 0,
            )
        )

        # The distributed B3 loss is a global query mean multiplied by world
        # size before differentiating. DDP later averages parameter gradients,
        # so both rectangular cache sides use 1/local_query_count. Target
        # slicing uses the rank's full 6x local target block.
        cache_scale = 1.0 / local_query_count
        query_start = rank * local_query_count
        query_stop = query_start + local_query_count
        target_start = rank * local_target_count
        target_stop = target_start + local_target_count
        grad_query = grad_query_global[query_start:query_stop] * cache_scale
        grad_target = grad_target_global[target_start:target_stop] * cache_scale

        if self.fp16:
            grad_scale = float(self.scaler.get_scale())
            grad_query = grad_query * grad_scale
            grad_target = grad_target * grad_scale

        grad_query = grad_query.to(dtype=local_query.dtype)
        grad_target = grad_target.to(dtype=local_target.dtype)
        reported_loss = mean_loss * world_size

        if should_log and rank == 0:
            diagnostics["local_query_cache_norm_mean"] = float(
                grad_query.float().norm(dim=1).mean().item()
            )
            diagnostics["local_target_cache_norm_mean"] = float(
                grad_target.float().norm(dim=1).mean().item()
            )
            formatted = " ".join(
                f"{name}={value:.8g}" for name, value in diagnostics.items()
            )
            print(
                "[B3_EGA_GLOBAL_GRADIENT_OK] "
                f"step={self._trace_step} alpha={self.ega_alpha:.8g} "
                f"temperature={self.loss_fn.temperature:.8g} "
                f"world_size={world_size} local_queries={local_query_count} "
                f"local_targets={local_target_count} "
                f"global_queries={global_query.size(0)} "
                f"global_targets={global_target.size(0)} "
                f"target_per_query={expected_target_per_query} "
                f"explicit_hard_negatives_per_query={expected_target_per_query - 1} "
                f"cache_scale={cache_scale:.8g} cache_dtype={grad_query.dtype} "
                f"{formatted}",
                flush=True,
            )
        self._ega_config_logged = True
        return [grad_query, grad_target], reported_loss.detach()

    def forward_backward(
        self,
        model: nn.Module,
        model_inputs,
        cached_gradients: List[Tensor],
        random_states: List[RandContext],
        no_sync_except_last: bool = False,
        expected_representations: List[Tensor] = None,
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        replay_max_abs = torch.zeros(
            (), device=cached_gradients[0].device, dtype=torch.float32
        )
        replay_max_chunk = -1
        replay_mean_at_max = 0.0

        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        if expected_representations is None:
            expected_representations = repeat(None)

        for chunk_index, (x, state, gradient, expected, sync_context) in enumerate(zip(
                model_inputs,
                random_states,
                cached_gradients,
                expected_representations,
                sync_contexts,
        )):
            self._trace(f"backward_chunk_{chunk_index}_begin")
            with sync_context():
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)

                # Gradient Cache is valid only if the graph-enabled replay is
                # the same forward as the graph-less pass. Check this before
                # applying cached representation gradients so a changed
                # model/PEFT implementation cannot silently train in the
                # wrong direction for hundreds of steps.
                if (
                    expected is not None
                    and self._trace_step <= self._numeric_debug_steps
                    and self._numeric_debug_steps > 0
                ):
                    replay_delta = (
                        reps.detach().float() - expected.detach().float()
                    ).abs()
                    max_abs = replay_delta.max()
                    mean_abs = replay_delta.mean()
                    # Rank-local dynamic chunking means ranks can execute a
                    # different number of iterations here. Never place a
                    # collective (or a rank-local exception) in this loop:
                    # either would strand peers at a different DDP operation.
                    if max_abs.item() > replay_max_abs.item():
                        replay_max_abs.copy_(max_abs.detach())
                        replay_max_chunk = chunk_index
                        replay_mean_at_max = mean_abs.item()

                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                surrogate.backward()
            self._trace(f"backward_chunk_{chunk_index}_end")

        return replay_max_abs, replay_max_chunk, replay_mean_at_max

    def cache_step(
        self,
        *model_inputs,
        no_sync_except_last: bool = False,
        **loss_kwargs
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        all_reps = []
        all_rnd_states = []
        self._trace_step += 1
        self._trace("cache_step_begin")

        if no_sync_except_last:
            assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), self.models)), \
                'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
                'proper initializations.'
            # B3 intentionally allows rank-local dynamic chunk sizes. This
            # means ranks can execute a different number of no-sync forwards.
            # DDP buffer broadcasts would introduce a collective in every
            # forward and deadlock in that situation, so fail fast unless the
            # Trainer wrapper was created with broadcast_buffers=False.
            broadcast_buffer_models = [
                model for model in self.models
                if getattr(model, "broadcast_buffers", False)
            ]
            if broadcast_buffer_models:
                raise RuntimeError(
                    "B3 rank-local GradCache chunking requires "
                    "DDP(broadcast_buffers=False). Pass "
                    "--ddp_broadcast_buffers False."
                )

        # Preserve B3's rank-local dynamic chunking exactly. Besides memory
        # usage, chunk boundaries determine the order in which LoRA dropout
        # consumes RNG, so synchronizing chunk sizes across ranks changes the
        # stochastic training trajectory even when the cached-gradient math is
        # otherwise equivalent.
        local_numtokens_qry = model_inputs[0]['qry']['attention_mask'].shape[1]
        local_numtokens_pos = model_inputs[1]['tgt']['attention_mask'].shape[1]
        numtokens_qry = local_numtokens_qry
        numtokens_pos = local_numtokens_pos

        qry_multiplier_idx = min(
            bisect.bisect_left(self.thresholds_qry, numtokens_qry),
            len(self.multipliers_qry) - 1,
        )
        pos_multiplier_idx = min(
            bisect.bisect_left(self.thresholds_pos, numtokens_pos),
            len(self.multipliers_pos) - 1,
        )
        dynamic_chunk_size_qry = self.chunk_sizes[0] // self.multipliers_qry[qry_multiplier_idx]
        dynamic_chunk_size_pos = self.chunk_sizes[1] // self.multipliers_pos[pos_multiplier_idx]

        self.dynamic_chunk_size = [dynamic_chunk_size_qry, dynamic_chunk_size_pos]

        logger.info(
            "Dynamic GradCache chunk size: %s "
            "(rank-local tokens: qry=%s, pos=%s)",
            self.dynamic_chunk_size,
            local_numtokens_qry,
            local_numtokens_pos,
        )

        if self.length_aware_packing:
            packed_inputs = []
            packed_chunk_lengths = []
            packed_chunk_widths = []
            for input_index, (model_input, base_chunk_size, thresholds) in enumerate(
                zip(
                    model_inputs,
                    self.chunk_sizes,
                    (self.thresholds_qry, self.thresholds_pos),
                )
            ):
                chunks, chunk_lengths, chunk_widths = self._split_inputs_length_aware(
                    model_input,
                    base_chunk_size,
                    thresholds,
                )
                packed_inputs.append(chunks)
                packed_chunk_lengths.append(chunk_lengths)
                packed_chunk_widths.append(chunk_widths)
            model_inputs = packed_inputs
            self._active_chunk_lengths = packed_chunk_lengths

            rank = dist.get_rank() if dist.is_initialized() else 0
            debug_interval = max(1, int(os.environ.get("B3_DEBUG_INTERVAL", "100")))
            should_log_pack = rank == 0 and (
                self._trace_step == 1
                or (
                    os.environ.get("B3_DEBUG", "0") == "1"
                    and self._trace_step % debug_interval == 0
                )
            )
            if should_log_pack:
                for side, original, chunks, widths in zip(
                    ("query", "target"),
                    model_inputs,
                    packed_chunk_lengths,
                    packed_chunk_widths,
                ):
                    active_tokens = sum(
                        int(chunk[next(iter(chunk))]["attention_mask"].sum().item())
                        for chunk in original
                    )
                    packed_tokens = sum(
                        int(chunk_size) * int(width)
                        for chunk_size, width in zip(chunks, widths)
                    )
                    batch_rows = sum(chunks)
                    original_width = (
                        local_numtokens_qry if side == "query" else local_numtokens_pos
                    )
                    original_tokens = batch_rows * original_width
                    histogram = {}
                    for chunk_size in chunks:
                        histogram[chunk_size] = histogram.get(chunk_size, 0) + 1
                    print(
                        "[UNIME_V2_LENGTH_PACK] "
                        f"step={self._trace_step} side={side} rows={batch_rows} "
                        f"chunks={len(chunks)} chunk_histogram={histogram} "
                        f"original_padded_tokens={original_tokens} "
                        f"packed_padded_tokens={packed_tokens} "
                        f"active_tokens={active_tokens} "
                        f"padding_compute_reduction="
                        f"{1.0 - packed_tokens / max(1, original_tokens):.6f} "
                        f"packed_utilization={active_tokens / max(1, packed_tokens):.6f}",
                        flush=True,
                    )
        else:
            model_inputs = [
                self.split_inputs(x, chunk_size)
                for x, chunk_size in zip(model_inputs, self.dynamic_chunk_size)
            ]
            self._active_chunk_lengths = None
        self._trace(
            "chunk_plan_"
            f"qry={len(model_inputs[0])}_tgt={len(model_inputs[1])}_"
            f"qry_chunk={dynamic_chunk_size_qry}_tgt_chunk={dynamic_chunk_size_pos}"
        )

        if self.process_fn:
            # each minibatch -> self.process_fn(minibatch)
            _model_inputs = []
            for arg_group in model_inputs:
                _arg_groups = []
                for key2val_dict in arg_group:
                    _key2val_dict = {}
                    for arg_key, arg_val in key2val_dict.items():
                        _key2val_dict[arg_key] = self.process_fn(arg_val)
                    _arg_groups.append(_key2val_dict)
                _model_inputs.append(_arg_groups)
            model_inputs = _model_inputs

        for input_index, (model, x) in enumerate(zip(self.models, model_inputs)):
            self._trace(f"no_grad_forward_{input_index}_begin")
            model_reps, rnd_states = self.forward_no_grad(model, x)
            self._trace(f"no_grad_forward_{input_index}_end")
            self._check_tensor_finite(model_reps, f"no_grad_reps_{input_index}")
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        # Notify callback with the freshly computed embeddings (zero extra cost)
        if self.reps_callback is not None:
            self.reps_callback(all_reps)

        self._trace("build_cache_begin")
        cache, loss = self.build_cache(*all_reps, **loss_kwargs)
        self._trace("build_cache_end")
        self._check_tensor_finite(loss, "contrastive_loss")
        for cache_index, cache_tensor in enumerate(cache):
            self._check_tensor_finite(cache_tensor, f"cache_gradient_{cache_index}")
        if self._active_chunk_lengths is not None:
            cache = [
                list(torch.split(cache_tensor, chunk_lengths, dim=0))
                for cache_tensor, chunk_lengths in zip(
                    cache, self._active_chunk_lengths
                )
            ]
            expected_reps = [
                list(torch.split(reps, chunk_lengths, dim=0))
                for reps, chunk_lengths in zip(
                    all_reps, self._active_chunk_lengths
                )
            ]
        else:
            cache = [
                c.split(chunk_size)
                for c, chunk_size in zip(cache, self.dynamic_chunk_size)
            ]
            expected_reps = [
                reps.split(chunk_size)
                for reps, chunk_size in zip(all_reps, self.dynamic_chunk_size)
            ]

        # Match B3's reducer semantics exactly. Chunk counts may differ by
        # rank, but each side performs exactly one synchronized last backward;
        # all earlier rank-local chunks run under DDP.no_sync().
        for input_index, (
            model,
            x,
            model_cache,
            rnd_states,
            expected,
        ) in enumerate(zip(
            self.models,
            model_inputs,
            cache,
            all_rnd_states,
            expected_reps,
        )):
            self._trace(f"backward_{input_index}_begin")
            (
                replay_local_max,
                replay_local_chunk,
                replay_local_mean,
            ) = self.forward_backward(
                model,
                x,
                model_cache,
                rnd_states,
                no_sync_except_last=no_sync_except_last,
                expected_representations=expected,
            )
            self._trace(f"backward_{input_index}_end")

            # This boundary is reached exactly once per input on every rank,
            # after the last DDP gradient reduction has completed. It is safe
            # to aggregate the replay check here even when local chunk counts
            # differ. Raising on all ranks also avoids turning a numeric
            # diagnostic into an apparent NCCL hang.
            if (
                self._trace_step <= self._numeric_debug_steps
                and self._numeric_debug_steps > 0
            ):
                replay_global_max = replay_local_max.clone()
                if dist.is_initialized():
                    dist.all_reduce(replay_global_max, op=dist.ReduceOp.MAX)
                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    print(
                        f"[B3_RECOMPUTE] step={self._trace_step} "
                        f"input={input_index} "
                        f"global_max_abs={replay_global_max.item():.8g} "
                        f"rank0_max_chunk={replay_local_chunk} "
                        f"rank0_mean_at_max={replay_local_mean:.8g}",
                        flush=True,
                    )
                # Keep this as an observation only. B3's original GradCache
                # does not require the graph-less BF16 forward and the
                # graph-enabled replay to be bit-identical. PyTorch may use
                # different forward/autograd kernels, producing small finite
                # differences even after RNG restoration. Non-finite reps,
                # cache gradients and model gradients are checked separately.
            self._check_model_gradients(model, f"backward_{input_index}")

        self._trace("cache_step_end")
        return loss
