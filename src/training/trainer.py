import collections
import contextlib
import functools
import json
import shutil
import sys
import time
from datetime import timedelta

from packaging import version
from accelerate import skip_first_batches, DistributedType, InitProcessGroupKwargs
from transformers import PretrainedConfig
from transformers.trainer import Trainer, TRAINING_ARGS_NAME, TRAINER_STATE_NAME
import torch.distributed as dist
from typing import Optional
import os
import torch
import math

from src.training.data.collator import get_dense_rep, split_and_process_vlm_inputs
from src.training.model import (
    MMEBModel,
    qwen2_embedding_token_checkpoint_metadata,
)
from src.training.data.collator import process_vlm_inputs_fns
from src.training.loss import (
    SimpleContrastiveLoss,
    DistributedContrastiveLoss,
    LLaVESimpleContrastiveLoss,
    LLaVEDistributedContrastiveLoss,
)
from src.training.grad_cache.grad_cache import GradCache
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.trainer_callback import (
    ExportableState,
    TrainerState,
)
from transformers.trainer_utils import (
    TrainOutput,
    has_length,
    speed_metrics, seed_worker,
)

try:
    # Transformers 5.x moved this checkpoint-resume helper out of Trainer.
    from transformers.trainer_utils import compare_trainer_and_checkpoint_args
except ImportError:  # Transformers 4.x exposes it as a Trainer instance method.
    compare_trainer_and_checkpoint_args = None

from transformers.trainer_pt_utils import (
    get_model_param_count,
)

from transformers.trainer import FSDP_MODEL_NAME
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_accelerate_available,
    is_apex_available,
    is_torch_xla_available,
    logging, is_sagemaker_mp_enabled,
    CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME
)

from src.training.utils import print_rank, batch_to_device
from src.training.data.dataset import EvalDataset
import copy
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from datasets import load_dataset
from contextlib import contextmanager

POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
                "ImageNet-1K": POS_MOD_CLASS_LABEL,"HatefulMemes":POS_MOD_CLASS_LABEL,"SUN397":POS_MOD_CLASS_LABEL,"N24News":POS_MOD_CLASS_LABEL,"VOC2007":POS_MOD_CLASS_LABEL, "Place365":POS_MOD_CLASS_LABEL,"ImageNet-A":POS_MOD_CLASS_LABEL,"ImageNet-R":POS_MOD_CLASS_LABEL,"ObjectNet":POS_MOD_CLASS_LABEL,"Country211":POS_MOD_CLASS_LABEL,

                "OK-VQA":POS_MOD_ANSWER, "A-OKVQA":POS_MOD_ANSWER, "DocVQA":POS_MOD_ANSWER, "InfographicsVQA":POS_MOD_ANSWER, "ChartQA":POS_MOD_ANSWER, "Visual7W":POS_MOD_ANSWER,"ScienceQA":POS_MOD_ANSWER, "GQA":POS_MOD_ANSWER, "TextVQA":POS_MOD_ANSWER, "VizWiz":POS_MOD_ANSWER,

                "MSCOCO_i2t":POS_MOD_IMAGE_CAPTION, "VisualNews_i2t":POS_MOD_IMAGE_CAPTION,
                }


def _embedding_token_peft_save_kwargs(trainer):
    model_args = getattr(trainer, "model_args", None)
    if getattr(model_args, "qwen2_embedding_token", False):
        # PEFT's automatic heuristic saves the full embedding matrix when it
        # suspects vocabulary growth. This branch uses trainable_token_indices,
        # whose single-row adapter state is sufficient and is validated on load.
        return {"save_embedding_layers": False}
    return {}


def _validate_embedding_token_adapter_state(trainer, state_dict):
    model_args = getattr(trainer, "model_args", None)
    if not getattr(model_args, "qwen2_embedding_token", False):
        return
    from peft.utils import get_peft_model_state_dict

    adapter_state = get_peft_model_state_dict(
        trainer.model.encoder,
        state_dict=state_dict,
        save_embedding_layers=False,
    )
    token_keys = [
        key for key in adapter_state if "trainable_tokens_delta" in key
    ]
    full_embedding_keys = [
        key
        for key in adapter_state
        if key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight")
    ]
    if len(token_keys) != 1 or full_embedding_keys:
        raise RuntimeError(
            "Qwen2 embedding-token adapter state is not isolated to one PEFT "
            f"row: token_keys={token_keys}, "
            f"full_embedding_keys={full_embedding_keys}"
        )
    token_tensor = adapter_state[token_keys[0]]
    if tuple(token_tensor.shape) != (1, int(trainer.model.config.hidden_size)):
        raise RuntimeError(
            "Saved Qwen2 embedding-token tensor has the wrong shape: "
            f"{tuple(token_tensor.shape)}"
        )
    print_rank(
        "[QWEN2_EMBED_TOKEN_ADAPTER_STATE_OK] "
        f"key={token_keys[0]} shape={tuple(token_tensor.shape)} "
        f"adapter_tensor_count={len(adapter_state)} "
        "full_embedding_matrix_present=False"
    )


def _write_embedding_token_checkpoint_metadata(trainer, output_dir):
    model_args = getattr(trainer, "model_args", None)
    if not getattr(model_args, "qwen2_embedding_token", False):
        return
    token_id = getattr(model_args, "qwen2_embedding_token_id", None)
    if not isinstance(token_id, int):
        raise RuntimeError(
            "Cannot save Qwen2 embedding-token checkpoint without its token ID"
        )
    metadata = qwen2_embedding_token_checkpoint_metadata(
        model_args, token_id
    )
    with open(
        os.path.join(output_dir, "qwen2_embedding_token_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print_rank(
        "[QWEN2_EMBED_TOKEN_CHECKPOINT_SAVED] "
        f"output_dir={output_dir} token_id={token_id} "
        "full_embedding_matrix_saved=False"
    )


def _write_qwen3_5_bidirectional_checkpoint_metadata(trainer, output_dir):
    model_args = getattr(trainer, "model_args", None)
    if not getattr(model_args, "qwen3_5_bidirectional_attention", False):
        return
    from src.training.qwen3_5_bidirectional import write_checkpoint_config

    runtime_config = getattr(
        model_args,
        "qwen3_5_bidirectional_runtime_config",
        None,
    )
    if not isinstance(runtime_config, dict):
        raise RuntimeError(
            "Qwen3.5 bidirectional runtime config was not recorded"
        )
    write_checkpoint_config(output_dir, runtime_config)
    print_rank(
        "[QWEN3_5_BIDIRECTIONAL_CHECKPOINT_CONFIG_SAVED] "
        f"output_dir={output_dir}"
    )


def _write_qwen3_5_prompt_checkpoint_metadata(trainer, output_dir):
    model_args = getattr(trainer, "model_args", None)
    profile = getattr(model_args, "qwen3_5_prompt_profile", "legacy")
    if profile == "legacy":
        return
    from src.training.qwen3_5_prompt import write_checkpoint_config

    write_checkpoint_config(output_dir, profile)
    print_rank(
        "[QWEN3_5_PROMPT_CHECKPOINT_CONFIG_SAVED] "
        f"output_dir={output_dir} profile={profile}"
    )

@contextmanager
def time_block(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[Timer] {name}: {elapsed:.4f}s")




if is_apex_available():
    from apex import amp

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False

logger = logging.get_logger(__name__)

class MMEBTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MMEBTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self.processor = self.processing_class
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        # make sure it contains all the keys defined in the output of src.dataset.TrainTextImageDataset
        self._signature_columns = ['query_text', 'query_image', 'target_text', 'target_image', 'label_ids', 'label']

    def get_batch_samples(self, epoch_iterator, num_batches):

        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break
        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.item()
        return batch_samples, num_items_in_batch

    def compute_loss(self, model, inputs, *args, **kwargs):
        qry_inputs, tgt_inputs = inputs
        return model(qry=qry_inputs, tgt=tgt_inputs)

    def _before_optimizer_step(self, model):
        """Extension point for isolated algorithms needing a pre-update probe."""

    def _after_optimizer_step(self, model, optimizer_was_run):
        """Extension point paired with :meth:`_before_optimizer_step`."""

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        _validate_embedding_token_adapter_state(self, state_dict)
        self.model.encoder.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            **_embedding_token_peft_save_kwargs(self),
        )

        processor = getattr(self, "processing_class", None)
        if processor is None:
            # Compatibility with Transformers versions where Trainer still
            # exposed the processor through the deprecated tokenizer alias.
            processor = getattr(self, "tokenizer", None)
        if processor is not None:
            processor.save_pretrained(output_dir)

        _write_embedding_token_checkpoint_metadata(self, output_dir)
        _write_qwen3_5_bidirectional_checkpoint_metadata(self, output_dir)
        _write_qwen3_5_prompt_checkpoint_metadata(self, output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # override original trainer's method
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
            return RandomSampler(self.train_dataset)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.model_args.checkpoint_path = resume_from_checkpoint
        logger.info(f"Loading checkpoint from {resume_from_checkpoint}")
        self.model = MMEBModel.load(self.model_args)
        self.model_wrapped = self.model

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if getattr(args, 'include_tokens_per_second', False):
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if getattr(args, 'include_tokens_per_second', False):
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if getattr(args, 'include_tokens_per_second', False):
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if getattr(self, 'use_apex', False):
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # @ruimeng use steps_trained_in_current_epoch to skip batches for finding buggy data
        # steps_trained_in_current_epoch = 42

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if compare_trainer_and_checkpoint_args is not None:
                compare_trainer_and_checkpoint_args(self.args, self.state)
            else:
                self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader.dataset, "set_epoch"):
                # print(f'\t\tSetting new epoch={epoch}')
                epoch_dataloader.dataset.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if getattr(args, 'past_index', -1) >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            num_items_in_batch = None
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    total_batched_samples += 1

                    print_rank(f"dataset name: {str(set(inputs[0]['global_dataset_name']))}")
                    dataset_stat = collections.Counter(inputs[0]['global_dataset_name'])
                    for dname, count in sorted(dataset_stat.items(), key=lambda t:t[1], reverse=True):
                        print_rank(f"\t\tdataset_name={dname}, count={count}")

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )
                    do_sync_step = is_last_step_and_steps_less_than_grad_acc or (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                    )
                    # Since we perform prefetching, we need to manually set sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(False)
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Newer Transformers represents this option as
                    # "no"/"non_padding"/"all" instead of a plain bool.
                    # Treating the string "no" as truthy flooded every rank's
                    # log once per step for this tuple-shaped multimodal batch.
                    token_count_mode = getattr(
                        self.args, "include_num_input_tokens_seen", False
                    )
                    if str(token_count_mode).lower() not in {
                        "false", "none", "no", "0", ""
                    }:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).cpu().item()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if getattr(self, 'use_apex', False):
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = torch.nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:

                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )


                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm


                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self._before_optimizer_step(model)

                        self.optimizer.step()

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        self._after_optimizer_step(model, optimizer_was_run)

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=time.time())
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=time.time())

            if self.control.should_training_stop:
                break

        if getattr(args, 'past_index', 0) and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            # `_sorted_checkpoints` is an internal Transformers helper and is
            # not present in every Transformers release. It is only needed for this
            # narrow best-checkpoint cleanup path, so do not call it
            # unconditionally after every successful training run.
            sorted_checkpoints = getattr(self, "_sorted_checkpoints", None)
            if sorted_checkpoints is None:
                logger.warning(
                    "Skipping final save_total_limit=1 best-checkpoint cleanup "
                    "because this Transformers version has no "
                    "Trainer._sorted_checkpoints; regular checkpoint rotation "
                    "has already run during saving."
                )
                checkpoints_sorted = []
            else:
                run_dir = self._get_output_dir(trial)
                checkpoints_sorted = sorted_checkpoints(
                    use_mtime=False,
                    output_dir=run_dir,
                )
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)



class GradCacheLateProcessTrainer(MMEBTrainer):
    """
    Adapted from gradcache repo.
    """
    def __init__(self, *args, **kwargs):

        self.data_args = kwargs.get("data_args", None)
        self.model_args = kwargs.get("model_args", None)
        self.training_args = kwargs.get("training_args", None)

        self.data_args_eval = copy.deepcopy(self.data_args)
        self.data_args_eval.dataset_name = self.data_args_eval.eval_dataset_name
        self.data_args_eval.subset_name = self.data_args_eval.eval_subset_name
        self.data_args_eval.dataset_split = "test"
        self.data_args_eval.image_dir = self.data_args_eval.eval_image_dir

        self.train_dataset = kwargs.get("train_dataset", None)
        del kwargs["data_args"]
        del kwargs["training_args"]



        self.max_length = kwargs.get("max_length", 512)
        if "max_length" in kwargs:
            del kwargs["max_length"]
        self.model_args = kwargs.get("model_args", None)
        if "model_args" in kwargs:
            del kwargs["model_args"]
        super(GradCacheLateProcessTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        if self.args.embedding_gradient_strategy == "llave":
            loss_fn_cls = (
                LLaVEDistributedContrastiveLoss
                if self.is_ddp
                else LLaVESimpleContrastiveLoss
            )
            loss_fn = loss_fn_cls(
                temperature=self.model.temperature,
                alpha=self.args.llave_alpha,
            )
        else:
            loss_fn_cls = (
                DistributedContrastiveLoss if self.is_ddp else SimpleContrastiveLoss
            )
            loss_fn = loss_fn_cls(temperature=self.model.temperature)
        if getattr(self.data_args, "negative_strategy", "b3") == "vlm2vec_inbatch":
            # The loss sees gathered representations under DDP. This assertion
            # therefore verifies the actual global negative pool, not merely
            # the local collator output.
            loss_fn.expected_target_per_query = 1
        elif (
            getattr(self.data_args, "negative_strategy", "b3") == "b3"
            and getattr(self.model_args, "moe_lora", False)
        ):
            loss_fn.expected_target_per_query = 6
            loss_fn.expected_layout_name = "b3_moe"
        elif self.args.embedding_gradient_strategy == "b3_ega":
            # B3+EGA is intentionally tied to the verified B3 layout: each
            # query is followed by one positive and five explicit negatives.
            loss_fn.expected_target_per_query = 6
        elif getattr(self.data_args, "b3_ablation", "none") != "none":
            ablation = self.data_args.b3_ablation
            loss_fn.expected_target_per_query = (
                1 if ablation == "no_instance_hn" else 6
            )
            loss_fn.expected_layout_name = f"b3_ablation:{ablation}"
        # process_fn = functools.partial(process_vlm_inputs_fns[self.args.model_backbone], processor=self.processing_class, max_length=self.max_length)

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_and_process_vlm_inputs,
            # process_fn=process_fn, #!preprocessing already done
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
            dynamic_limit=self.args.gc_dynamic_limit,
            embedding_gradient_strategy=self.args.embedding_gradient_strategy,
            ega_alpha=self.args.ega_alpha,
        )

        # Register reps capture hook for embedding monitoring (zero extra forward pass)
        self._monitor_reps = None
        self.gc.reps_callback = self._on_reps_captured
        self._inbatch_shape_validated = False
        self._b3_moe_shape_validated = False
        self._b3_ega_shape_validated = False

    def _on_reps_captured(self, all_reps):
        """Called by GradCache after forward_no_grad with [query_reps, target_reps]."""
        if (
            getattr(self.data_args, "negative_strategy", "b3") == "vlm2vec_inbatch"
            and not self._inbatch_shape_validated
        ):
            query_reps, target_reps = all_reps
            if query_reps.size(0) != target_reps.size(0):
                raise RuntimeError(
                    "VLM2Vec ordinary in-batch training requires exactly one "
                    "target per query, but captured representations have "
                    f"query={tuple(query_reps.shape)} target={tuple(target_reps.shape)}"
                )
            rank = dist.get_rank() if dist.is_initialized() else 0
            query_finite = int(torch.isfinite(query_reps).sum().item())
            target_finite = int(torch.isfinite(target_reps).sum().item())
            if query_finite != query_reps.numel() or target_finite != target_reps.numel():
                raise FloatingPointError(
                    "VLM2Vec ordinary in-batch representations are non-finite: "
                    f"query={query_finite}/{query_reps.numel()} "
                    f"target={target_finite}/{target_reps.numel()}"
                )
            print(
                "[VLM2VEC_INBATCH_REPS_OK] "
                f"rank={rank} local_pairs={query_reps.size(0)} "
                f"hidden_size={query_reps.size(1)} target_per_query=1 "
                f"query_finite={query_finite}/{query_reps.numel()} "
                f"target_finite={target_finite}/{target_reps.numel()} "
                f"query_norm_mean={query_reps.float().norm(dim=-1).mean().item():.8g} "
                f"target_norm_mean={target_reps.float().norm(dim=-1).mean().item():.8g}",
                flush=True,
            )
            self._inbatch_shape_validated = True
        if (
            getattr(self.data_args, "negative_strategy", "b3") == "b3"
            and getattr(self.model_args, "moe_lora", False)
            and not self._b3_moe_shape_validated
        ):
            query_reps, target_reps = all_reps
            expected_targets = query_reps.size(0) * 6
            if target_reps.size(0) != expected_targets:
                raise RuntimeError(
                    "B3+MoE captured an invalid target layout: "
                    f"query={tuple(query_reps.shape)} "
                    f"target={tuple(target_reps.shape)} "
                    f"expected_targets={expected_targets}"
                )
            query_finite = int(torch.isfinite(query_reps).sum().item())
            target_finite = int(torch.isfinite(target_reps).sum().item())
            if (
                query_finite != query_reps.numel()
                or target_finite != target_reps.numel()
            ):
                raise FloatingPointError(
                    "B3+MoE representations are non-finite: "
                    f"query={query_finite}/{query_reps.numel()} "
                    f"target={target_finite}/{target_reps.numel()}"
                )
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                "[B3_MOE_REPS_OK] "
                f"rank={rank} local_queries={query_reps.size(0)} "
                f"local_targets={target_reps.size(0)} target_per_query=6 "
                "explicit_hard_negatives_per_query=5 "
                f"hidden_size={query_reps.size(1)} "
                f"query_finite={query_finite}/{query_reps.numel()} "
                f"target_finite={target_finite}/{target_reps.numel()} "
                f"query_norm_mean={query_reps.float().norm(dim=-1).mean().item():.8g} "
                f"target_norm_mean={target_reps.float().norm(dim=-1).mean().item():.8g}",
                flush=True,
            )
            self._b3_moe_shape_validated = True
        if (
            self.args.embedding_gradient_strategy == "b3_ega"
            and not self._b3_ega_shape_validated
        ):
            query_reps, target_reps = all_reps
            expected_targets = query_reps.size(0) * 6
            if target_reps.size(0) != expected_targets:
                raise RuntimeError(
                    "B3+EGA captured an invalid target layout: "
                    f"query={tuple(query_reps.shape)} "
                    f"target={tuple(target_reps.shape)} "
                    f"expected_targets={expected_targets}"
                )
            query_finite = int(torch.isfinite(query_reps).sum().item())
            target_finite = int(torch.isfinite(target_reps).sum().item())
            if query_finite != query_reps.numel() or target_finite != target_reps.numel():
                raise FloatingPointError(
                    "B3+EGA representations are non-finite: "
                    f"query={query_finite}/{query_reps.numel()} "
                    f"target={target_finite}/{target_reps.numel()}"
                )
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                "[B3_EGA_REPS_OK] "
                f"rank={rank} local_queries={query_reps.size(0)} "
                f"local_targets={target_reps.size(0)} target_per_query=6 "
                "explicit_hard_negatives_per_query=5 "
                f"hidden_size={query_reps.size(1)} "
                f"query_finite={query_finite}/{query_reps.numel()} "
                f"target_finite={target_finite}/{target_reps.numel()} "
                f"query_norm_mean={query_reps.float().norm(dim=-1).mean().item():.8g} "
                f"target_norm_mean={target_reps.float().norm(dim=-1).mean().item():.8g}",
                flush=True,
            )
            self._b3_ega_shape_validated = True
        self._monitor_reps = [r.detach() for r in all_reps]

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        model.train()
        queries, targets = inputs
        queries, targets = {'qry': queries}, {'tgt': targets}

        # With torchrun + newer Transformers, the process rank is supplied via
        # LOCAL_RANK and TrainingArguments.local_rank can still be -1 here.
        # The model is already wrapped in DDP, so checking the parsed CLI field
        # incorrectly disabled GradCache's distributed synchronization.
        _distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        self.gc.models = [model, model]

        loss = self.gc(queries, targets, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        print(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        _validate_embedding_token_adapter_state(self, state_dict)
        self.model.encoder.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            **_embedding_token_peft_save_kwargs(self),
        )

        processor = getattr(self, "processing_class", None)
        if processor is None:
            # Compatibility with Transformers versions where Trainer still
            # exposed the processor through the deprecated tokenizer alias.
            processor = getattr(self, "tokenizer", None)
        if processor is not None:
            processor.save_pretrained(output_dir)

        _write_embedding_token_checkpoint_metadata(self, output_dir)
        _write_qwen3_5_bidirectional_checkpoint_metadata(self, output_dir)
        _write_qwen3_5_prompt_checkpoint_metadata(self, output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.encoder.config.to_json_file(os.path.join(output_dir, 'config.json'))
        b3_ablation = getattr(self.data_args, "b3_ablation", "none")
        if b3_ablation != "none":
            ablation_checkpoint_config = {
                "method": "qwen2_vl_b3_ablation",
                "ablation": b3_ablation,
                "target_per_query": (
                    1 if b3_ablation == "no_instance_hn" else 6
                ),
            }
            with open(
                os.path.join(output_dir, "b3_ablation_config.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    ablation_checkpoint_config,
                    handle,
                    sort_keys=True,
                    indent=2,
                )
                handle.write("\n")
        if self.args.embedding_gradient_strategy == "ega":
            ega_checkpoint_config = {
                "method": "ega",
                "alpha": float(self.args.ega_alpha),
                "temperature": float(self.model.temperature),
                "negative_strategy": self.data_args.negative_strategy,
                "target_per_query": 1,
            }
            with open(
                os.path.join(output_dir, "ega_training_config.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(ega_checkpoint_config, handle, sort_keys=True, indent=2)
        if self.args.embedding_gradient_strategy == "b3_ega":
            b3_ega_checkpoint_config = {
                "method": "b3_ega",
                "alpha": float(self.args.ega_alpha),
                "temperature": float(self.model.temperature),
                "negative_strategy": self.data_args.negative_strategy,
                "target_per_query": 6,
                "explicit_hard_negatives_per_query": 5,
                "negative_scope": "all_global_targets_except_aligned_positive",
                "preserve_negative_mass": True,
            }
            with open(
                os.path.join(output_dir, "b3_ega_training_config.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    b3_ega_checkpoint_config,
                    handle,
                    sort_keys=True,
                    indent=2,
                )
                handle.write("\n")
        if (
            getattr(self.model_args, "moe_lora", False)
            and self.data_args.negative_strategy == "b3"
        ):
            b3_moe_checkpoint_config = {
                "method": "b3_tsembed_moe_dora_only",
                "negative_strategy": "b3",
                "embedding_gradient_strategy": "info_nce",
                "target_per_query": 6,
                "explicit_hard_negatives_per_query": 5,
                "num_experts": int(self.model_args.moe_num_experts),
                "rank_per_expert": int(self.model_args.lora_r),
                "alpha_per_expert": float(self.model_args.lora_alpha),
                "router_temperature": float(
                    self.model_args.moe_router_temperature
                ),
                "eans": False,
                "two_stage_training": False,
            }
            with open(
                os.path.join(output_dir, "b3_moe_training_config.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    b3_moe_checkpoint_config,
                    handle,
                    sort_keys=True,
                    indent=2,
                )
                handle.write("\n")
        if self.args.embedding_gradient_strategy == "llave":
            llave_checkpoint_config = {
                "method": "llave_hardness_weighted_contrastive_learning",
                "alpha": float(self.args.llave_alpha),
                "temperature": float(self.model.temperature),
                "negative_strategy": self.data_args.negative_strategy,
                "target_per_query": 1,
                "reward_model": "policy_model_synchronized_each_step",
                "reward_gradient": "stopped",
                "reward_scope": "negative_pairs_only",
                "directions": ["query_to_target", "target_to_query"],
                "direction_reduction": "mean",
                "cross_device_negatives": bool(self.is_ddp),
            }
            with open(
                os.path.join(output_dir, "llave_training_config.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(llave_checkpoint_config, handle, sort_keys=True, indent=2)
                handle.write("\n")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Custom evaluation logic to handle a different eval dataset format and compute accuracy.
        """
        # Lazy imports for evaluation-only dependencies
        from src.training.data.collator import EvalCollator

        def get_pred(qry_t, tgt_t, normalization=False):
            qry_t = np.asarray(qry_t)
            tgt_t = np.asarray(tgt_t)
            if normalization:
                qry_norm = np.linalg.norm(qry_t)
                tgt_norms = np.linalg.norm(tgt_t, axis=1)
                scores = np.dot(tgt_t, qry_t) / (tgt_norms * qry_norm)
            else:
                scores = np.dot(tgt_t, qry_t)
            return scores, int(np.argmax(scores))

        # print(self.tokenizer, flush=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        device = self.args.device

        # Run evaluation only on rank 0
        self.model.eval()
        processor = self.processing_class


        data_args = self.data_args_eval
        model_args = self.model_args
        training_args = self.training_args


        eval_collator = EvalCollator(
            data_args=data_args,
            model_args=model_args,
            processor=processor,
        )



        accuracy_list = []
        metrics = {}
        accuracy_list_ph_all = [[]]*world_size
        # for idx, subset in enumerate(data_args.subset_name):
        subset=data_args.subset_name[0]
        if rank == 0:
            accuracy_list_ph_all[rank].append(0.0)
            # accuracy_list.append(rank*100.0+100)
            eval_qry_dataset = EvalDataset(
                data_args=data_args,
                model_args=model_args,
                subset=subset,
                text_field="qry_text",
                img_path_field="qry_img_path",
            )
            eval_tgt_dataset = EvalDataset(
                data_args=data_args,
                model_args=model_args,
                subset=subset,
                text_field="tgt_text",
                img_path_field="tgt_img_path",
                mod_instruction=POS_MOD_DICT.get(subset, None) if data_args.tgt_prefix_mod else None
            )

            eval_qry_loader = DataLoader(
                eval_qry_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=eval_collator,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
            eval_tgt_loader = DataLoader(
                eval_tgt_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=eval_collator,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )

            print("Rank", rank, "tgt_length", len(eval_tgt_dataset))

            encoded_tensor = []
            with torch.no_grad():
                for batch in tqdm(eval_qry_loader, desc=f"Encode query - {subset}"):
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = self.model(qry=batch)
                    encoded_tensor.append(output["qry_reps"].cpu().detach().float())
            encoded_tensor = np.concatenate(encoded_tensor)
            qry_tensor, qry_index = encoded_tensor, eval_qry_dataset.paired_data


            encoded_tensor = []
            with torch.no_grad():
                for batch in tqdm(eval_tgt_loader, desc=f"Encode target - {subset}"):
                    batch = batch_to_device(batch, training_args.device)
                    output = self.model(tgt=batch)
                    encoded_tensor.append(output["tgt_reps"].cpu().detach().float())

            encoded_tensor = np.concatenate(encoded_tensor)
            tgt_tensor, tgt_index = encoded_tensor, eval_tgt_dataset.paired_data

            # build a map for dedup
            qry_key2emb, tgt_key2emb = OrderedDict(), OrderedDict()
            for qry_t, tt in zip(qry_tensor, qry_index):
                text, img_path = tt["text"], tt["img_path"]
                qry_key2emb[(text, img_path)] = qry_t
            for tgt_t, tt in zip(tgt_tensor, tgt_index):
                text, img_path = tt["text"], tt["img_path"]
                tgt_key2emb[(text, img_path)] = tgt_t
            print(f'tgt_tensor.shape = {tgt_tensor.shape}')
            # print(f'len(tkey_to_rowid) = {len(tgtkey_to_rowid)}')
            # print(f'len(rowid_to_tkey) = {len(rowid_to_tgtkey)}')
            print(f'len(tgt_key2emb) = {len(tgt_key2emb)}')


            eval_data = load_dataset(
                data_args.dataset_name,
                subset,
                split=data_args.dataset_split,
            )

            n_correct = 0
            all_pred = []
            for row in tqdm(eval_data, desc=f"calculate score for {subset}"):
                # with time_block("Query vector lookup"):
                qry_t = qry_key2emb[(row["qry_text"], row["qry_img_path"])]  # (dim,)
                tgt_t, all_candidates = [], []
                # with time_block("Target vector & candidate prep"):
                for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                    tgt_t.append(tgt_key2emb[tt])
                    all_candidates.append(tt)
                qry_t = torch.from_numpy(np.array(qry_t))
                tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
                scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
                if pred == 0:
                    n_correct += 1
                all_pred.append(all_candidates[pred])


            accuracy = n_correct/len(eval_data)
            metrics[f"{subset}_accuracy"] = accuracy
            accuracy_list.append(accuracy)

            accuracy_list_rank_tensor = torch.tensor(accuracy_list, device=device)

            print("World size", world_size, "Rank", rank, "Accuracy_list", accuracy_list_rank_tensor,  flush=True)

            metrics["eval_accuracy"]=torch.mean(accuracy_list_rank_tensor).item()
            self.log(metrics)

        self.model.train()
        return metrics  # Return the metrics
