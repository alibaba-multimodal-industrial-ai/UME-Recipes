from datasets.distributed import split_dataset_by_node

from src.training.data.dataset import AutoPairDataset
from datasets import load_dataset, concatenate_datasets
from src.training.utils import print_master
import torch


def init_concat_dataset(dataset_config, model_args, data_args, training_args):
    weights = [d['weight'] for d in dataset_config.values()]
    w_sum = sum(weights)
    probs = [w / w_sum for w in weights]
    world_size = torch.distributed.get_world_size()
    train_datasets = []
    for data_idx, (global_dataset_name, source_config) in enumerate(dataset_config.items()):
        prompt_task = None
        if getattr(model_args, "qwen3_5_prompt_profile", "legacy") != "legacy":
            from src.training.qwen3_5_prompt import task_for_dataset

            prompt_task = task_for_dataset(global_dataset_name)
            if prompt_task == "generic":
                raise ValueError(
                    "Qwen3.5 prompt-v2 training requires a canonical MMEB "
                    f"dataset name, but {global_dataset_name!r} maps to generic"
                )
        # Keep the physical subset name for locating the preprocessed B3
        # parquet files, but pass the YAML key separately as the canonical
        # dataset identity used by role-aware prompt routing.
        source_config = dict(source_config)
        source_config["global_dataset_name"] = global_dataset_name
        train_dataset = AutoPairDataset.instantiate(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            **source_config,
        )
        print_master(
            f"\t\tDataset#{data_idx} "
            f"(dataset_parser={source_config.get('dataset_parser', 'n/a')}): "
            f"{global_dataset_name}, prompt_task={prompt_task}, "
            f"num_rows={train_dataset.num_rows}, "
            f"prob={probs[data_idx] * 100.0}"
        )
        train_datasets.append(train_dataset)

    if training_args.interleave_batch_size and training_args.interleave_batch_size <= 1.0:
        interleave_batch_size = training_args.per_device_train_batch_size * world_size * training_args.interleave_batch_size
    else:
        interleave_batch_size = training_args.interleave_batch_size
    total_num_rows = sum([d.num_rows for d in train_datasets])
    print_master(f"\nInitializing interleave datasets:"
                 f"\n\t\tworld_size={torch.distributed.get_world_size()}"
                 f"\n\t\ttotal num rows={total_num_rows}"
                 f"\n\t\tglobal batch size={training_args.per_device_train_batch_size * world_size}"
                 f"\n\t\testimated num step per epoch={total_num_rows/(training_args.per_device_train_batch_size * world_size)}"
                 f"\n\t\tinterleave_batch_size={interleave_batch_size}"
                 )
    assert total_num_rows >= (training_args.per_device_train_batch_size * world_size), \
        f"total_num_rows(={total_num_rows}) must be greater than or equal to global batch size (={training_args.per_device_train_batch_size * world_size}), since the last batch will be dropped."

    dataset_lens = [len(td) for td in train_datasets]
    train_dataset = concatenate_datasets(train_datasets)

    return train_dataset, dataset_lens
