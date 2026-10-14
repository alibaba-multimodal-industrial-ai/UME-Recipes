"""Dedicated entry point for UniME-V2 training on verified VLM backbones."""

import train as standard_train
from src.training.unime_v2 import (
    UniMEV2DataCollator,
    UniMEV2Trainer,
    init_unime_v2_dataset,
)


standard_train.init_concat_dataset = init_unime_v2_dataset
standard_train.MultimodalDataCollator = UniMEV2DataCollator
standard_train.GradCacheLateProcessTrainer = UniMEV2Trainer


if __name__ == "__main__":
    standard_train.main()
