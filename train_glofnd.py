"""Dedicated Qwen2-VL entry point for GloFND-filtered InfoNCE."""

import train as standard_train
from src.training.glofnd import GloFNDDataCollator, GloFNDTrainer


standard_train.MultimodalDataCollator = GloFNDDataCollator
standard_train.GradCacheLateProcessTrainer = GloFNDTrainer


if __name__ == "__main__":
    standard_train.main()

