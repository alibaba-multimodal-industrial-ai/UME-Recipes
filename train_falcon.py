"""Dedicated entry point for the isolated Qwen2-VL FALCON experiment."""

import train as standard_train
from src.training.falcon import FALCONCollator, FALCONSampler, FALCONTrainer


# Patch only this process.  Standard train.py and every existing launcher keep
# their historical sampler, collator and trainer classes.
standard_train.MultimodalDataCollator = FALCONCollator
standard_train.VLM2VecInBatchSampler = FALCONSampler
standard_train.GradCacheLateProcessTrainer = FALCONTrainer


if __name__ == "__main__":
    standard_train.main()
