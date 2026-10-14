"""Dedicated entry point for HOBIT-C batch construction."""

import train as standard_train
from src.training.hobit import HOBITCollator, HOBITSampler, HOBITTrainer

standard_train.MultimodalDataCollator = HOBITCollator
standard_train.VLM2VecInBatchSampler = HOBITSampler
standard_train.GradCacheLateProcessTrainer = HOBITTrainer

if __name__ == "__main__":
    standard_train.main()
