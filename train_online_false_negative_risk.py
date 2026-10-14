"""Dedicated entry point for Qwen2-VL/Qwen3.5 online FN-risk deletion."""

import train as standard_train
from src.training.online_false_negative_risk import (
    OnlineFalseNegativeRiskCollator,
    OnlineFalseNegativeRiskTrainer,
)


standard_train.MultimodalDataCollator = OnlineFalseNegativeRiskCollator
standard_train.GradCacheLateProcessTrainer = OnlineFalseNegativeRiskTrainer


if __name__ == "__main__":
    standard_train.main()
