"""
Training Loops & Loss Functions
"""

from .losses import FocalLoss, BCEDiceLoss
from .trainer import RetinoblastomaDedicatedTrainer

CrossValidationTrainer = RetinoblastomaDedicatedTrainer

__all__ = [
    "FocalLoss",
    "BCEDiceLoss",
    "RetinoblastomaDedicatedTrainer",
    "CrossValidationTrainer",
]
