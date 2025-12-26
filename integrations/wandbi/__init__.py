"""
W&B Integration - Composable Weights & Biases integration for training.
"""

from .config import WandbConfig
from .callback import WandbCallback, setup_wandb_for_yolo

__all__ = ["WandbConfig", "WandbCallback", "setup_wandb_for_yolo"]
