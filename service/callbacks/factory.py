"""
Factory function for creating default callback configurations.
"""

from typing import Optional

from service.callbacks.base import TrainingCallback
from service.callbacks.early_stopping import EarlyStoppingCallback
from service.callbacks.manager import CallbackManager
from service.callbacks.metrics_logger import MetricsLoggerCallback
from service.callbacks.tensorboard import TensorBoardCallback
from service.callbacks.wandb import WandBCallback


def create_default_callbacks(
    output_dir: str,
    tensorboard: bool = True,
    wandb_project: Optional[str] = None,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
) -> CallbackManager:
    """
    Create a CallbackManager with common callbacks preconfigured.

    Args:
        output_dir: Directory for logs and outputs
        tensorboard: Enable TensorBoard logging
        wandb_project: W&B project name (enables W&B if set)
        early_stopping: Enable early stopping
        early_stopping_patience: Epochs to wait before stopping

    Returns:
        Configured CallbackManager
    """
    manager = CallbackManager()

    # Always add metrics logger
    manager.add(MetricsLoggerCallback(output_dir=output_dir))

    if tensorboard:
        manager.add(TensorBoardCallback(log_dir=output_dir))

    if wandb_project:
        manager.add(WandBCallback(project=wandb_project))

    if early_stopping:
        manager.add(EarlyStoppingCallback(patience=early_stopping_patience))

    return manager
