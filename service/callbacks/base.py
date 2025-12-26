"""
Base callback class for training hooks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class TrainingCallback(ABC):
    """
    Base class for training callbacks.

    Subclasses should implement the hooks they need.
    All hooks receive a `metrics` dict with training state.
    """

    def on_train_start(self, metrics: Dict[str, Any]) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, metrics: Dict[str, Any]) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_end(self, metrics: Dict[str, Any]) -> None:
        """Called at the end of each batch."""
        pass
