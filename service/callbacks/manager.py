"""
Callback manager for coordinating multiple training callbacks.
"""

from typing import Any, Dict, List

from service.callbacks.base import TrainingCallback


class CallbackManager:
    """
    Manages a collection of training callbacks.

    Usage:
        from service.callbacks import CallbackManager, TensorBoardCallback

        manager = CallbackManager()
        manager.add(TensorBoardCallback("runs/experiment1"))

        # During training
        manager.on_epoch_end({"epoch": 1, "loss": 0.5, "mAP50": 0.85})
    """

    def __init__(self):
        self.callbacks: List[TrainingCallback] = []

    def add(self, callback: TrainingCallback) -> "CallbackManager":
        """Add a callback. Returns self for chaining."""
        self.callbacks.append(callback)
        return self

    def remove(self, callback: TrainingCallback) -> None:
        """Remove a callback."""
        self.callbacks.remove(callback)

    def on_train_start(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_start(metrics)

    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(metrics)

    def on_epoch_start(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(metrics)

    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(metrics)

    def on_batch_end(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(metrics)
