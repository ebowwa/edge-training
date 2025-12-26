"""
Early stopping callback.
"""

import logging
from typing import Any, Callable, Dict, Optional

from service.callbacks.base import TrainingCallback


class EarlyStoppingCallback(TrainingCallback):
    """
    Stops training when a metric stops improving.

    Inspired by RF-DETR's EarlyStoppingCallback.

    Args:
        monitor: Metric to monitor (default: "mAP50")
        patience: Epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: "max" for metrics to maximize, "min" to minimize
        stop_training_fn: Callable to invoke when stopping (optional)
    """

    def __init__(
        self,
        monitor: str = "mAP50",
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        stop_training_fn: Optional[Callable] = None,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.stop_training_fn = stop_training_fn

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        else:
            return current < self.best_value - self.min_delta

    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        if self.monitor not in metrics:
            logging.warning(
                f"EarlyStopping: metric '{self.monitor}' not found in metrics"
            )
            return

        current = metrics[self.monitor]
        epoch = metrics.get("epoch", "?")

        if self._is_improvement(current):
            self.best_value = current
            self.counter = 0
            logging.info(
                f"EarlyStopping: {self.monitor} improved to {current:.4f}"
            )
        else:
            self.counter += 1
            logging.info(
                f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs "
                f"(best: {self.best_value:.4f}, current: {current:.4f})"
            )

        if self.counter >= self.patience:
            self.should_stop = True
            logging.warning(
                f"EarlyStopping triggered at epoch {epoch}: "
                f"No improvement in {self.monitor} for {self.patience} epochs"
            )
            if self.stop_training_fn:
                self.stop_training_fn()
