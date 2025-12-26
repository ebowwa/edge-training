"""
TensorBoard logging callback.
"""

import logging
from typing import Any, Dict

# Optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from service.callbacks.base import TrainingCallback


class TensorBoardCallback(TrainingCallback):
    """
    Logs training metrics to TensorBoard.

    Inspired by RF-DETR's MetricsTensorBoardSink.

    Args:
        log_dir: Directory for TensorBoard logs
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None

        if SummaryWriter is None:
            logging.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
        else:
            self.writer = SummaryWriter(log_dir=log_dir)
            logging.info(
                f"TensorBoard logging initialized. "
                f"Run: tensorboard --logdir {log_dir}"
            )

    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        if self.writer is None:
            return

        epoch = metrics.get("epoch", 0)

        # Log common metrics
        for key in ["loss", "train_loss", "val_loss", "box_loss", "cls_loss"]:
            if key in metrics:
                self.writer.add_scalar(f"Loss/{key}", metrics[key], epoch)

        for key in ["mAP50", "mAP50-95", "precision", "recall"]:
            if key in metrics:
                self.writer.add_scalar(f"Metrics/{key}", metrics[key], epoch)

        if "lr" in metrics:
            self.writer.add_scalar("Training/learning_rate", metrics["lr"], epoch)

        self.writer.flush()

    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        if self.writer:
            self.writer.close()
