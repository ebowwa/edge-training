"""
Weights & Biases logging callback.
"""

import logging
from typing import Any, Dict, Optional

# Optional dependencies
try:
    import wandb
except ImportError:
    wandb = None

from service.callbacks.base import TrainingCallback


class WandBCallback(TrainingCallback):
    """
    Logs training metrics to Weights & Biases.

    Inspired by RF-DETR's MetricsWandBSink.

    Args:
        project: W&B project name
        run_name: Name for this run (optional)
        config: Training config to log (optional)
    """

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.project = project
        self.run_name = run_name
        self.run = None

        if wandb is None:
            logging.warning(
                "W&B not available. Install with: pip install wandb"
            )
        else:
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config,
            )
            logging.info(f"W&B logging initialized: {wandb.run.url}")

    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        if self.run is None or wandb is None:
            return

        # W&B handles epoch automatically via step
        log_dict = {}

        for key in ["loss", "train_loss", "val_loss", "mAP50", "mAP50-95",
                    "precision", "recall", "lr"]:
            if key in metrics:
                log_dict[key] = metrics[key]

        if log_dict:
            wandb.log(log_dict, step=metrics.get("epoch", None))

    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        if self.run:
            self.run.finish()
