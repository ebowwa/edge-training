"""
Metrics logging callback for console and file output.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from service.callbacks.base import TrainingCallback


class MetricsLoggerCallback(TrainingCallback):
    """
    Logs metrics to console and optionally saves to file.

    Args:
        output_dir: Directory to save metrics JSON (optional)
        log_every_n_epochs: Log every N epochs (default: 1)
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        log_every_n_epochs: int = 1,
    ):
        self.output_dir = output_dir
        self.log_every_n_epochs = log_every_n_epochs
        self.history: List[Dict[str, Any]] = []

    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        self.history.append(metrics.copy())

        epoch = metrics.get("epoch", 0)
        if epoch % self.log_every_n_epochs == 0:
            # Format key metrics for logging
            parts = [f"Epoch {epoch}"]
            for key in ["loss", "mAP50", "mAP50-95", "precision", "recall"]:
                if key in metrics:
                    parts.append(f"{key}={metrics[key]:.4f}")
            logging.info(" | ".join(parts))

    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        if self.output_dir:
            import json
            output_path = Path(self.output_dir) / "training_metrics.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(self.history, f, indent=2)
            logging.info(f"Metrics saved to {output_path}")
