"""W&B Callback for YOLO training."""

from typing import Optional, Dict, Any
from .config import WandbConfig


class WandbCallback:
    """
    Callback for logging to W&B during training.
    
    Can be used with custom training loops or integrated with Ultralytics.
    """
    
    def __init__(self, config: WandbConfig):
        self.config = config
        self._run = None
    
    def on_train_start(self, trainer: Any) -> None:
        """Called at start of training."""
        import wandb
        
        # Merge trainer config with user config
        full_config = {
            **self.config.config,
            "model": getattr(trainer, "model_name", "unknown"),
            "epochs": getattr(trainer, "epochs", 0),
            "batch_size": getattr(trainer, "batch_size", 0),
        }
        
        self._run = wandb.init(
            **self.config.to_init_kwargs(),
            config=full_config,
        )
    
    def on_train_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at end of each epoch."""
        import wandb
        
        if self._run:
            wandb.log(metrics, step=epoch)
    
    def on_train_end(self, results: Any) -> None:
        """Called at end of training."""
        import wandb
        
        if self._run:
            # Log final metrics
            if hasattr(results, "box"):
                wandb.log({
                    "final/mAP50": results.box.map50,
                    "final/mAP50-95": results.box.map,
                })
            
            wandb.finish()
            self._run = None


def setup_wandb_for_yolo(config: Optional[WandbConfig] = None) -> None:
    """
    Enable W&B for Ultralytics YOLO training.
    
    This enables Ultralytics' native W&B integration which
    automatically logs training metrics, model artifacts, etc.
    
    Args:
        config: Optional WandbConfig for custom settings
    """
    from ultralytics import settings
    
    # Enable native W&B integration
    settings.update(wandb=True)
    
    # If custom config provided, set environment variables
    if config:
        import os
        if config.project:
            os.environ["WANDB_PROJECT"] = config.project
        if config.entity:
            os.environ["WANDB_ENTITY"] = config.entity
        if config.name:
            os.environ["WANDB_NAME"] = config.name
