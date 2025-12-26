"""W&B Configuration."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class WandbConfig:
    """
    Configuration for Weights & Biases integration.
    
    Usage:
        config = WandbConfig(
            project="usd-detection",
            name="yolo11-50ep",
            tags=["yolo11", "usd", "a100"]
        )
    """
    
    # Project settings
    project: str = "usd-detection"
    entity: Optional[str] = None
    name: Optional[str] = None
    
    # Run settings
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    group: Optional[str] = None
    
    # Logging settings
    log_model: bool = True
    log_dataset: bool = False
    log_predictions: bool = True
    
    # Config to log
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Modal secret name (for API key)
    secret_name: str = "wandb-secret"
    
    def to_init_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for wandb.init()."""
        return {
            "project": self.project,
            "entity": self.entity,
            "name": self.name,
            "tags": self.tags,
            "notes": self.notes,
            "group": self.group,
            "config": self.config,
        }
