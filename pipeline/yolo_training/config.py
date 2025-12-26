"""YOLO Training Configuration."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class YOLOConfig:
    """
    Configuration for YOLO training.
    
    Usage:
        config = YOLOConfig(
            model="yolo11l.pt",
            epochs=50,
            batch_size=32
        )
    """
    
    # Model settings
    model: str = "yolo11l.pt"
    imgsz: int = 640
    
    # Training settings
    epochs: int = 50
    batch_size: int = 32
    
    # Data settings
    data_yaml: str = "/data/data.yaml"
    
    # Performance settings (Modal-safe defaults)
    workers: int = 0  # Avoid multiprocessing issues in containers
    cache: bool = False
    amp: bool = True
    
    # Checkpoint settings
    project: str = "/checkpoints/usd-yolo11"
    name: str = "train"
    exist_ok: bool = True
    resume: bool = False
    save: bool = True
    save_period: int = 10
    
    # Logging
    verbose: bool = True
    plots: bool = True
    
    def to_train_kwargs(self) -> dict:
        """Get kwargs for model.train()."""
        return {
            "data": self.data_yaml,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch_size,
            "device": 0,
            "project": self.project,
            "name": self.name,
            "exist_ok": self.exist_ok,
            "resume": self.resume,
            "amp": self.amp,
            "cache": self.cache,
            "workers": self.workers,
            "verbose": self.verbose,
            "plots": self.plots,
            "save": self.save,
            "save_period": self.save_period,
        }


@dataclass
class ModalYOLOConfig:
    """
    Modal-specific configuration for YOLO training.
    """
    
    # GPU settings
    gpu_type: str = "A100"
    timeout: int = 7200  # 2 hours
    
    # Volume settings
    data_volume: str = "usd-dataset-test"
    checkpoint_volume: str = "usd-checkpoints"
    
    # Secrets
    wandb_secret: str = "wandb-secret"
    
    # App settings
    app_name: str = "usd-yolo11-training"
    
    # Dependencies
    pip_packages: List[str] = field(default_factory=lambda: [
        "ultralytics",
        "torch", 
        "torchvision",
        "opencv-python-headless",
        "pyyaml",
        "wandb"
    ])
    
    apt_packages: List[str] = field(default_factory=lambda: [
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ])
