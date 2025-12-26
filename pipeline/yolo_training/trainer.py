"""
YOLO Trainer - Composable YOLO training processor.

Uses Modal volumes for data and checkpoint persistence.
"""

import modal
from typing import Dict, Any, Optional
from pathlib import Path

from .config import YOLOConfig, ModalYOLOConfig


def create_yolo_training_app(
    yolo_config: Optional[YOLOConfig] = None,
    modal_config: Optional[ModalYOLOConfig] = None,
    enable_wandb: bool = True
) -> modal.App:
    """
    Create a Modal app for YOLO training.
    
    This is a factory function that creates a configured Modal app
    with all necessary volumes, secrets, and functions.
    
    Args:
        yolo_config: YOLO training configuration
        modal_config: Modal-specific configuration
        enable_wandb: Enable W&B logging
        
    Returns:
        Configured Modal app
    """
    yolo_config = yolo_config or YOLOConfig()
    modal_config = modal_config or ModalYOLOConfig()
    
    # Create app
    app = modal.App(modal_config.app_name)
    
    # Setup volumes
    data_volume = modal.Volume.from_name(
        modal_config.data_volume, 
        create_if_missing=False
    )
    checkpoint_volume = modal.Volume.from_name(
        modal_config.checkpoint_volume,
        create_if_missing=True
    )
    
    # Build image
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(*modal_config.pip_packages)
        .apt_install(*modal_config.apt_packages)
    )
    
    # Setup secrets
    secrets = []
    if enable_wandb:
        secrets.append(modal.Secret.from_name(modal_config.wandb_secret))
    
    @app.function(
        image=image,
        gpu=modal_config.gpu_type,
        timeout=modal_config.timeout,
        volumes={
            "/data": data_volume,
            "/checkpoints": checkpoint_volume
        },
        secrets=secrets,
    )
    def train(
        epochs: int = yolo_config.epochs,
        batch_size: int = yolo_config.batch_size,
        model: str = yolo_config.model,
    ) -> Dict[str, Any]:
        """Train YOLO model on Modal GPU."""
        from ultralytics import YOLO, settings
        from pathlib import Path
        
        # Enable W&B if available
        if enable_wandb:
            settings.update(wandb=True)
        
        print(f"🚀 Starting YOLO Training")
        print(f"GPU: {modal_config.gpu_type}, Batch: {batch_size}, Epochs: {epochs}")
        
        # Check for checkpoint
        checkpoint_path = f"{yolo_config.project}/{yolo_config.name}/weights/last.pt"
        if Path(checkpoint_path).exists():
            print(f"📦 Resuming from: {checkpoint_path}")
            yolo_model = YOLO(checkpoint_path)
        else:
            print(f"📦 Loading base model: {model}")
            yolo_model = YOLO(model)
        
        # Build training kwargs
        train_kwargs = yolo_config.to_train_kwargs()
        train_kwargs.update({
            "epochs": epochs,
            "batch": batch_size,
        })
        
        # Train
        results = yolo_model.train(**train_kwargs)
        
        # Commit checkpoints
        checkpoint_volume.commit()
        
        # Extract metrics
        metrics = {
            "epochs": epochs,
            "mAP50": float(results.box.map50) if hasattr(results, 'box') else 0,
            "mAP50-95": float(results.box.map) if hasattr(results, 'box') else 0,
        }
        
        print(f"\n✅ Training Complete!")
        print(f"Metrics: {metrics}")
        
        return metrics
    
    @app.function(
        image=image,
        gpu="T4",
        timeout=600,
        volumes={
            "/data": data_volume,
            "/checkpoints": checkpoint_volume
        },
    )
    def test(image_bytes: bytes) -> Dict[str, Any]:
        """Test YOLO on an image."""
        from ultralytics import YOLO
        import tempfile
        
        model = YOLO(f"{yolo_config.project}/{yolo_config.name}/weights/best.pt")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name
        
        results = model(temp_path, conf=0.25)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })
        
        return {"detections": detections, "count": len(detections)}
    
    @app.local_entrypoint()
    def main(epochs: int = 50, batch_size: int = 32):
        """Train YOLO for USD detection."""
        print(f"🎯 YOLO USD Detection Training ({epochs} epochs)")
        
        result = train.remote(epochs=epochs, batch_size=batch_size)
        print(f"\n✅ Training Complete!")
        print(f"Results: {result}")
    
    return app


class YOLOTrainer:
    """
    Composable YOLO trainer using Modal.
    
    This class wraps the Modal app creation for easier integration
    with the ModalPipeline system.
    
    Usage:
        trainer = YOLOTrainer(
            yolo_config=YOLOConfig(epochs=50),
            modal_config=ModalYOLOConfig(gpu_type="A100")
        )
        
        # Run training
        result = trainer.run_remote(epochs=50)
    """
    
    def __init__(
        self,
        yolo_config: Optional[YOLOConfig] = None,
        modal_config: Optional[ModalYOLOConfig] = None,
        enable_wandb: bool = True
    ):
        self.yolo_config = yolo_config or YOLOConfig()
        self.modal_config = modal_config or ModalYOLOConfig()
        self.enable_wandb = enable_wandb
        
        self.app = create_yolo_training_app(
            self.yolo_config,
            self.modal_config,
            self.enable_wandb
        )
    
    def run_remote(self, **kwargs) -> Dict[str, Any]:
        """Run training on Modal."""
        # The app.local_entrypoint handles this
        # This would require modal run to execute
        raise NotImplementedError(
            "Use 'modal run' with the generated app script. "
            "See scripts/train_yolo11.py for example."
        )
