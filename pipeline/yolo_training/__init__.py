"""
YOLO Training Pipeline - Composable YOLO training on Modal.

Usage:
    from pipeline.yolo_training import YOLOTrainer, YOLOConfig
    
    config = YOLOConfig(
        model="yolo11l.pt",
        epochs=50,
        batch_size=32
    )
    
    trainer = YOLOTrainer(
        volume_name="usd-dataset-test",
        checkpoint_volume="usd-checkpoints",
        config=config
    )
    
    result = trainer.run_remote()
"""

from .config import YOLOConfig, ModalYOLOConfig
from .trainer import YOLOTrainer, create_yolo_training_app

__all__ = ["YOLOConfig", "ModalYOLOConfig", "YOLOTrainer", "create_yolo_training_app"]
