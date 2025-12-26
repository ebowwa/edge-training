"""
YOLO11 Training on Modal - Composable Configuration

Uses pipeline/yolo_training/config.py for all settings.
Uses Ultralytics native W&B integration.
"""

import modal
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.yolo_training.config import YOLOConfig, ModalYOLOConfig

# Load config
modal_config = ModalYOLOConfig()
yolo_config = YOLOConfig()

# Modal setup
app = modal.App(modal_config.app_name)
data_volume = modal.Volume.from_name(modal_config.data_volume, create_if_missing=False)
checkpoint_volume = modal.Volume.from_name(modal_config.checkpoint_volume, create_if_missing=True)

# Build image from config
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*modal_config.pip_packages)
    .apt_install(*modal_config.apt_packages)
)


@app.function(
    image=image,
    gpu=modal_config.gpu_type,
    timeout=modal_config.timeout,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name(modal_config.wandb_secret)],
)
def train_yolo11(epochs: int = None, batch_size: int = None):
    """Train YOLO11 with full W&B training metrics logging."""
    from ultralytics import YOLO, settings
    
    # Override config with passed args
    config = YOLOConfig(
        epochs=epochs or yolo_config.epochs,
        batch_size=batch_size or yolo_config.batch_size,
    )
    
    # Set W&B env vars
    os.environ["WANDB_PROJECT"] = "usd-detection"
    os.environ["WANDB_NAME"] = f"yolo11l-{config.epochs}ep-{modal_config.gpu_type.lower()}"
    settings.update(wandb=True)
    
    print(f"🚀 YOLO11 Training | GPU: {modal_config.gpu_type} | Epochs: {config.epochs}")
    
    # Check for checkpoint
    checkpoint_path = Path(config.project) / config.name / "weights/last.pt"
    if checkpoint_path.exists():
        print(f"📦 Resuming from: {checkpoint_path}")
        model = YOLO(str(checkpoint_path))
    else:
        print(f"📦 Loading: {config.model}")
        model = YOLO(config.model)
    
    # Train using config
    results = model.train(**config.to_train_kwargs())
    
    checkpoint_volume.commit()
    
    return {
        "epochs": config.epochs,
        "mAP50": float(results.box.map50) if hasattr(results, 'box') else 0,
        "mAP50-95": float(results.box.map) if hasattr(results, 'box') else 0,
    }


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
)
def test_yolo11(image_bytes: bytes):
    """Test YOLO11 on an image."""
    from ultralytics import YOLO
    import tempfile
    
    model = YOLO("/checkpoints/usd-yolo11/train/weights/best.pt")
    
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
    """Train YOLO11 with configurable epochs and batch size."""
    print(f"🎯 YOLO11 USD Detection | {epochs} epochs | batch {batch_size}")
    result = train_yolo11.remote(epochs=epochs, batch_size=batch_size)
    print(f"✅ Complete! {result}")
