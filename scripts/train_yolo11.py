"""
YOLO11 Training on Modal - Proper W&B Integration

Uses ONLY Ultralytics native W&B integration (no manual wandb.init)
This logs ALL training metrics: box_loss, cls_loss, dfl_loss, mAP, precision, recall
"""

import modal
import os

# App and Volumes
app = modal.App("usd-yolo11-training")
# Fresh volume with HuggingFace dataset - old one was deleted
data_volume = modal.Volume.from_name("usd-dataset-hf", create_if_missing=False)
checkpoint_volume = modal.Volume.from_name("usd-checkpoints", create_if_missing=True)

# Configuration
GPU_TYPE = "H100"
BATCH_SIZE = 32
EPOCHS = 50
IMGSZ = 640

# Build image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ultralytics",
        "torch",
        "torchvision", 
        "opencv-python-headless",
        "pyyaml",
        "wandb",
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=7200,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_yolo11(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """Train YOLO11 with full W&B training metrics logging."""
    import os
    from ultralytics import YOLO, settings
    from pathlib import Path
    
    # Set W&B project/name via env vars BEFORE enabling integration
    # This lets Ultralytics create the run and log ALL training metrics:
    # box_loss, cls_loss, dfl_loss, mAP50, mAP50-95, precision, recall,
    # confusion matrix, PR curves, sample predictions
    os.environ["WANDB_PROJECT"] = "usd-detection"
    os.environ["WANDB_NAME"] = f"yolo11l-{epochs}ep-h100"
    
    # Enable Ultralytics native W&B - this is the ONLY W&B setup needed
    # It automatically logs: box_loss, cls_loss, dfl_loss, mAP50, mAP50-95, 
    # precision, recall, confusion matrix, PR curves, sample predictions
    settings.update(wandb=True)
    
    print(f"🚀 Starting YOLO11 Training")
    print(f"GPU: {GPU_TYPE}, Batch: {batch_size}, Epochs: {epochs}")
    print(f"W&B Project: usd-detection")
    
    # Check for checkpoint
    checkpoint_path = "/checkpoints/usd-yolo11/train/weights/last.pt"
    if Path(checkpoint_path).exists():
        print(f"📦 Resuming from: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print(f"📦 Loading base model: yolo11l.pt")
        model = YOLO("yolo11l.pt")
    
    # Train - Ultralytics handles ALL W&B logging automatically
    results = model.train(
        data="/data/data.yaml",
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch_size,
        device=0,
        project="/checkpoints/usd-yolo11",
        name="train",
        exist_ok=True,
        resume=False,
        amp=True,
        cache=False,
        workers=0,
        verbose=True,
        plots=True,
        save=True,
        save_period=10,
    )
    
    # Commit to volume
    checkpoint_volume.commit()
    
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
def main(epochs: int = 50):
    """Train YOLO11 with full W&B metrics."""
    print(f"🎯 YOLO11 USD Detection Training ({epochs} epochs) on H100")
    print(f"📊 W&B: All training metrics will be logged")
    
    result = train_yolo11.remote(epochs=epochs)
    print(f"\n✅ Training Complete!")
    print(f"Results: {result}")
