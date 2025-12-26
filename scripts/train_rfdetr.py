"""
RT-DETR Training on Modal with Doppler Secrets

Autonomous training pipeline for USD detection using cleaned dataset.
Run with: uv run modal run scripts/train_rfdetr.py
"""

import modal
from pathlib import Path

app = modal.App("usd-rfdetr-training")
volume = modal.Volume.from_name("usd-dataset-test", create_if_missing=False)

# Training configuration
GPU_TYPE = "A100"  # Fast training
BATCH_SIZE = 16    # A100 optimized
EPOCHS = 100       # Full training
IMGSZ = 640        # Standard detection size

# Training image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ultralytics",  # RT-DETR
        "torch",
        "torchvision",
        "opencv-python-headless",
        "pyyaml",
        "wandb",  # Weights & Biases
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV dependencies
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=10800,  # 3 hours max
    volumes={"/data": volume, "/checkpoints": modal.Volume.from_name("usd-checkpoints", create_if_missing=True)},
)
def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    imgsz: int = IMGSZ,
    model: str = "rtdetr-l.pt",
    resume: bool = False
):
    """
    Train RT-DETR model on cleaned USD dataset.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Input image size
        model: Base model (rtdetr-l.pt, rtdetr-x.pt)
        resume: Resume from last checkpoint
    """
    import os
    from ultralytics import RTDETR
    
    print(f"🚀 Starting RT-DETR Training")
    print(f"GPU: {GPU_TYPE}, Batch: {batch_size}, Epochs: {epochs}")
    
    # Initialize model - resume from checkpoint if exists
    checkpoint_path = "/checkpoints/usd-rfdetr/train/weights/last.pt"
    from pathlib import Path
    if Path(checkpoint_path).exists():
        print(f"\n📦 Resuming from checkpoint: {checkpoint_path}")
        model = RTDETR(checkpoint_path)
    else:
        print(f"\n📦 Loading base model: {model}...")
        model = RTDETR(model)
    
    # Configure training
    results = model.train(
        data="/data/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0,  # GPU
        project="/checkpoints/usd-rfdetr",
        name="train",
        exist_ok=True,
        resume=False,  # We load checkpoint directly above
        # Optimizations
        amp=True,  # Automatic Mixed Precision
        cache=False,  # Don't cache (volume access)
        workers=4,
        # Logging
        verbose=True,
        plots=True,
        # Save best model
        save=True,
        save_period=10,  # Save every 10 epochs
    )
    
    # Get metrics
    metrics = {
        "mAP50": float(results.box.map50) if hasattr(results, 'box') else 0,
        "mAP50-95": float(results.box.map) if hasattr(results, 'box') else 0,
        "best_epoch": results.best_epoch if hasattr(results, 'best_epoch') else epochs,
    }
    
    print(f"\n✅ Training Complete!")
    print(f"mAP50: {metrics['mAP50']:.4f}")
    print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"Best epoch: {metrics['best_epoch']}")
    
    # Commit checkpoints to volume
    volume.commit()
    
    return metrics


@app.function(
    image=image,
    gpu="T4",  # Lighter GPU for validation
    timeout=1800,
    volumes={"/data": volume, "/checkpoints": modal.Volume.from_name("usd-checkpoints")},
)
def validate_model(checkpoint_path: str = "/checkpoints/usd-rfdetr/train/weights/best.pt"):
    """Validate trained model on test set."""
    from ultralytics import RTDETR
    
    print(f"🔍 Validating model: {checkpoint_path}")
    
    model = RTDETR(checkpoint_path)
    
    # Validate on test set
    metrics = model.val(
        data="/data/data.yaml",
        split="test",
        batch=8,
        device=0
    )
    
    results = {
        "test_mAP50": float(metrics.box.map50),
        "test_mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    
    print(f"\n📊 Test Results:")
    print(f"mAP50: {results['test_mAP50']:.4f}")
    print(f"mAP50-95: {results['test_mAP50-95']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    return results


@app.function(
    image=image,
    volumes={"/checkpoints": modal.Volume.from_name("usd-checkpoints")},
)
def export_for_edge(
    checkpoint_path: str = "/checkpoints/usd-rfdetr/train/weights/best.pt",
    format: str = "onnx",
    quantize: bool = True
):
    """Export model for edge deployment."""
    from ultralytics import RTDETR
    
    print(f"📤 Exporting {checkpoint_path} to {format}")
    
    model = RTDETR(checkpoint_path)
    
    # Export
    export_path = model.export(
        format=format,
        int8=quantize,
        imgsz=640,
        simplify=True,  # Simplify ONNX
    )
    
    print(f"✓ Exported to: {export_path}")
    
    # Commit export to volume
    volume.commit()
    
    return {"export_path": str(export_path), "format": format, "quantized": quantize}


@app.local_entrypoint()
def main(
    train: bool = True,
    validate: bool = True,
    export: bool = False,
    epochs: int = EPOCHS,
    resume: bool = False
):
    """
    Main training pipeline.
    
    Args:
        train: Run training
        validate: Run validation after training
        export: Export model after validation
        epochs: Number of epochs
        resume: Resume from checkpoint
    """
    print("🎯 USD Detection RT-DETR Training Pipeline\n")
    
    if train:
        print("=" * 60)
        print("PHASE 1: TRAINING")
        print("=" * 60)
        train_metrics = train_model.remote(epochs=epochs, resume=resume)
        print(f"\nTraining metrics: {train_metrics}")
    
    if validate:
        print("\n" + "=" * 60)
        print("PHASE 2: VALIDATION")
        print("=" * 60)
        val_metrics = validate_model.remote()
        print(f"\nValidation metrics: {val_metrics}")
    
    if export:
        print("\n" + "=" * 60)
        print("PHASE 3: EXPORT FOR EDGE")
        print("=" * 60)
        export_result = export_for_edge.remote(format="onnx", quantize=True)
        print(f"\nExport result: {export_result}")
    
    print("\n✅ Pipeline Complete!")
