"""
Chunked RF-DETR Training - Run in 10-epoch increments to avoid Modal timeouts
"""

import modal
from pathlib import Path

app = modal.App("usd-rfdetr-chunked")
volume = modal.Volume.from_name("usd-dataset-test", create_if_missing=False)

GPU_TYPE = "A100"
BATCH_SIZE = 16
IMGSZ = 640
CHUNK_SIZE = 10  # Train 10 epochs per run

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("ultralytics", "torch", "torchvision", "opencv-python-headless", "pyyaml", "wandb")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,  # 30 min max per chunk
    volumes={"/data": volume, "/checkpoints": modal.Volume.from_name("usd-checkpoints", create_if_missing=True)},
)
def train_chunk(target_epochs: int = 20, current_epoch: int = 0):
    """Train for CHUNK_SIZE epochs from current checkpoint."""
    import os
    from ultralytics import RTDETR
    
    print(f"🚀 Training chunk: epochs {current_epoch} → {target_epochs}")
    print(f"GPU: {GPU_TYPE}, Batch: {BATCH_SIZE}")
    
    # Load from checkpoint if exists
    checkpoint_path = "/checkpoints/usd-rfdetr/train/weights/last.pt"
    from pathlib import Path
    if Path(checkpoint_path).exists():
        print(f"📦 Resuming from: {checkpoint_path}")
        model = RTDETR(checkpoint_path)
    else:
        print(f"📦 Loading base model: rtdetr-l.pt")
        model = RTDETR("rtdetr-l.pt")
    
    # Train for this chunk
    results = model.train(
        data="/data/data.yaml",
        epochs=target_epochs,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=0,
        project="/checkpoints/usd-rfdetr",
        name="train",
        exist_ok=True,
        resume=False,
        amp=True,
        cache=False,
        workers=4,
        verbose=True,
        plots=True,
        save=True,
        save_period=5,  # Save every 5 epochs
    )
    
    # Commit to volume
    volume.commit()
    
    metrics = {
        "target_epochs": target_epochs,
        "mAP50": float(results.box.map50) if hasattr(results, 'box') else 0,
        "mAP50-95": float(results.box.map) if hasattr(results, 'box') else 0,
    }
    
    print(f"\n✅ Chunk complete: {current_epoch}→{target_epochs} epochs")
    print(f"Metrics: {metrics}")
    
    return metrics


@app.local_entrypoint()
def main(total_epochs: int = 100):
    """Train in chunks to avoid timeouts."""
    print(f"🎯 Chunked Training: {total_epochs} epochs in {CHUNK_SIZE}-epoch increments\n")
    
    current = 0
    results = []
    
    while current < total_epochs:
        target = min(current + CHUNK_SIZE, total_epochs)
        print(f"\n{'='*60}")
        print(f"CHUNK: Epochs {current} → {target}")
        print(f"{'='*60}")
        
        try:
            result = train_chunk.remote(target_epochs=target, current_epoch=current)
            results.append(result)
            current = target
            print(f"✓ Progress: {current}/{total_epochs} epochs")
        except Exception as e:
            print(f"⚠️  Chunk failed: {e}")
            print(f"Completed {current}/{total_epochs} epochs before failure")
            break
    
    print(f"\n{'='*60}")
    print(f"FINAL: Completed {current}/{total_epochs} epochs")
    print(f"{'='*60}")
    print(f"Results: {results}")
