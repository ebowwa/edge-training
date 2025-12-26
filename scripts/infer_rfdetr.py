"""
RF-DETR Inference on New Images

Run USD detection on new images using trained model.
Usage: uv run python scripts/infer_rfdetr.py path/to/image.jpg
"""

import sys
from pathlib import Path
import modal

# Modal setup
app = modal.App("usd-inference")
volume_checkpoints = modal.Volume.from_name("usd-checkpoints")

image = (
    modal.Image.debian_slim("3.11")
    .pip_install("ultralytics", "torch", "torchvision", "opencv-python-headless", "pillow")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/checkpoints": volume_checkpoints}
)
def detect_usd(image_bytes: bytes):
    """
    Run USD detection on an image.
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        dict with detections (boxes, classes, confidences)
    """
    from ultralytics import RTDETR
    from PIL import Image
    import io
    import tempfile
    
    print("🔍 Running USD detection...")
    
    # Load model
    model = RTDETR("/checkpoints/usd-rfdetr/train/weights/best.pt")
    
    # Save bytes to temp file for inference
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        temp_path = f.name
    
    # Run inference
    results = model(temp_path, conf=0.25, iou=0.45)
    
    # Extract detections
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            det = {
                "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])]
            }
            detections.append(det)
    
    print(f"✓ Found {len(detections)} USD bills")
    for det in detections:
        print(f"  {det['class_name']}: {det['confidence']:.2f}")
    
    return {
        "detections": detections,
        "count": len(detections)
    }


@app.function(
    image=image,
    volumes={"/checkpoints": volume_checkpoints}
)
def download_model():
    """Download trained model from Modal volume to local."""
    import shutil
    from pathlib import Path
    
    checkpoint_path = Path("/checkpoints/usd-rfdetr/train/weights/best.pt")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    # Return file bytes
    with open(checkpoint_path, 'rb') as f:
        return f.read()


@app.local_entrypoint()
def main(image_path: str):
    """
    Run inference on an image.
    
    Args:
        image_path: Path to image file
    """
    
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        return
    
    print(f"📸 Processing: {img_path.name}")
    
    # Read image bytes
    with open(img_path, 'rb') as f:
        image_bytes = f.read()
    
    # Run detection on Modal
    result = detect_usd.remote(image_bytes)
    
    print(f"\n📊 Results:")
    print(f"Total detections: {result['count']}")
    
    if result['detections']:
        print("\nDetections:")
        for i, det in enumerate(result['detections'], 1):
            print(f"{i}. {det['class_name']} ({det['confidence']:.1%} confidence)")
            print(f"   Bbox: {det['bbox']}")


# Local inference (if model downloaded)
def infer_local(image_path: str, model_path: str = "best.pt"):
    """Run inference locally without Modal."""
    from ultralytics import RTDETR
    from pathlib import Path
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Download with: modal run scripts/infer_rfdetr.py --download")
        return
    
    print(f"📸 Processing: {image_path}")
    
    # Load model
    model = RTDETR(model_path)
    
    # Run inference
    results = model(image_path, conf=0.25, save=True)
    
    # Print results
    for result in results:
        boxes = result.boxes
        print(f"\n✓ Found {len(boxes)} USD bills:")
        for box in boxes:
            class_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"  {class_name}: {conf:.1%}")
    
    print(f"\nResults saved to: {results[0].save_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--download":
            print("📥 Downloading model from Modal...")
            model_bytes = download_model.remote()
            with open("best.pt", "wb") as f:
                f.write(model_bytes)
            print(f"✓ Model saved to: best.pt ({len(model_bytes)/1024/1024:.1f}MB)")
        elif sys.argv[1] == "--local":
            if len(sys.argv) < 3:
                print("Usage: python scripts/infer_rfdetr.py --local path/to/image.jpg")
            else:
                infer_local(sys.argv[2])
        else:
            main(sys.argv[1])
    else:
        print("Usage:")
        print("  Modal: uv run modal run scripts/infer_rfdetr.py /path/to/image.jpg")
        print("  Download model: python scripts/infer_rfdetr.py --download")
        print("  Local: python scripts/infer_rfdetr.py --local /path/to/image.jpg")
