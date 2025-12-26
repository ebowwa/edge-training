"""Quick validation script for trained RF-DETR model."""
import modal

app = modal.App("test-rfdetr")
volume_data = modal.Volume.from_name("usd-dataset-test")
volume_checkpoints = modal.Volume.from_name("usd-checkpoints")

image = modal.Image.debian_slim("3.11").pip_install("ultralytics", "torch", "torchvision", "opencv-python-headless", "pyyaml").apt_install("libgl1-mesa-glx", "libglib2.0-0")

@app.function(image=image, gpu="T4", timeout=1800, volumes={"/data": volume_data, "/checkpoints": volume_checkpoints})
def test_model():
    from ultralytics import RTDETR
    
    print("🔍 Testing RF-DETR model...")
    model = RTDETR("/checkpoints/usd-rfdetr/train/weights/best.pt")
    
    # Test on test set
    metrics = model.val(data="/data/data.yaml", split="test", batch=8)
    
    print(f"\n📊 Test Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return {"mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)}

@app.local_entrypoint()
def main():
    result = test_model.remote()
    print(f"\nFinal metrics: {result}")
