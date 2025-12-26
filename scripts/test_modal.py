"""
Test Modal Integration

Quick tests to verify Modal integration works before running full dedup.
"""

import modal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.modal import ModalConfig

# Test 1: Simple hello world
app = modal.App("modal-test")

# Mount integrations module
integrations_path = Path(__file__).parent.parent / "integrations"

config_dict = {
    "gpu_type": "T4",
    "python_version": "3.11",
    "pip_packages": ["numpy", "torch"]
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch")
    .add_local_dir(integrations_path, remote_path="/root/integrations")
)


@app.function(image=image, gpu="T4", timeout=300)
def test_gpu():
    """Test GPU is available."""
    import torch
    
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        return {
            "success": True,
            "gpu": gpu_name,
            "cuda_version": torch.version.cuda
        }
    else:
        return {
            "success": False,
            "error": "No GPU detected"
        }


@app.function(image=image, timeout=60)
def test_retry():
    """Test retry logic."""
    import sys
    sys.path.insert(0, "/root")
    from integrations.modal.utils import with_retry
    import random
    
    attempt_count = [0]
    
    @with_retry(max_retries=3, delay=0.5)
    def flaky_function():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ValueError(f"Simulated failure (attempt {attempt_count[0]})")
        return "Success after retries!"
    
    try:
        result = flaky_function()
        return {
            "success": True,
            "attempts": attempt_count[0],
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.function(image=image, timeout=120)
def test_small_dataset():
    """Test with tiny embedded dataset."""
    from pathlib import Path
    
    # Check if data is mounted
    data_dir = Path("/data")
    if not data_dir.exists():
        return {"success": False, "error": "/data not found"}
    
    # Count files
    file_count = sum(1 for _ in data_dir.rglob("*.jpg"))
    
    return {
        "success": True,
        "files_found": file_count,
        "splits": [d.name for d in data_dir.iterdir() if d.is_dir()]
    }


@app.local_entrypoint()
def main(test: str = "all"):
    """
    Run Modal integration tests.
    
    Args:
        test: Which test to run (gpu, retry, dataset, or all)
    """
    print("🧪 Testing Modal Integration\n")
    
    if test in ["gpu", "all"]:
        print("1️⃣ Testing GPU availability...")
        result = test_gpu.remote()
        if result["success"]:
            print(f"   ✓ GPU detected: {result['gpu']}")
            print(f"   ✓ CUDA version: {result['cuda_version']}")
        else:
            print(f"   ✗ {result['error']}")
        print()
    
    if test in ["retry", "all"]:
        print("2️⃣ Testing retry logic...")
        result = test_retry.remote()
        if result["success"]:
            print(f"   ✓ Retry worked after {result['attempts']} attempts")
            print(f"   ✓ Result: {result['result']}")
        else:
            print(f"   ✗ {result['error']}")
        print()
    
    if test in ["dataset", "all"]:
        print("3️⃣ Testing dataset access...")
        print("   Note: This requires dataset to be embedded in image")
        print("   Skipping for now - will test in full dedup run")
        print()
    
    print("✅ Tests complete!")
