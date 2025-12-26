# Disk & Memory Management

## Current Issues

### 1. Image Size
Using `Image.add_local_dir()` embeds the **entire dataset (5.5GB)** into the image:

```python
image = image.add_local_dir(dataset_path, "/data")  # BAD for large datasets
```

**Problems:**
- Slow image builds (minutes)
- Large image push (5.5GB upload)
- Modal has ~10GB image size limits
- Re-builds required for any data change

### 2. GPU Memory
Different GPUs have different memory:
- **T4**: 16GB VRAM
- **A10G**: 24GB VRAM  
- **A100**: 40GB VRAM

DINOv2 + batch processing needs ~8-12GB, so A10G is good.

## Better Solution: Modal Volumes

### Use Volumes for Large Datasets

```python
from integrations.modal.volumes import ModalVolumeManager

# 1. Upload once
volume_mgr = ModalVolumeManager("usd-dataset")
volume_mgr.upload_dataset(Path("./datasets/usd_detection"))

# 2. Mount in functions
@app.function(
    volumes={"/data": volume_mgr.volume}  # Mount, don't embed
)
def process():
    # Data is at /data
    pass
```

**Benefits:**
- Upload dataset **once**, reuse forever
- Small, fast image builds
- Update data without rebuilding image
- Share volumes across apps

### Memory-Optimized Config

```python
from integrations.modal.volumes import get_gpu_memory_config

config = get_gpu_memory_config("A10G")
# {"memory_gb": 24, "batch_size": 64}

@app.function(gpu="A10G")
def process(batch_size: int = config["batch_size"]):
    # Auto-sized for GPU memory
    pass
```

## Disk Space

Modal containers have:
- **Ephemeral disk**: ~50GB (cleared after run)
- **Volumes**: Persistent, unlimited (pay per GB)

For dedup:
- Dataset: 5.5GB in volume
- Embeddings: ~350MB in memory
- Results: ~1MB saved locally

**Total**: ~6GB needed, well within limits.

## Updated modal_dedup.py

To use volumes instead of embedding:

```python
from integrations.modal.volumes import ModalVolumeManager

app = modal.App("usd-dataset-dedup")
volume_mgr = ModalVolumeManager()

# Lightweight image (no dataset)
image = modal.Image.debian_slim("3.11").pip_install(
    "torch", "transformers", "faiss-cpu"
)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume_mgr.volume}  # Mount volume
)
def run_dedup(threshold: float = 0.99):
    # Dataset at /data
    pass
```

## Upload Dataset to Volume

```bash
# One-time upload
uv run modal volume put usd-dataset datasets/usd_detection/ /

# Verify
uv run modal volume ls usd-dataset
```

Then all runs reuse this data without re-uploading.

## Memory Monitoring

Add to functions:

```python
import psutil

def check_memory():
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.percent}% used")
    print(f"Available: {mem.available / (1024**3):.1f}GB")
```

Modal logs show this automatically in dashboard.
