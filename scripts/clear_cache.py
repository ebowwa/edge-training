"""Delete cache files and restart training with correct structure."""
import modal

app = modal.App("clear-cache")
volume = modal.Volume.from_name("usd-dataset-test")

image = modal.Image.debian_slim("3.11")
def clear_cache():
    import os
    from pathlib import Path
    
    for split in ["train", "valid", "test"]:
        cache = Path(f"/data/{split}.cache")
        if cache.exists():
            os.remove(cache)
            print(f"✓ Deleted {split}.cache")
    
    volume.commit()
    return "Cache cleared"

@app.local_entrypoint()
def main():
    result = clear_cache.remote()
    print(result)
