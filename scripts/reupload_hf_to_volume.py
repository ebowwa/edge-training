"""
Download USD dataset from HuggingFace and upload to Modal volume.

Uses pipeline/dataset/config.py for dataset configuration.
Uses pipeline/dataset/converter.py for COCO→YOLO conversion.
"""

import modal
import shutil
from pathlib import Path

# Import composable configs and utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.dataset.config import DatasetConfig

# Load config
config = DatasetConfig()

# Modal setup
app = modal.App("reupload-hf-dataset")
volume = modal.Volume.from_name(config.volume_name, create_if_missing=True)

image = (
    modal.Image.debian_slim("3.11")
    .pip_install("datasets", "pillow", "tqdm", "pycocotools")
)


@app.function(
    image=image,
    timeout=7200,
    volumes={"/data": volume},
)
def reupload_dataset(dry_run: bool = False):
    """Download from HuggingFace and upload to Modal volume."""
    from datasets import load_dataset
    from tqdm import tqdm
    from PIL import Image
    from io import BytesIO
    
    # Import converter functions
    from pipeline.dataset.converter import (
        extract_image_data,
        extract_annotations,
        write_yolo_label,
    )
    
    print(f"📦 Downloading from: {config.repo_id}")
    print(f"🔍 Mode: {'DRY RUN' if dry_run else 'LIVE'}\n")
    
    stats = {"images": 0, "annotations": 0}
    
    for hf_split, yolo_dir in config.splits:
        print(f"\n{'='*60}")
        print(f"📥 {hf_split} → {yolo_dir}/")
        print(f"{'='*60}")
        
        try:
            ds = load_dataset(config.repo_id, split=hf_split)
            print(f"✓ Loaded {len(ds)} items")
        except Exception as e:
            print(f"⚠️ Failed: {e}")
            continue
        
        images_dir = Path(f"/data/{yolo_dir}/images")
        labels_dir = Path(f"/data/{yolo_dir}/labels")
        
        if not dry_run:
            for d in [images_dir, labels_dir]:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
        
        for idx, item in enumerate(tqdm(ds, desc=yolo_dir)):
            image_id = item.get("image_id", item.get("id", f"img_{idx}"))
            
            # Extract image
            img_data, img_width, img_height = extract_image_data(item)
            
            if not dry_run:
                with open(images_dir / f"{image_id}.jpg", "wb") as f:
                    f.write(img_data)
            
            # Extract and write annotations
            bboxes, categories = extract_annotations(item)
            
            if not dry_run:
                write_yolo_label(
                    labels_dir / f"{image_id}.txt",
                    bboxes, categories, img_width, img_height
                )
            
            stats["images"] += 1
            stats["annotations"] += len(bboxes)
        
        # Clean caches
        for cache in [f"/data/{yolo_dir}/{yolo_dir}.cache", 
                      f"/data/{yolo_dir}/labels.cache"]:
            if Path(cache).exists() and not dry_run:
                Path(cache).unlink()
    
    # Generate data.yaml
    if not dry_run:
        yaml_path = Path("/data/data.yaml")
        yaml_path.write_text(config.generate_yaml())
        print(f"✓ Created data.yaml")
        volume.commit()
    
    print(f"\n📊 {stats['images']} images, {stats['annotations']} annotations")
    return stats


@app.local_entrypoint()
def main():
    import os
    dry_run = os.getenv("DRY_RUN", "0") in ("1", "true", "yes")
    print("🔄 Re-downloading USD dataset from HuggingFace...")
    result = reupload_dataset.remote(dry_run=dry_run)
    print(f"✅ Complete! {result}")
