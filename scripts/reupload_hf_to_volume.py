"""
Download USD Side Detection dataset from HuggingFace and upload to Modal volume.

This replaces the volume contents with the cleaned HuggingFace dataset.
"""

import modal
import os
import shutil
import json
from pathlib import Path

app = modal.App("reupload-hf-dataset")
# Fresh volume - old one was deleted
volume = modal.Volume.from_name("usd-dataset-hf", create_if_missing=True)

# 24 classes: 12 regular + 12 counterfeit (front/back for each denomination)
USD_CLASSES = {
    0: "100USD-Back",
    1: "100USD-Front",
    2: "10USD-Back",
    3: "10USD-Front",
    4: "1USD-Back",
    5: "1USD-Front",
    6: "20USD-Back",
    7: "20USD-Front",
    8: "50USD-Back",
    9: "50USD-Front",
    10: "5USD-Back",
    11: "5USD-Front",
    12: "Counterfeit 100 USD Back",
    13: "Counterfeit 100 USD Front",
    14: "Counterfeit 10USD Back",
    15: "Counterfeit 10USD Front",
    16: "Counterfeit 1USD Back",
    17: "Counterfeit 1USD Front",
    18: "Counterfeit 20USD Back",
    19: "Counterfeit 20USD Front",
    20: "Counterfeit 50USD Back",
    21: "Counterfeit 50USD Front",
    22: "Counterfeit 5USD Back",
    23: "Counterfeit 5USD Front",
}

# Image with datasets library and PIL for image handling
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
    """
    Download from HuggingFace and replace volume contents.

    Args:
        dry_run: If True, download and convert but don't upload to volume
    """
    from datasets import load_dataset
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    from io import BytesIO

    repo_id = "ebowwa/usd-side-coco-annotations"
    # HuggingFace split name -> YOLO directory name
    splits = [("train", "train"), ("validation", "valid"), ("test", "test")]

    mode_str = "DRY RUN" if dry_run else "LIVE"
    print(f"🔍 Mode: {mode_str}")
    print(f"📦 Downloading from: {repo_id}\n")

    stats = {"images": 0, "annotations": 0}

    for hf_split, yolo_dir in splits:
        print(f"\n{'='*60}")
        print(f"📥 Processing {hf_split} split -> {yolo_dir}/...")
        print(f"{'='*60}")

        # Load from HuggingFace
        try:
            ds = load_dataset(repo_id, split=hf_split)
            print(f"✓ Loaded {len(ds)} items from HuggingFace")
        except Exception as e:
            print(f"⚠️  Failed to load {hf_split}: {e}")
            continue

        # Create directories (use YOLO naming convention)
        images_dir = Path(f"/data/{yolo_dir}/images")
        labels_dir = Path(f"/data/{yolo_dir}/labels")

        if not dry_run:
            # Clear existing data for this split
            if images_dir.exists():
                shutil.rmtree(images_dir)
            if labels_dir.exists():
                shutil.rmtree(labels_dir)

            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

        # Process each item
        for idx, item in enumerate(tqdm(ds, desc=f"Converting {yolo_dir}")):
            # Get image data - HuggingFace datasets library returns PIL Image
            img = item["image"]

            # Generate image ID from index or use provided ID
            image_id = item.get("image_id", item.get("id", f"img_{idx}"))

            if isinstance(img, Image.Image):
                # PIL Image - convert to bytes
                img_bytes = BytesIO()
                img.save(img_bytes, format="JPEG")
                img_data = img_bytes.getvalue()
                img_width, img_height = img.size
            elif isinstance(img, bytes):
                img_data = img
                img_width = item.get("width", 640)
                img_height = item.get("height", 640)
            else:
                # numpy array or other
                pil_img = Image.fromarray(img)
                img_bytes = BytesIO()
                pil_img.save(img_bytes, format="JPEG")
                img_data = img_bytes.getvalue()
                img_width, img_height = pil_img.size

            # Save image
            img_filename = f"{image_id}.jpg"
            img_path = images_dir / img_filename

            if not dry_run:
                with open(img_path, "wb") as f:
                    f.write(img_data)

            # Get objects/bboxes - handle different COCO formats
            # Try multiple possible field names
            if "objects" in item:
                objects = item["objects"]
                if isinstance(objects, dict):
                    bboxes = objects.get("bbox", [])
                    categories = objects.get("category", [])
                else:
                    bboxes = [obj.get("bbox", [0,0,0,0]) for obj in objects]
                    categories = [obj.get("category", 0) for obj in objects]
            elif "bboxes" in item:
                bboxes = item["bboxes"]
                categories = item.get("category_id", item.get("categories", []))
            else:
                # No annotations
                bboxes = []
                categories = []

            # Create YOLO label file
            label_path = labels_dir / f"{image_id}.txt"

            if not dry_run:
                with open(label_path, "w") as f:
                    for bbox, category_id in zip(bboxes, categories):
                        # COCO format: [x, y, width, height]
                        x, y, w, h = bbox

                        # Convert to YOLO format (normalized center + width/height)
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        w_norm = w / img_width
                        h_norm = h / img_height

                        # Clip to [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        w_norm = max(0, min(1, w_norm))
                        h_norm = max(0, min(1, h_norm))

                        f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            stats["images"] += 1
            stats["annotations"] += len(bboxes)

        print(f"✓ {yolo_dir}: {len(ds)} images processed")

        # Remove cache files
        for cache_name in [f"/data/{yolo_dir}/{yolo_dir}.cache", f"/data/{yolo_dir}/labels.cache", f"/data/{yolo_dir}/images.cache"]:
            cache_path = Path(cache_name)
            if cache_path.exists():
                if dry_run:
                    print(f"   Would remove cache: {cache_name}")
                else:
                    cache_path.unlink()
                    print(f"   ✓ Removed cache: {cache_name}")

    # Create data.yaml for YOLO training
    if not dry_run:
        yaml_content = f"""# USD Side Detection Dataset
# 24 classes: 12 regular + 12 counterfeit (front/back)
# Source: https://huggingface.co/datasets/ebowwa/usd-side-coco-annotations

path: /data
train: train/images
val: valid/images
test: test/images

nc: 24
names:
"""
        for class_id, class_name in USD_CLASSES.items():
            yaml_content += f"  {class_id}: {class_name}\n"

        yaml_path = Path("/data/data.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        print(f"✓ Created data.yaml with {len(USD_CLASSES)} classes")

        volume.commit()
        print(f"\n✅ Volume committed!")

    print(f"\n📊 Statistics:")
    print(f"   Images: {stats['images']}")
    print(f"   Annotations: {stats['annotations']}")

    return stats


@app.local_entrypoint()
def main():
    import os

    dry_run = os.getenv("DRY_RUN", "0").strip() in ("1", "true", "yes")

    print("🔄 Re-downloading USD Side Detection dataset from HuggingFace...")
    print("   This will REPLACE the current volume contents!\n")

    result = reupload_dataset.remote(dry_run=dry_run)
    print(f"\n✅ Complete! Result: {result}")
