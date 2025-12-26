"""
Convert COCO annotations to YOLO format for RT-DETR training.

Creates label txt files alongside images with format:
class_id x_center y_center width height (normalized 0-1)
"""

import json
from pathlib import Path
from tqdm import tqdm


def coco_to_yolo(coco_json_path: Path, output_dir: Path, img_dir: Path):
    """
    Convert COCO JSON to YOLO txt format.
    
    Args:
        coco_json_path: Path to _annotations.coco.json
        output_dir: Where to save label txt files
        img_dir: Image directory (for getting image dimensions)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO JSON
    with open(coco_json_path) as f:
        coco = json.load(f)
    
    # Build image ID to filename/dimensions mapping
    images = {img['id']: img for img in coco['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"Converting {len(annotations_by_image)} images...")
    
    # Convert each image's annotations
    for img_id, anns in tqdm(annotations_by_image.items()):
        img_info = images[img_id]
        img_w = img_info['width']
        img_h = img_info['height']
        filename = Path(img_info['file_name']).stem
        
        # Create YOLO label file
        label_path = output_dir / f"{filename}.txt"
        
        with open(label_path, 'w') as f:
            for ann in anns:
                # COCO bbox: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                class_id = ann['category_id']
                
                # Write YOLO format: class x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    print(f"✓ Created {len(annotations_by_image)} label files in {output_dir}")


def convert_all_splits():
    """Convert train, valid, test splits."""
    dataset_dir = Path("datasets/usd_detection")
    
    for split in ["train", "valid", "test"]:
        print(f"\n📝 Converting {split}...")
        split_dir = dataset_dir / split
        coco_json = split_dir / "_annotations.coco.json"
        
        if not coco_json.exists():
            print(f"⚠️  {coco_json} not found, skipping")
            continue
        
        # Create labels directory
        labels_dir = split_dir / "labels"
        
        coco_to_yolo(coco_json, labels_dir, split_dir)
    
    print("\\n✅ COCO to YOLO conversion complete!")


if __name__ == "__main__":
    convert_all_splits()
