#!/usr/bin/env python3
"""
Convert COCO annotations to HuggingFace metadata.jsonl format.

This creates metadata.jsonl files that HuggingFace's ImageFolder loader
can understand, enabling proper viewer display of object detection data.

Usage:
    python scripts/coco_to_jsonl.py --dataset-dir datasets/usd_detection
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def convert_coco_to_jsonl(coco_path: Path, output_path: Path) -> dict:
    """
    Convert a COCO annotation file to HuggingFace metadata.jsonl format.
    
    Returns stats about the conversion.
    """
    with open(coco_path) as f:
        coco = json.load(f)
    
    # Build category ID -> index mapping (0-indexed for HF)
    categories = sorted(coco['categories'], key=lambda x: x['id'])
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # Group annotations by image_id
    img_annotations = defaultdict(list)
    for ann in coco['annotations']:
        img_annotations[ann['image_id']].append(ann)
    
    # Build image_id -> filename mapping
    img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    
    # Convert to JSONL format
    rows = []
    for img in coco['images']:
        img_id = img['id']
        anns = img_annotations.get(img_id, [])
        
        if not anns:
            # Skip images without annotations
            continue
        
        row = {
            "file_name": img['file_name'],
            "objects": {
                "bbox": [ann['bbox'] for ann in anns],  # COCO format: [x, y, width, height]
                "categories": [cat_id_to_idx[ann['category_id']] for ann in anns]
            }
        }
        rows.append(row)
    
    # Write JSONL
    with open(output_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    
    # Also save category mapping for reference
    return {
        'images': len(rows),
        'annotations': sum(len(r['objects']['bbox']) for r in rows),
        'categories': len(categories),
        'category_names': [cat['name'] for cat in categories]
    }


def main():
    parser = argparse.ArgumentParser(description="Convert COCO to HuggingFace JSONL")
    parser.add_argument("--dataset-dir", type=str, default="datasets/usd_detection")
    parser.add_argument("--source-file", type=str, default="_annotations.coco.json",
                        help="Which COCO file to convert (default: _annotations.coco.json)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    print("=" * 60)
    print("COCO to HuggingFace JSONL Converter")
    print("=" * 60)
    print(f"\nDataset: {dataset_dir}")
    print(f"Source: {args.source_file}")
    print()
    
    # Show available COCO files for reference
    print("Available annotation files per split:")
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        
        json_files = list(split_dir.glob("*.json"))
        print(f"\n  {split}/")
        for f in sorted(json_files):
            size_kb = f.stat().st_size / 1024
            is_target = f.name == args.source_file
            marker = " <-- USING THIS" if is_target else ""
            print(f"    {f.name}: {size_kb:.1f}KB{marker}")
    
    print("\n" + "=" * 60)
    
    if args.dry_run:
        print("[DRY RUN] Would create metadata.jsonl files")
        return
    
    # Convert each split
    all_categories = None
    total_stats = {'images': 0, 'annotations': 0}
    
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_dir / split
        coco_path = split_dir / args.source_file
        jsonl_path = split_dir / "metadata.jsonl"
        
        if not coco_path.exists():
            print(f"\n[SKIP] {split}: {args.source_file} not found")
            continue
        
        print(f"\nConverting {split}...")
        stats = convert_coco_to_jsonl(coco_path, jsonl_path)
        
        print(f"  ✓ Created {jsonl_path}")
        print(f"    Images: {stats['images']}")
        print(f"    Annotations: {stats['annotations']}")
        print(f"    Categories: {stats['categories']}")
        
        total_stats['images'] += stats['images']
        total_stats['annotations'] += stats['annotations']
        
        if all_categories is None:
            all_categories = stats['category_names']
    
    # Save category reference file
    if all_categories:
        cat_file = dataset_dir / "categories.json"
        with open(cat_file, 'w') as f:
            json.dump({
                "num_classes": len(all_categories),
                "categories": all_categories
            }, f, indent=2)
        print(f"\n✓ Saved category mapping to {cat_file}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {total_stats['images']}")
    print(f"Total annotations: {total_stats['annotations']}")
    print(f"Total categories: {len(all_categories)}")
    print("\nCategory list:")
    for i, cat in enumerate(all_categories):
        print(f"  {i}: {cat}")
    
    print("\n✅ Done! metadata.jsonl files created for HuggingFace.")
    print("\nTo upload to HuggingFace:")
    print("  1. Copy metadata.jsonl files to HuggingFace repo")
    print("  2. Push with git")


if __name__ == "__main__":
    main()
