"""
Remove duplicate images from dataset based on dedup results.

Deletes duplicate images and updates COCO annotations.
"""

import json
from pathlib import Path
import shutil
from typing import Set

def remove_duplicates(results_file: str = "dedup_results.json", dry_run: bool = False):
    """
    Remove duplicate images and update annotations.
    
    Args:
        results_file: Path to dedup results JSON
        dry_run: If True, only show what would be deleted
    """
    # Load results
    with open(results_file) as f:
        results = json.load(f)
    
    to_remove = results['remove_list']
    print(f"📋 Found {len(to_remove)} duplicates to remove")
    print(f"💾 Will keep {results['to_keep']} unique images\n")
    
    if dry_run:
        print("🔍 DRY RUN - showing what would be deleted:\n")
    
    # Convert Modal paths to local paths
    dataset_dir = Path("datasets/usd_detection")
    removed_count = 0
    removed_by_split = {"test": 0, "train": 0, "valid": 0}
    removed_files = {"test": set(), "train": set(), "valid": set()}
    
    for modal_path in to_remove:
        # Modal path: /data/train/image.jpg or /data/test/image.jpg
        # Convert to: datasets/usd_detection/train/image.jpg
        path_obj = Path(modal_path)
        
        # Remove /data prefix, get split and filename
        # path_obj.parts = ('/', 'data', 'train', 'image.jpg')
        if len(path_obj.parts) < 3:
            print(f"  ⚠️  Invalid path: {modal_path}")
            continue
        
        split = path_obj.parts[2]  # train/valid/test (skip / and data)
        filename = path_obj.parts[3]  # image.jpg
        
        local_path = dataset_dir / split / filename
        
        if local_path.exists():
            if dry_run:
                print(f"  Would delete: {local_path}")
            else:
                local_path.unlink()
                print(f"  ✓ Deleted: {local_path}")
            
            removed_count += 1
            removed_by_split[split] += 1
            removed_files[split].add(filename)
        else:
            print(f"  ⚠️  Not found: {local_path}")
    
    print(f"\n{'Would remove' if dry_run else 'Removed'} {removed_count} files:")
    for split, count in removed_by_split.items():
        print(f"  {split}: {count}")
    
    if not dry_run:
        # Update COCO annotations
        print("\n📝 Updating COCO annotations...")
        for split in ['test', 'train', 'valid']:
            if removed_files[split]:
                update_coco_annotations(dataset_dir / split, removed_files[split])
        
        print(f"\n✅ Dataset cleaned!")
        print(f"Final size: {results['to_keep']} images")
    else:
        print("\n⚠️  This was a dry run. Run with dry_run=False to actually delete.")
    
    return removed_count


def update_coco_annotations(split_dir: Path, removed_files: Set[str]):
    """Update COCO JSON to remove references to deleted images."""
    coco_file = split_dir / "_annotations.coco.json"
    
    if not coco_file.exists():
        print(f"  ⚠️  No COCO file found: {coco_file}")
        return
    
    with open(coco_file) as f:
        coco = json.load(f)
    
    # Remove images
    original_image_count = len(coco['images'])
    coco['images'] = [
        img for img in coco['images']
        if img['file_name'] not in removed_files
    ]
    
    # Get remaining image IDs
    valid_image_ids = {img['id'] for img in coco['images']}
    
    # Remove annotations for deleted images
    original_ann_count = len(coco['annotations'])
    coco['annotations'] = [
        ann for ann in coco['annotations']
        if ann['image_id'] in valid_image_ids
    ]
    
    # Save updated COCO
    with open(coco_file, 'w') as f:
        json.dump(coco, f, indent=2)
    
    removed_images = original_image_count - len(coco['images'])
    removed_anns = original_ann_count - len(coco['annotations'])
    
    print(f"  ✓ {split_dir.name}: Removed {removed_images} images, {removed_anns} annotations")


if __name__ == "__main__":
    import sys
    
    # First run dry run to preview
    if "--dry-run" in sys.argv:
        remove_duplicates(dry_run=True)
    else:
        # Confirm before deleting
        print("⚠️  WARNING: This will permanently delete 425 duplicate images!")
        response = input("Are you sure you want to continue? [y/N]: ")
        
        if response.lower() == 'y':
            remove_duplicates(dry_run=False)
        else:
            print("Cancelled.")
