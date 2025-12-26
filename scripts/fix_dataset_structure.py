"""
Fix YOLO dataset structure on Modal volume.

Problem: Images are in /train/*.jpg but YOLO expects /train/images/*.jpg
Solution: Move images to images/ subfolder so labels in labels/ are found.
"""

import modal
from pathlib import Path

app = modal.App("fix-dataset-structure")
volume = modal.Volume.from_name("usd-dataset-test", create_if_missing=False)

image = modal.Image.debian_slim("3.11").pip_install("tqdm")


@app.function(
    image=image,
    timeout=3600,
    volumes={"/data": volume},
)
def fix_structure():
    """Move images to images/ subfolder for each split."""
    import os
    import shutil
    from pathlib import Path
    from tqdm import tqdm
    
    splits = ["train", "valid", "test"]
    
    for split in splits:
        split_dir = Path(f"/data/{split}")
        images_dir = split_dir / "images"
        
        if not split_dir.exists():
            print(f"⚠️  {split} not found, skipping")
            continue
        
        # Create images subdirectory
        images_dir.mkdir(exist_ok=True)
        
        # Find all jpg files in split root
        jpg_files = list(split_dir.glob("*.jpg"))
        
        if not jpg_files:
            print(f"✓ {split}: No images to move (already organized)")
            continue
        
        print(f"📦 {split}: Moving {len(jpg_files)} images to images/")
        
        for img in tqdm(jpg_files, desc=split):
            dest = images_dir / img.name
            shutil.move(str(img), str(dest))
        
        print(f"✓ {split}: Moved {len(jpg_files)} images")
        
        # Verify labels exist
        labels_dir = split_dir / "labels"
        if labels_dir.exists():
            label_count = len(list(labels_dir.glob("*.txt")))
            print(f"  Labels: {label_count} files")
        else:
            print(f"  ⚠️  No labels/ directory!")
    
    # Commit changes
    volume.commit()
    
    print("\n✅ Dataset structure fixed!")
    print("New structure:")
    print("  /train/images/*.jpg + /train/labels/*.txt")
    print("  /valid/images/*.jpg + /valid/labels/*.txt")  
    print("  /test/images/*.jpg + /test/labels/*.txt")
    
    return {"status": "fixed"}


@app.local_entrypoint()
def main():
    print("🔧 Fixing YOLO dataset structure on Modal volume...")
    result = fix_structure.remote()
    print(f"\nResult: {result}")
    print("\n⚠️  IMPORTANT: Delete .cache files and restart training!")
