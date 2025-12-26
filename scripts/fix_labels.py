"""
Fix corrupt YOLO label files directly on Modal volume.
Problem: Labels have '\\n' literal strings instead of actual newlines.
"""

import modal

app = modal.App("fix-labels")
volume = modal.Volume.from_name("usd-dataset-test")
image = modal.Image.debian_slim("3.11").pip_install("tqdm")


@app.function(
    image=image,
    timeout=3600,
    volumes={"/data": volume},
)
def fix_labels():
    """Fix escaped newlines in all label files."""
    from pathlib import Path
    from tqdm import tqdm
    
    splits = ["train", "valid", "test"]
    total_fixed = 0
    
    for split in splits:
        labels_dir = Path(f"/data/{split}/labels")
        
        if not labels_dir.exists():
            print(f"⚠️ {split}/labels not found")
            continue
        
        label_files = list(labels_dir.glob("*.txt"))
        print(f"\n📝 Fixing {len(label_files)} labels in {split}...")
        
        for label_path in tqdm(label_files, desc=split):
            with open(label_path, 'r') as f:
                content = f.read()
            
            # Fix escaped newlines
            if '\\n' in content:
                fixed_content = content.replace('\\n', '\n')
                with open(label_path, 'w') as f:
                    f.write(fixed_content)
                total_fixed += 1
        
        # Delete old cache
        cache_path = Path(f"/data/{split}/labels.cache")
        if cache_path.exists():
            cache_path.unlink()
            print(f"  ✓ Deleted {split}/labels.cache")
    
    # Commit changes
    volume.commit()
    
    print(f"\n✅ Fixed {total_fixed} label files!")
    return {"fixed": total_fixed}


@app.local_entrypoint()
def main():
    print("🔧 Fixing label files on Modal volume...")
    result = fix_labels.remote()
    print(f"Result: {result}")
