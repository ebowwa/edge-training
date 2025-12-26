"""
Remove background images (images without labels) from Modal volume.
These are images that don't contain USD bills.

Set environment variable DRY_RUN=1 to preview changes without deleting.
"""

import modal
import os

app = modal.App("remove-backgrounds")
volume_name = os.getenv("VOLUME_NAME", "usd-dataset-test")
volume = modal.Volume.from_name(volume_name)
image = modal.Image.debian_slim("3.11").pip_install("tqdm")


@app.function(
    image=image,
    timeout=3600,
    volumes={"/data": volume},
)
def remove_backgrounds(dry_run: bool = False):
    """Remove images that don't have corresponding label files."""
    from pathlib import Path
    from tqdm import tqdm

    splits = ["train", "valid", "test"]
    total_removed = 0
    total_errors = 0
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    mode_str = "DRY RUN" if dry_run else "LIVE"
    print(f"🔍 Mode: {mode_str}")

    for split in splits:
        images_dir = Path(f"/data/{split}/images")
        labels_dir = Path(f"/data/{split}/labels")

        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️ {split} missing images or labels dir")
            continue

        # Get all image files with supported extensions
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
        label_stems = {f.stem for f in labels_dir.glob("*.txt")}

        print(f"\n📦 {split}: {len(image_files)} images, {len(label_stems)} labels")

        # Find images without labels
        to_remove = [img for img in image_files if img.stem not in label_stems]

        print(f"   Found {len(to_remove)} background images to remove...")

        for img in tqdm(to_remove, desc=f"{split}"):
            if dry_run:
                print(f"   Would remove: {img.name}")
                total_removed += 1
            else:
                try:
                    os.remove(img)
                    total_removed += 1
                except Exception as e:
                    print(f"   ❌ Error removing {img}: {e}")
                    total_errors += 1

        # Delete YOLO cache file (single cache per split)
        cache_path = Path(f"/data/{split}/{split}.cache")
        if cache_path.exists():
            if dry_run:
                print(f"   Would remove cache: {cache_path}")
            else:
                try:
                    os.remove(cache_path)
                    print(f"   ✓ Removed cache: {cache_path.name}")
                except Exception as e:
                    print(f"   ❌ Error removing cache: {e}")
                    total_errors += 1

        action = "Would remove" if dry_run else "Removed"
        print(f"   {action} {len(to_remove)} from {split}")

    if not dry_run:
        volume.commit()

    print(f"\n✅ Total removed: {total_removed} background images")
    if total_errors > 0:
        print(f"⚠️ Total errors: {total_errors}")
    return {"removed": total_removed, "errors": total_errors}


@app.local_entrypoint()
def main():
    import os

    dry_run = os.getenv("DRY_RUN", "0").strip() in ("1", "true", "yes")
    print("🧹 Removing background images from dataset...")
    result = remove_backgrounds.remote(dry_run=dry_run)
    print(f"Result: {result}")
