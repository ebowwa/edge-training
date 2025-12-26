"""
Add train and valid splits using the same approach that worked for test.
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.modal.volumes import ModalVolumeManager

logging.basicConfig(level=logging.INFO)

def add_train_and_valid():
    print("\n📦 Adding train and valid to usd-dataset-test\n")
    
    mgr = ModalVolumeManager("usd-dataset-test")
    
    # Add train
    train_path = Path(__file__).parent.parent / "datasets" / "usd_detection" / "train"
    print("1️⃣ Uploading train...")
    result = mgr.upload_dataset(train_path, "/train", force=True, max_retries=3)
    
    if result["success"]:
        print(f"✓ Train: {result['size_bytes'] / (1024**3):.1f}GB")
    else:
        print(f"✗ Train failed: {result.get('error')}")
        return False
    
    # Add valid
    valid_path = Path(__file__).parent.parent / "datasets" / "usd_detection" / "valid"
    print("\n2️⃣ Uploading valid...")
    result = mgr.upload_dataset(valid_path, "/valid", force=True, max_retries=3)
    
    if result["success"]:
        print(f"✓ Valid: {result['size_bytes'] / (1024**3):.1f}GB")
    else:
        print(f"✗ Valid failed: {result.get('error')}")
        return False
    
    # Verify
    print("\n3️⃣ Verifying...")
    for split in ["test", "train", "valid"]:
        files = mgr.list_files(f"/{split}")
        print(f"  {split}: {len(files) if files else 0} files")
    
    print("\n✅ Done!")
    return True


if __name__ == "__main__":
    success = add_train_and_valid()
    sys.exit(0 if success else 1)
