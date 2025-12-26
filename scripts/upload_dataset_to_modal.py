"""
Add train and valid splits to existing usd-dataset-test volume.
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.modal.volumes import ModalVolumeManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_remaining_splits():
    """Add train and valid to existing volume."""
    print("\n📦 Adding remaining splits to usd-dataset-test\n")
    
    dataset_path = Path(__file__).parent.parent / "datasets" / "usd_detection"
    
    # Use existing volume
    mgr = ModalVolumeManager("usd-dataset-test")
    
    # Upload train and valid
    for split in ["train", "valid"]:
        split_path = dataset_path / split
        
        if not split_path.exists():
            print(f"⚠️  {split} not found, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"📤 Uploading {split} split...")
        print(f"{'='*60}")
        
        result = mgr.upload_dataset(
            split_path,
            remote_path=f"/{split}",
            force=True,
            max_retries=5
        )
        
        if result["success"]:
            print(f"✅ {split}: {result['size_bytes'] / (1024**3):.1f}GB uploaded")
        else:
            print(f"❌ {split} failed: {result.get('error', 'Unknown')}")
            return False
    
    # Verify all splits
    print(f"\n{'='*60}")
    print("🔍 Verifying all splits...")
    print(f"{'='*60}")
    
    for split in ["test", "train", "valid"]:
        files = mgr.list_files(f"/{split}")
        if files:
            print(f"✓ {split}: {len(files)} files")
        else:
            print(f"⚠️  {split}: no files found")
    
    print("\n✅ Full dataset in 'usd-dataset-test' volume!")
    
    return True


if __name__ == "__main__":
    success = add_remaining_splits()
    sys.exit(0 if success else 1)
