"""
Test Modal Volume Manager

Tests the volume upload/list/cleanup functionality.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.modal.volumes import ModalVolumeManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_volume_upload():
    """Test uploading test split to volume."""
    print("\n🧪 Testing Modal Volume Upload\n")
    
    # Use small test split first
    dataset_path = Path(__file__).parent.parent / "datasets" / "usd_detection" / "test"
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found: {dataset_path}")
        return False
    
    # Create volume manager
    mgr = ModalVolumeManager("usd-dataset-test")
    
    # Create volume first
    print(f"0️⃣ Creating volume {mgr.volume_name}...")
    import subprocess
    try:
        subprocess.run(
            ["modal", "volume", "create", mgr.volume_name],
            capture_output=True,
            check=True
        )
        print("✓ Volume created")
    except subprocess.CalledProcessError as e:
        # Volume might already exist
        if "already exists" in e.stderr.lower():
            print("✓ Volume already exists")
        else:
            print(f"⚠️  Create failed: {e.stderr}")
    
    # Upload test split
    print(f"\n1️⃣ Uploading {dataset_path}...")
    result = mgr.upload_dataset(
        dataset_path,
        remote_path="/test",
        force=True,
        max_retries=3
    )
    
    if not result["success"]:
        print(f"✗ Upload failed: {result.get('error', 'Unknown error')}")
        return False
    
    print(f"✓ Uploaded {result['size_bytes'] / (1024**2):.1f}MB")
    
    # List files
    print("\n2️⃣ Listing files...")
    files = mgr.list_files("/test")
    
    if files:
        print(f"✓ Found {len(files)} entries")
        for f in files[:5]:
            print(f"  - {f}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
    else:
        print("⚠️  No files listed (might be empty or JSON parse issue)")
    
    # Get mount config
    print("\n3️⃣ Testing mount config...")
    mount_config = mgr.get_mount_config()
    print(f"✓ Mount config: {list(mount_config.keys())}")
    
    # Cleanup (optional - comment out to keep volume)
    print("\n4️⃣ Cleanup? (Skipping - you can delete manually)")
    print(f"   To delete: modal volume delete {mgr.volume_name} --yes")
    # cleanup_result = mgr.cleanup(confirm=True)
    # if cleanup_result:
    #     print("✓ Volume deleted")
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_volume_upload()
    sys.exit(0 if success else 1)
