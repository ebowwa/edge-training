#!/usr/bin/env python3
"""Upload USD detection dataset to HuggingFace using requests."""

import json
import os
import sys
from pathlib import Path
import tarfile
import tempfile

def create_tarball(dataset_dir):
    """Create a tarball of the dataset."""
    print("📦 Creating tarball...")
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tar_path = tmp.name
    
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(dataset_dir, arcname='usd_detection')
    
    print(f"✅ Created tarball: {tar_path}")
    return tar_path

def upload_to_hf():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set")
        return 1
    repo_id = "ebowwa/usd-detection"
    dataset_dir = Path("datasets/usd_detection")
    
    try:
        # Try importing requests
        import requests
    except ImportError:
        print("ERROR: requests not installed")
        print("Install with: pip3 install requests")
        return 1
    
    print(f"📤 Uploading dataset to {repo_id}")
    print(f"📁 Dataset directory: {dataset_dir}")
    
    # Create repository
    api_url = "https://huggingface.co/api"
    headers = {"Authorization": f"Bearer {token}"}
    
    print("\n1️⃣  Creating repository...")
    response = requests.post(
        f"{api_url}/repos/create",
        headers=headers,
        json={
            "name": repo_id.split('/')[-1],
            "type": "dataset",
            "private": False,
        }
    )
    
    if response.status_code == 409:
        print("✅ Repository already exists")
    elif response.status_code in [200, 201]:
        print("✅ Repository created")
    else:
        print(f"⚠️  Response: {response.status_code} - {response.text}")
    
    print(f"\n2️⃣  Uploading files...")
    
    # Upload files one by one
    files_to_upload = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(dataset_dir)
            files_to_upload.append((file_path, str(rel_path)))
    
    print(f"Found {len(files_to_upload)} files to upload")
    
    for i, (file_path, rel_path) in enumerate(files_to_upload, 1):
        if file_path.stat().st_size > 50 * 1024 * 1024:  # Skip files > 50MB for now
            print(f"⏭️  Skipping large file ({i}/{len(files_to_upload)}): {rel_path}")
            continue
        
        print(f"📤 ({i}/{len(files_to_upload)}): {rel_path}...", end='', flush=True)
        
        with open(file_path, 'rb') as f:
            response = requests.put(
                f"https://huggingface.co/{repo_id}/raw/main/{rel_path}",
                headers=headers,
                data=f
            )
        
        if response.status_code in [200, 201]:
            print(" ✅")
        else:
            print(f" ❌ {response.status_code}")
    
    print(f"\n✅ Upload complete!")
    print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
    return 0

if __name__ == "__main__":
    exit(upload_to_hf())
