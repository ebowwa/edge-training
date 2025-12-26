#!/usr/bin/env python3
"""
Upload USD Detection dataset to HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py --repo-id <username/dataset-name>
    
Example:
    python scripts/upload_to_hf.py --repo-id ebowwa/usd-detection
"""

import argparse
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description='Upload USD detection dataset to HuggingFace')
    parser.add_argument('--repo-id', required=True, help='HuggingFace repo ID (e.g., username/dataset-name)')
    parser.add_argument('--token', help='HuggingFace API token (or set HF_TOKEN env var)')
    parser.add_argument('--private', action='store_true', help='Make dataset private')
    parser.add_argument('--dataset-dir', default='datasets/usd_detection', help='Path to dataset directory')
    
    args = parser.parse_args()
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Install with: pip install 'huggingface_hub[cli]'")
        return 1
    
    # Get token
    token = args.token or os.getenv('HF_TOKEN')
    if not token:
        print("ERROR: No HuggingFace token provided")
        print("Either use --token or set HF_TOKEN environment variable")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return 1
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return 1
    
    print(f"📦 Uploading dataset from: {dataset_dir}")
    print(f"🎯 Target repo: {args.repo_id}")
    print(f"🔒 Private: {args.private}")
    
    # Create API client
    api = HfApi()
    
    # Create repository
    try:
        print("\n📝 Creating repository...")
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=token
        )
        print("✅ Repository created/exists")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return 1
    
    # Upload files
    try:
        print("\n📤 Uploading files...")
        
        # Upload entire dataset folder
        api.upload_folder(
            folder_path=str(dataset_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
        )
        
        print(f"\n✅ Dataset uploaded successfully!")
        print(f"🔗 View at: https://huggingface.co/datasets/{args.repo_id}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
