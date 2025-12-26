#!/usr/bin/env python3
"""
Verify and Correct USD Front/Back Classifications

Re-runs Roboflow inference on ALL annotations to verify and correct
any misclassified Front/Back labels.

Usage:
    python scripts/verify_classifications.py --dataset-dir datasets/usd_detection
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from inference_sdk import InferenceHTTPClient

# Model class to dataset class mapping
MODEL_TO_FRONT_BACK = {
    "100-front": "Front", "100-back": "Back",
    "fifty-front": "Front", "fifty-back": "Back",
    "twenty-front": "Front", "twenty-back": "Back",
    "ten-front": "Front", "ten-back": "Back",
    "five-front": "Front", "five-back": "Back",
    "one-front": "Front", "one-back": "Back",
    "two-front": "Front", "two-back": "Back",
}

# Map denomination names
DENOM_MAP = {
    "100USD": "100", "10USD": "ten", "1USD": "one",
    "20USD": "twenty", "50USD": "fifty", "5USD": "five",
    "Counterfeit 100 USD": "100", "Counterfeit 10USD": "ten",
    "Counterfeit 1USD": "one", "Counterfeit 20USD": "twenty",
    "Counterfeit 50USD": "fifty", "Counterfeit 5USD": "five",
}


def get_base_class(class_name: str) -> tuple:
    """Extract base denomination and current side from class name."""
    for base in ["100USD", "50USD", "20USD", "10USD", "5USD", "1USD"]:
        if base in class_name:
            is_counterfeit = "Counterfeit" in class_name
            current_side = "Front" if "Front" in class_name else "Back"
            return base, is_counterfeit, current_side
    return None, None, None


def infer_side(client, model_id: str, image_path: Path, denom: str) -> str:
    """Run inference to determine Front or Back."""
    try:
        result = client.infer(str(image_path), model_id=model_id)
        predictions = result.get('predictions', {})
        
        if not predictions:
            return None
        
        # Get model prefix for this denomination
        model_prefix = DENOM_MAP.get(denom)
        if not model_prefix:
            return None
        
        front_key = f"{model_prefix}-front"
        back_key = f"{model_prefix}-back"
        
        front_conf = predictions.get(front_key, {}).get('confidence', 0.0)
        back_conf = predictions.get(back_key, {}).get('confidence', 0.0)
        
        if front_conf > back_conf and front_conf >= 0.5:
            return "Front"
        elif back_conf >= 0.5:
            return "Back"
        
        # Fallback: aggregate all front/back predictions
        total_front = sum(p.get('confidence', 0) for k, p in predictions.items() if 'front' in k)
        total_back = sum(p.get('confidence', 0) for k, p in predictions.items() if 'back' in k)
        
        if total_front > total_back:
            return "Front"
        elif total_back > 0:
            return "Back"
        
        return None
        
    except Exception as e:
        print(f"Error inferring {image_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/usd_detection")
    parser.add_argument("--model-id", default="usd-classification/1")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: ROBOFLOW_API_KEY not set")
        return
    
    client = InferenceHTTPClient(
        api_url="https://classify.roboflow.com",
        api_key=args.api_key
    )
    
    dataset_dir = Path(args.dataset_dir)
    
    stats = defaultdict(lambda: {"checked": 0, "correct": 0, "corrected": 0, "failed": 0})
    
    for split in args.splits:
        split_dir = dataset_dir / split
        ann_path = split_dir / "_annotations.coco.json"
        
        if not ann_path.exists():
            print(f"Skipping {split}: no annotations")
            continue
        
        print(f"\n=== Processing {split} ===")
        
        with open(ann_path) as f:
            coco = json.load(f)
        
        cats = {c['id']: c['name'] for c in coco['categories']}
        name_to_id = {c['name']: c['id'] for c in coco['categories']}
        imgs = {i['id']: i['file_name'] for i in coco['images']}
        
        changes = []
        count = 0
        
        for ann in coco['annotations']:
            if args.limit and count >= args.limit:
                break
            
            cat_name = cats.get(ann['category_id'], '')
            base, is_counterfeit, current_side = get_base_class(cat_name)
            
            if not base:
                continue
            
            count += 1
            stats[split]["checked"] += 1
            
            # Get image path
            img_file = imgs.get(ann['image_id'])
            img_path = split_dir / img_file
            
            if not img_path.exists():
                stats[split]["failed"] += 1
                continue
            
            # Run inference
            denom = f"Counterfeit {base}" if is_counterfeit else base
            predicted_side = infer_side(client, args.model_id, img_path, denom)
            
            if not predicted_side:
                stats[split]["failed"] += 1
                continue
            
            if predicted_side == current_side:
                stats[split]["correct"] += 1
            else:
                stats[split]["corrected"] += 1
                
                # Build new class name
                if is_counterfeit:
                    new_class = f"Counterfeit {base.replace('USD', ' USD')} {predicted_side}".replace("  ", " ")
                else:
                    new_class = f"{base}-{predicted_side}"
                
                print(f"  {img_file}: {cat_name} -> {new_class}")
                
                # Update annotation
                if new_class in name_to_id:
                    changes.append((ann, name_to_id[new_class]))
                else:
                    print(f"    Warning: {new_class} not in categories")
            
            # Rate limit
            time.sleep(0.1)
            
            if count % 50 == 0:
                print(f"  Processed {count}...")
        
        # Apply changes
        if not args.dry_run and changes:
            for ann, new_cat_id in changes:
                ann['category_id'] = new_cat_id
            
            with open(ann_path, 'w') as f:
                json.dump(coco, f, indent=2)
            print(f"  Saved {len(changes)} corrections to {ann_path}")
    
    # Summary
    print("\n=== Summary ===")
    for split, s in stats.items():
        total = s["checked"]
        if total:
            acc = s["correct"] / total * 100
            print(f"{split}: {s['checked']} checked, {s['correct']} correct ({acc:.1f}%), {s['corrected']} corrected, {s['failed']} failed")


if __name__ == "__main__":
    main()
