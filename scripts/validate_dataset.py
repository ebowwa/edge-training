"""
Pre-Training Dataset Validation Script

Runs comprehensive checks on the Modal volume before training:
1. Image/label counts per split
2. Label format validation
3. Class distribution
4. Bounding box validation
5. data.yaml consistency
"""

import modal

app = modal.App("dataset-validation")
volume = modal.Volume.from_name("usd-dataset-test", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pyyaml", "tqdm", "numpy"
)


@app.function(
    image=image,
    timeout=600,
    volumes={"/data": volume},
)
def validate_dataset():
    """Run comprehensive dataset validation."""
    from pathlib import Path
    import yaml
    from collections import Counter
    
    results = {
        "status": "pass",
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    data_path = Path("/data")
    
    # 1. Check data.yaml exists
    yaml_path = data_path / "data.yaml"
    if not yaml_path.exists():
        results["errors"].append("data.yaml not found!")
        results["status"] = "fail"
        return results
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    nc = config.get("nc", 0)
    names = config.get("names", {})
    print(f"✅ data.yaml: {nc} classes defined")
    results["stats"]["num_classes"] = nc
    
    # 2. Check each split
    splits = ["train", "valid", "test"]
    all_class_counts = Counter()
    
    for split in splits:
        split_path = data_path / split
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"
        
        if not images_dir.exists():
            results["errors"].append(f"{split}/images not found!")
            continue
        
        if not labels_dir.exists():
            results["errors"].append(f"{split}/labels not found!")
            continue
        
        # Count files
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        labels = list(labels_dir.glob("*.txt"))
        
        print(f"\n📁 {split}:")
        print(f"   Images: {len(images)}")
        print(f"   Labels: {len(labels)}")
        
        results["stats"][f"{split}_images"] = len(images)
        results["stats"][f"{split}_labels"] = len(labels)
        
        # Check for images without labels
        image_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}
        
        missing_labels = image_stems - label_stems
        if missing_labels:
            results["warnings"].append(f"{split}: {len(missing_labels)} images without labels")
            print(f"   ⚠️ {len(missing_labels)} images without labels")
        
        # 3. Validate label format and class distribution
        invalid_labels = 0
        out_of_range_classes = 0
        split_class_counts = Counter()
        
        for label_path in labels[:500]:  # Sample first 500
            try:
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            invalid_labels += 1
                            continue
                        
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        
                        # Check class ID range
                        if class_id < 0 or class_id >= nc:
                            out_of_range_classes += 1
                        
                        split_class_counts[class_id] += 1
                        
                        # Check bbox values (should be 0-1)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            invalid_labels += 1
            except Exception as e:
                invalid_labels += 1
        
        all_class_counts.update(split_class_counts)
        
        if invalid_labels > 0:
            results["errors"].append(f"{split}: {invalid_labels} invalid label lines")
            print(f"   ❌ {invalid_labels} invalid label lines")
        
        if out_of_range_classes > 0:
            results["errors"].append(f"{split}: {out_of_range_classes} class IDs out of range [0, {nc-1}]")
            print(f"   ❌ {out_of_range_classes} class IDs out of range")
    
    # 4. Class distribution
    print(f"\n📊 Class Distribution (sampled):")
    for class_id in sorted(all_class_counts.keys()):
        name = names.get(class_id, f"Unknown-{class_id}")
        count = all_class_counts[class_id]
        print(f"   {class_id}: {name} = {count}")
    
    results["stats"]["class_distribution"] = dict(all_class_counts)
    
    # Check for class imbalance
    if all_class_counts:
        max_count = max(all_class_counts.values())
        min_count = min(all_class_counts.values())
        if max_count > min_count * 10:
            results["warnings"].append(f"Significant class imbalance: max={max_count}, min={min_count}")
    
    # 5. Check for any class with < 50 samples
    rare_classes = [c for c, count in all_class_counts.items() if count < 50]
    if rare_classes:
        results["warnings"].append(f"Rare classes (< 50 samples): {rare_classes}")
    
    # Final status
    if results["errors"]:
        results["status"] = "fail"
    elif results["warnings"]:
        results["status"] = "warning"
    
    print(f"\n{'='*50}")
    print(f"STATUS: {results['status'].upper()}")
    if results["errors"]:
        print(f"ERRORS: {len(results['errors'])}")
        for e in results["errors"]:
            print(f"  ❌ {e}")
    if results["warnings"]:
        print(f"WARNINGS: {len(results['warnings'])}")
        for w in results["warnings"]:
            print(f"  ⚠️ {w}")
    
    return results


@app.local_entrypoint()
def main():
    """Run dataset validation."""
    print("🔍 Running Pre-Training Dataset Validation...\n")
    results = validate_dataset.remote()
    print(f"\n📋 Final Results: {results}")
