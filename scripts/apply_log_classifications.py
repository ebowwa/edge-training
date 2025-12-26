#!/usr/bin/env python3
"""
Apply counterfeit classifications from log output.

This script parses log lines like:
  2025-12-24 15:40:51,273 - INFO - [UPDATED] btend-f46_jpg.rf.5c93d885c248bd6a128888fd2bcb9cc1.jpg: Counterfeit 10USD -> Counterfeit 10USD Back

And applies those classifications to the COCO annotation file.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Regex to parse log lines
LOG_PATTERN = re.compile(
    r'\[UPDATED\]\s+([^:]+):\s+(.+?)\s+->\s+(.+?)$'
)


def parse_log_file(log_path: Path) -> dict:
    """Parse log file and return {filename: new_class} mapping."""
    mappings = {}
    
    with open(log_path) as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if match:
                filename = match.group(1).strip()
                old_class = match.group(2).strip()
                new_class = match.group(3).strip()
                mappings[filename] = {
                    'old': old_class,
                    'new': new_class
                }
    
    return mappings


def apply_to_coco(coco_path: Path, mappings: dict, output_path: Path):
    """Apply classifications to COCO annotation file."""
    
    with open(coco_path) as f:
        coco = json.load(f)
    
    # Build image_id -> filename lookup
    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    filename_to_id = {v: k for k, v in id_to_filename.items()}
    
    # Build category name -> id lookup
    name_to_cat_id = {cat['name']: cat['id'] for cat in coco['categories']}
    
    # Track new categories we need to add
    new_categories = {}
    max_cat_id = max(cat['id'] for cat in coco['categories'])
    
    # First pass: identify new categories needed
    for filename, mapping in mappings.items():
        new_class = mapping['new']
        if new_class not in name_to_cat_id and new_class not in new_categories:
            max_cat_id += 1
            new_categories[new_class] = max_cat_id
            print(f"  New category: {new_class} (ID: {max_cat_id})")
    
    # Add new categories to COCO
    for name, cat_id in new_categories.items():
        coco['categories'].append({
            'id': cat_id,
            'name': name,
            'supercategory': 'currency'
        })
        name_to_cat_id[name] = cat_id
    
    # Second pass: update annotations
    updated = 0
    skipped = 0
    errors = []
    
    for ann in coco['annotations']:
        image_id = ann['image_id']
        filename = id_to_filename.get(image_id)
        
        if filename in mappings:
            mapping = mappings[filename]
            old_class = mapping['old']
            new_class = mapping['new']
            
            # Verify the old class matches
            current_cat_id = ann['category_id']
            current_name = None
            for cat in coco['categories']:
                if cat['id'] == current_cat_id:
                    current_name = cat['name']
                    break
            
            if current_name == old_class:
                # Update to new class
                new_cat_id = name_to_cat_id[new_class]
                ann['category_id'] = new_cat_id
                updated += 1
            elif current_name == new_class:
                # Already updated (maybe from original dataset)
                skipped += 1
            else:
                errors.append(f"{filename}: expected '{old_class}' but found '{current_name}'")
    
    # Save updated COCO
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    
    return updated, skipped, errors


def main():
    if len(sys.argv) < 3:
        print("Usage: python apply_log_classifications.py <log_file> <coco_file> [output_file]")
        print("  log_file: Path to file containing log lines with [UPDATED] entries")
        print("  coco_file: Path to COCO annotation JSON file")
        print("  output_file: (optional) Output file, defaults to coco_file")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    coco_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else coco_path
    
    print(f"Parsing log file: {log_path}")
    mappings = parse_log_file(log_path)
    print(f"Found {len(mappings)} classifications")
    
    if not mappings:
        print("No classifications found in log. Check the format.")
        sys.exit(1)
    
    # Show sample mappings
    print("\nSample mappings:")
    for i, (filename, mapping) in enumerate(list(mappings.items())[:5]):
        print(f"  {filename}: {mapping['old']} -> {mapping['new']}")
    print(f"  ... and {len(mappings) - 5} more\n")
    
    print(f"Applying to COCO file: {coco_path}")
    print(f"Output file: {output_path}")
    
    updated, skipped, errors = apply_to_coco(coco_path, mappings, output_path)
    
    print(f"\nResults:")
    print(f"  Updated: {updated}")
    print(f"  Skipped (already correct): {skipped}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print(f"\nDone! Saved to: {output_path}")


if __name__ == "__main__":
    main()
