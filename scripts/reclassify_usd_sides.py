#!/usr/bin/env python3
"""
Reclassify USD Side Labels

Takes generic USD labels (e.g., "100USD") and reclassifies them to 
specific Front/Back variants (e.g., "100USD-Front", "100USD-Back") 
using Roboflow inference.

Usage:
    .venv/bin/python scripts/reclassify_usd_sides.py \
        --dataset datasets/usd_detection \
        --model-id front-back-of-usd/2 \
        --confidence 0.5 \
        --dry-run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Map generic dataset classes to Roboflow model predictions
# Dataset uses: 100USD, 50USD, etc.
# Model returns: 100-front, 100-back, fifty-front, fifty-back, etc.

# Mapping from dataset generic class -> (front_coco_name, back_coco_name)
GENERIC_TO_SPECIFIC = {
    "100USD": ("100USD-Front", "100USD-Back"),
    "50USD": ("50USD-Front", "50USD-Back"),
    "20USD": ("20USD-Front", "20USD-Back"),
    "10USD": ("10USD-Front", "10USD-Back"),
    "5USD": ("5USD-Front", "5USD-Back"),
    "1USD": ("1USD-Front", "1USD-Back"),
}

# Counterfeit classes to reclassify
COUNTERFEIT_GENERIC_TO_SPECIFIC = {
    "Counterfeit 100 USD": ("Counterfeit 100 USD Front", "Counterfeit 100 USD Back"),
    "Counterfeit 50USD": ("Counterfeit 50USD Front", "Counterfeit 50USD Back"),
    "Counterfeit 20USD": ("Counterfeit 20USD Front", "Counterfeit 20USD Back"),
    "Counterfeit 10USD": ("Counterfeit 10USD Front", "Counterfeit 10USD Back"),
    "Counterfeit 5USD": ("Counterfeit 5USD Front", "Counterfeit 5USD Back"),
    "Counterfeit 2USD": ("Counterfeit 2USD Front", "Counterfeit 2USD Back"),
    "Counterfeit 1USD": ("Counterfeit 1USD Front", "Counterfeit 1USD Back"),
}

# Mapping from Roboflow model class names -> dataset class names
# Model uses: 100-front, 100-back, fifty-front, fifty-back, etc.
MODEL_TO_DATASET = {
    "100-front": "100USD-Front",
    "100-back": "100USD-Back",
    "fifty-front": "50USD-Front",
    "fifty-back": "50USD-Back",
    "twenty-front": "20USD-Front",
    "twenty-back": "20USD-Back",
    "ten-front": "10USD-Front",
    "ten-back": "10USD-Back",
    "five-front": "5USD-Front",
    "five-back": "5USD-Back",
    "one-front": "1USD-Front",
    "one-back": "1USD-Back",
}

# Counterfeit model mappings (same model classes, different output names)
MODEL_TO_COUNTERFEIT = {
    "100-front": "Counterfeit 100 USD Front",
    "100-back": "Counterfeit 100 USD Back",
    "fifty-front": "Counterfeit 50USD Front",
    "fifty-back": "Counterfeit 50USD Back",
    "twenty-front": "Counterfeit 20USD Front",
    "twenty-back": "Counterfeit 20USD Back",
    "ten-front": "Counterfeit 10USD Front",
    "ten-back": "Counterfeit 10USD Back",
    "five-front": "Counterfeit 5USD Front",
    "five-back": "Counterfeit 5USD Back",
    "two-front": "Counterfeit 2USD Front",
    "two-back": "Counterfeit 2USD Back",
    "one-front": "Counterfeit 1USD Front",
    "one-back": "Counterfeit 1USD Back",
}

# Reverse mapping: dataset class prefix -> model class prefix
DATASET_PREFIX_TO_MODEL = {
    "100USD": "100",
    "50USD": "fifty",
    "20USD": "twenty",
    "10USD": "ten",
    "5USD": "five",
    "1USD": "one",
    # Counterfeit mappings
    "Counterfeit 100 USD": "100",
    "Counterfeit 50USD": "fifty",
    "Counterfeit 20USD": "twenty",
    "Counterfeit 10USD": "ten",
    "Counterfeit 5USD": "five",
    "Counterfeit 2USD": "two",
    "Counterfeit 1USD": "one",
}


class USDSideReclassifier:
    """Reclassify generic USD labels to Front/Back variants."""
    
    def __init__(
        self,
        model_id: str,
        api_key: str,
        confidence_threshold: float = 0.5,
        counterfeit_only: bool = False,
    ):
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.counterfeit_only = counterfeit_only
        
        # Initialize Roboflow client directly
        from inference_sdk import InferenceHTTPClient
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        
        # Build lookup for generic classes to reclassify
        if counterfeit_only:
            self.generic_classes = set(COUNTERFEIT_GENERIC_TO_SPECIFIC.keys())
        else:
            self.generic_classes = set(GENERIC_TO_SPECIFIC.keys())
        
        logger.info(f"Initialized with model: {model_id}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        logger.info(f"Counterfeit only: {counterfeit_only}")
        logger.info(f"Generic classes to reclassify: {len(self.generic_classes)}")
    
    def load_coco_annotations(self, json_path: Path) -> Dict:
        """Load COCO format annotations."""
        with open(json_path) as f:
            return json.load(f)
    
    def save_coco_annotations(self, data: Dict, json_path: Path):
        """Save COCO format annotations."""
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_category_map(self, coco_data: Dict) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Create category ID <-> name mappings."""
        id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
        return id_to_name, name_to_id
    
    def find_annotations_to_reclassify(
        self,
        coco_data: Dict,
    ) -> List[Tuple[Dict, str]]:
        """Find annotations with generic class labels."""
        id_to_name, _ = self.get_category_map(coco_data)
        
        to_reclassify = []
        for ann in coco_data['annotations']:
            cat_name = id_to_name.get(ann['category_id'], '')
            if cat_name in self.generic_classes:
                to_reclassify.append((ann, cat_name))
        
        return to_reclassify
    
    def get_image_path(
        self,
        coco_data: Dict,
        image_id: int,
        images_dir: Path,
    ) -> Optional[Path]:
        """Get image path from image ID."""
        for img in coco_data['images']:
            if img['id'] == image_id:
                return images_dir / img['file_name']
        return None
    
    def infer_side(
        self,
        image_path: Path,
        generic_class: str,
    ) -> Optional[str]:
        """
        Run inference to determine Front or Back.
        
        Returns the specific class name (e.g., "100USD-Front") or None if
        inference doesn't provide a confident answer.
        """
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        try:
            # Run Roboflow classification
            result = self.client.infer(str(image_path), model_id=self.model_id)
            
            predictions = result.get('predictions', {})
            if not predictions:
                logger.debug(f"No predictions for {image_path}")
                return None
            
            # Get the model class prefix for this dataset class
            model_prefix = DATASET_PREFIX_TO_MODEL.get(generic_class)
            if not model_prefix:
                logger.debug(f"No model mapping for {generic_class}")
                return None
            
            # Look for front/back predictions matching this denomination
            front_key = f"{model_prefix}-front"
            back_key = f"{model_prefix}-back"
            
            front_conf = predictions.get(front_key, {}).get('confidence', 0.0)
            back_conf = predictions.get(back_key, {}).get('confidence', 0.0)
            
            # Choose mapping based on whether this is a counterfeit class
            is_counterfeit = generic_class.startswith("Counterfeit")
            mapping = MODEL_TO_COUNTERFEIT if is_counterfeit else MODEL_TO_DATASET
            
            # Pick the higher confidence side
            if front_conf > back_conf and front_conf >= self.confidence_threshold:
                dataset_class = mapping.get(front_key)
                logger.debug(f"Reclassified {generic_class} -> {dataset_class} ({front_conf:.2f})")
                return dataset_class
            elif back_conf >= self.confidence_threshold:
                dataset_class = mapping.get(back_key)
                logger.debug(f"Reclassified {generic_class} -> {dataset_class} ({back_conf:.2f})")
                return dataset_class
            
            # Fallback: Use generalization approach (aggregate all front vs back)
            # This helps when model lacks specific denomination classes (e.g., $2 bills)
            if front_conf == 0.0 and back_conf == 0.0:
                # Model doesn't have this specific denomination
                # Aggregate ALL front vs ALL back predictions
                total_front = sum(v.get('confidence', 0.0) for k, v in predictions.items() if k.endswith('-front'))
                total_back = sum(v.get('confidence', 0.0) for k, v in predictions.items() if k.endswith('-back'))
                
                # Use a lower threshold for generalization since we're aggregating
                generalization_threshold = 0.05
                if total_front > total_back and total_front >= generalization_threshold:
                    # Infer Front based on general front features
                    dataset_class = mapping.get(front_key) or f"{generic_class} Front"
                    logger.debug(f"Reclassified {generic_class} -> {dataset_class} (generalized front: {total_front:.4f})")
                    return dataset_class
                elif total_back >= generalization_threshold:
                    # Infer Back based on general back features
                    dataset_class = mapping.get(back_key) or f"{generic_class} Back"
                    logger.debug(f"Reclassified {generic_class} -> {dataset_class} (generalized back: {total_back:.4f})")
                    return dataset_class
            
            # Neither side meets threshold
            logger.debug(f"Low confidence for {generic_class}: front={front_conf:.2f}, back={back_conf:.2f}")
            return None
            
        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {e}")
            return None
    
    def ensure_category_exists(
        self,
        coco_data: Dict,
        category_name: str,
    ) -> int:
        """Ensure category exists, create if not. Returns category ID."""
        _, name_to_id = self.get_category_map(coco_data)
        
        if category_name in name_to_id:
            return name_to_id[category_name]
        
        # Create new category
        max_id = max(cat['id'] for cat in coco_data['categories'])
        new_id = max_id + 1
        
        coco_data['categories'].append({
            'id': new_id,
            'name': category_name,
            'supercategory': 'currency'
        })
        
        logger.info(f"Created new category: {category_name} (ID: {new_id})")
        return new_id
    
    def reclassify_split(
        self,
        split_dir: Path,
        dry_run: bool = False,
        limit: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Reclassify annotations in a single split (train/valid/test).
        
        Returns statistics dict.
        """
        ann_file = split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            logger.warning(f"No annotations found: {ann_file}")
            return {"skipped": 0, "reclassified": 0, "failed": 0}
        
        # Load annotations
        coco_data = self.load_coco_annotations(ann_file)
        
        # Find annotations to reclassify
        to_reclassify = self.find_annotations_to_reclassify(coco_data)
        
        if limit:
            to_reclassify = to_reclassify[:limit]
        
        logger.info(f"Found {len(to_reclassify)} annotations to reclassify in {split_dir.name}")
        
        stats = {"skipped": 0, "reclassified": 0, "failed": 0}
        save_interval = 50  # Save every N successful reclassifications
        output_file = split_dir / "_annotations.reclassified.coco.json"
        
        for i, (ann, generic_class) in enumerate(to_reclassify):
            # Get image path
            image_path = self.get_image_path(coco_data, ann['image_id'], split_dir)
            
            if not image_path:
                stats["skipped"] += 1
                continue
            
            # Infer the specific side
            specific_class = self.infer_side(image_path, generic_class)
            
            if specific_class:
                if not dry_run:
                    # Update annotation
                    new_cat_id = self.ensure_category_exists(coco_data, specific_class)
                    ann['category_id'] = new_cat_id
                
                stats["reclassified"] += 1
                logger.info(f"[{'DRY-RUN' if dry_run else 'UPDATED'}] {image_path.name}: {generic_class} -> {specific_class}")
                
                # Incremental save every N items
                if not dry_run and stats["reclassified"] % save_interval == 0:
                    self.save_coco_annotations(coco_data, output_file)
                    logger.info(f"Progress saved: {stats['reclassified']} items ({output_file})")
            else:
                stats["failed"] += 1
        
        # Final save
        if not dry_run and stats["reclassified"] > 0:
            self.save_coco_annotations(coco_data, output_file)
            logger.info(f"Final save: {output_file}")
        
        return stats
    
    def reclassify_dataset(
        self,
        dataset_dir: Path,
        dry_run: bool = False,
        limit: Optional[int] = None,
        splits: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Reclassify all splits in a dataset.
        
        Args:
            dataset_dir: Path to dataset root
            dry_run: If True, don't save changes
            limit: Max annotations to process per split
            splits: Which splits to process (default: train, valid, test)
        
        Returns:
            Statistics per split
        """
        splits = splits or ["train", "valid", "test"]
        all_stats = {}
        
        for split in splits:
            split_dir = dataset_dir / split
            if split_dir.exists():
                logger.info(f"\n=== Processing {split} ===")
                all_stats[split] = self.reclassify_split(split_dir, dry_run, limit)
            else:
                logger.warning(f"Split not found: {split_dir}")
        
        return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Reclassify generic USD labels to Front/Back variants"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/usd_detection",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="front-back-of-usd/2",
        help="Roboflow model ID"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of annotations to process (for testing)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        help="Which splits to process"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save changes, just show what would be done"
    )
    parser.add_argument(
        "--counterfeit-only",
        action="store_true",
        help="Only reclassify counterfeit classes (not regular USD)"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("ERROR: --api-key required or set ROBOFLOW_API_KEY env var")
        sys.exit(1)
    
    # Initialize reclassifier
    reclassifier = USDSideReclassifier(
        model_id=args.model_id,
        api_key=args.api_key,
        confidence_threshold=args.confidence,
        counterfeit_only=args.counterfeit_only,
    )
    
    # Run reclassification
    dataset_path = Path(args.dataset)
    stats = reclassifier.reclassify_dataset(
        dataset_path,
        dry_run=args.dry_run,
        limit=args.limit,
        splits=args.splits,
    )
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    total_reclassified = 0
    total_failed = 0
    total_skipped = 0
    
    for split, split_stats in stats.items():
        print(f"\n{split}:")
        print(f"  Reclassified: {split_stats['reclassified']}")
        print(f"  Failed:       {split_stats['failed']}")
        print(f"  Skipped:      {split_stats['skipped']}")
        
        total_reclassified += split_stats['reclassified']
        total_failed += split_stats['failed']
        total_skipped += split_stats['skipped']
    
    print(f"\nTOTAL:")
    print(f"  Reclassified: {total_reclassified}")
    print(f"  Failed:       {total_failed}")
    print(f"  Skipped:      {total_skipped}")
    
    if args.dry_run:
        print("\n[DRY-RUN] No changes were saved.")
        print("Remove --dry-run to apply changes.")


if __name__ == "__main__":
    main()
