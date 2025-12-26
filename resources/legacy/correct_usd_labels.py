"""
USD Dataset Label Correction

Uses Roboflow usd-classification/1 model to verify/correct labels in the dataset.
Run with: doppler run --project seed -- python scripts/correct_usd_labels.py
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of label correction for an image."""
    image_path: str
    original_label: str
    predicted_label: str
    confidence: float
    needs_review: bool  # True if prediction differs from original


def load_coco_annotations(annotations_path: str) -> Tuple[Dict, Dict[int, str]]:
    """Load COCO annotations and return (annotations, category_id_to_name)."""
    with open(annotations_path) as f:
        coco = json.load(f)
    
    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
    return coco, cat_map


def correct_dataset(
    images_dir: str,
    annotations_path: str,
    output_path: str,
    model_id: str = "usd-classification/1",
    confidence_threshold: float = 0.7,
    dry_run: bool = True,
) -> List[CorrectionResult]:
    """
    Correct dataset labels using Roboflow classification model.
    
    Args:
        images_dir: Directory containing images
        annotations_path: Path to COCO annotations JSON
        output_path: Path to write corrected annotations
        model_id: Roboflow model for classification
        confidence_threshold: Only correct if prediction confidence > threshold
        dry_run: If True, don't write changes, just report
    """
    from integrations.roboflow import RoboflowConfig, RoboflowProvider
    
    # Initialize Roboflow
    config = RoboflowConfig(model_id=model_id)
    rf = RoboflowProvider(config)
    
    # Load annotations
    coco, cat_map = load_coco_annotations(annotations_path)
    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    
    # Build image_id -> annotations mapping
    img_annotations = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    results = []
    corrections_made = 0
    
    for img_id, img_info in images.items():
        image_path = Path(images_dir) / img_info["file_name"]
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        
        # Get original labels for this image
        img_anns = img_annotations.get(img_id, [])
        if not img_anns:
            continue
        
        # Run classification
        try:
            classification = rf.classify(str(image_path))
            if not classification:
                continue
            
            # Get top prediction
            top_class = max(classification, key=classification.get)
            top_confidence = classification[top_class]
            
        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {e}")
            continue
        
        # Check each annotation
        for ann in img_anns:
            original_label = cat_map.get(ann["category_id"], "unknown")
            
            # Compare with prediction
            needs_review = (
                top_class.lower().replace(" ", "-") != 
                original_label.lower().replace(" ", "-")
            )
            
            result = CorrectionResult(
                image_path=str(image_path),
                original_label=original_label,
                predicted_label=top_class,
                confidence=top_confidence,
                needs_review=needs_review,
            )
            results.append(result)
            
            if needs_review and top_confidence > confidence_threshold:
                logger.info(
                    f"MISMATCH: {img_info['file_name']} - "
                    f"original: {original_label}, predicted: {top_class} ({top_confidence:.2f})"
                )
                
                if not dry_run:
                    # Find or create category for predicted label
                    pred_cat_id = None
                    for cat_id, cat_name in cat_map.items():
                        if cat_name.lower() == top_class.lower():
                            pred_cat_id = cat_id
                            break
                    
                    if pred_cat_id:
                        ann["category_id"] = pred_cat_id
                        corrections_made += 1
    
    # Save corrected annotations
    if not dry_run and corrections_made > 0:
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)
        logger.info(f"Saved {corrections_made} corrections to {output_path}")
    
    # Summary
    mismatches = sum(1 for r in results if r.needs_review)
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total images checked: {len(images)}")
    logger.info(f"Total annotations: {len(results)}")
    logger.info(f"Mismatches found: {mismatches}")
    logger.info(f"Corrections made: {corrections_made}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Correct USD dataset labels")
    parser.add_argument("--images", required=True, help="Images directory")
    parser.add_argument("--annotations", required=True, help="COCO annotations JSON")
    parser.add_argument("--output", required=True, help="Output annotations path")
    parser.add_argument("--model", default="usd-classification/1", help="Roboflow model ID")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--apply", action="store_true", help="Apply corrections (default: dry run)")
    
    args = parser.parse_args()
    
    correct_dataset(
        images_dir=args.images,
        annotations_path=args.annotations,
        output_path=args.output,
        model_id=args.model,
        confidence_threshold=args.threshold,
        dry_run=not args.apply,
    )
