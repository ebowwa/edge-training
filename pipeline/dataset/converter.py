"""COCO to YOLO format converter utilities."""

from pathlib import Path
from typing import Tuple, List, Dict, Any
from PIL import Image
from io import BytesIO


def convert_bbox_coco_to_yolo(
    bbox: List[float], 
    img_width: int, 
    img_height: int
) -> Tuple[float, float, float, float]:
    """Convert COCO bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] normalized.
    
    Args:
        bbox: COCO format [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    x, y, w, h = bbox
    
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    # Clip to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    
    return x_center, y_center, w_norm, h_norm


def extract_image_data(item: Dict[str, Any]) -> Tuple[bytes, int, int]:
    """Extract image bytes and dimensions from HuggingFace dataset item.
    
    Args:
        item: Dataset item with 'image' key (PIL Image, bytes, or numpy array)
        
    Returns:
        Tuple of (image_bytes, width, height)
    """
    img = item["image"]
    
    if isinstance(img, Image.Image):
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        return img_bytes.getvalue(), img.size[0], img.size[1]
    
    if isinstance(img, bytes):
        return img, item.get("width", 640), item.get("height", 640)
    
    # numpy array or other
    pil_img = Image.fromarray(img)
    img_bytes = BytesIO()
    pil_img.save(img_bytes, format="JPEG")
    return img_bytes.getvalue(), pil_img.size[0], pil_img.size[1]


def extract_annotations(item: Dict[str, Any]) -> Tuple[List[List[float]], List[int]]:
    """Extract bboxes and category IDs from HuggingFace dataset item.
    
    Handles multiple COCO format variations.
    
    Args:
        item: Dataset item with 'objects', 'bboxes', or similar keys
        
    Returns:
        Tuple of (list of bboxes, list of category_ids)
    """
    if "objects" in item:
        objects = item["objects"]
        if isinstance(objects, dict):
            bboxes = objects.get("bbox", [])
            categories = objects.get("category", [])
        else:
            bboxes = [obj.get("bbox", [0, 0, 0, 0]) for obj in objects]
            categories = [obj.get("category", 0) for obj in objects]
    elif "bboxes" in item:
        bboxes = item["bboxes"]
        categories = item.get("category_id", item.get("categories", []))
    else:
        bboxes = []
        categories = []
    
    return bboxes, categories


def write_yolo_label(
    label_path: Path,
    bboxes: List[List[float]],
    categories: List[int],
    img_width: int,
    img_height: int
) -> int:
    """Write YOLO format label file.
    
    Args:
        label_path: Output .txt file path
        bboxes: List of COCO format bboxes
        categories: List of category IDs
        img_width: Image width
        img_height: Image height
        
    Returns:
        Number of annotations written
    """
    with open(label_path, "w") as f:
        for bbox, category_id in zip(bboxes, categories):
            x_center, y_center, w_norm, h_norm = convert_bbox_coco_to_yolo(
                bbox, img_width, img_height
            )
            f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    return len(bboxes)
