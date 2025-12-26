"""
Roboflow Provider

Preprocessing via Roboflow inference API (classification, detection, segmentation).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from inference_sdk import InferenceHTTPClient

from integrations.roboflow.config import RoboflowConfig

logger = logging.getLogger(__name__)


@dataclass
class RoboflowResult:
    """Result from Roboflow inference."""
    predictions: List[Dict[str, Any]]
    image_path: str
    model_id: str
    time_ms: float = 0.0


class RoboflowProvider:
    """
    Roboflow inference provider for preprocessing.
    
    Example:
        config = RoboflowConfig(model_id="usd-classification/1")
        provider = RoboflowProvider(config)
        
        # Single image
        result = provider.infer("image.jpg")
        
        # Batch
        results = provider.infer_batch(["img1.jpg", "img2.jpg"])
    """

    def __init__(self, config: RoboflowConfig):
        self.config = config
        self.client = InferenceHTTPClient(
            api_url=config.api_url,
            api_key=config.api_key,
        )
        logger.info(f"Roboflow initialized: {config.api_url}")

    def infer(
        self,
        image: Union[str, Path, bytes],
        model_id: Optional[str] = None,
    ) -> RoboflowResult:
        """
        Run inference on a single image.
        
        Args:
            image: Path to image, or raw bytes
            model_id: Override default model_id
        """
        model = model_id or self.config.model_id
        if not model:
            raise ValueError("model_id required")

        image_path = str(image) if isinstance(image, (str, Path)) else "<bytes>"
        
        result = self.client.infer(image, model_id=model)
        
        predictions = result.get("predictions", [])
        inference_time = result.get("time", 0.0) * 1000  # Convert to ms
        
        logger.info(f"Roboflow: {len(predictions)} predictions for {image_path}")
        
        return RoboflowResult(
            predictions=predictions,
            image_path=image_path,
            model_id=model,
            time_ms=inference_time,
        )

    def infer_batch(
        self,
        images: List[Union[str, Path]],
        model_id: Optional[str] = None,
    ) -> List[RoboflowResult]:
        """Run inference on multiple images."""
        return [self.infer(img, model_id) for img in images]

    def classify(self, image: Union[str, Path, bytes], model_id: Optional[str] = None) -> Dict[str, float]:
        """
        Classify an image and return class probabilities.
        
        Returns:
            Dict mapping class names to confidence scores
        """
        result = self.infer(image, model_id)
        return {
            p.get("class", "unknown"): p.get("confidence", 0.0)
            for p in result.predictions
        }

    def detect(self, image: Union[str, Path, bytes], model_id: Optional[str] = None) -> List[Dict]:
        """
        Detect objects in an image.
        
        Returns:
            List of detections with class, confidence, bbox
        """
        result = self.infer(image, model_id)
        return [
            {
                "class": p.get("class"),
                "confidence": p.get("confidence"),
                "bbox": [p.get("x"), p.get("y"), p.get("width"), p.get("height")],
            }
            for p in result.predictions
        ]
