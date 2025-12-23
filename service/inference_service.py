"""
Inference Service for YOLO.
Handles inference on images, videos, and raw frames.
"""

import logging
import os
import time
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .config import Detection, InferenceConfig, InferenceResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class InferenceService:
    """Service for model inference operations."""

    def __init__(self, model_path: str):
        """
        Initialize inference service with a trained model.

        Args:
            model_path: Path to trained model weights.

        Raises:
            FileNotFoundError: If model not found.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = YOLO(model_path)
        self.model_path = model_path
        logging.info(f"Model loaded from {model_path}")

    def _parse_results(
        self,
        results,
        image_size: tuple,
        include_annotated: bool = False
    ) -> InferenceResult:
        """
        Parse YOLO results into structured format.

        Args:
            results: Raw YOLO results.
            image_size: (width, height) of input image.
            include_annotated: Whether to include annotated image.

        Returns:
            InferenceResult with parsed detections.
        """
        detections = []
        result = results[0]

        if result.boxes is not None:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Get box coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Calculate normalized coordinates
                w, h = image_size
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = self.model.names.get(class_id, str(class_id))

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    bbox_normalized=[x_center, y_center, width, height]
                ))

        annotated = result.plot() if include_annotated else None

        return InferenceResult(
            detections=detections,
            inference_time_ms=result.speed.get('inference', 0.0),
            image_size=image_size,
            annotated_image=annotated
        )

    def infer_frame(
        self,
        frame: np.ndarray,
        config: Optional[InferenceConfig] = None
    ) -> InferenceResult:
        """
        Perform inference on a single frame (numpy array).

        Args:
            frame: Input image as numpy array (BGR format).
            config: Inference configuration.

        Returns:
            InferenceResult with detections.
        """
        if config is None:
            config = InferenceConfig()

        height, width = frame.shape[:2]
        results = self.model(
            frame,
            conf=config.conf_threshold,
            iou=config.iou_threshold
        )

        return self._parse_results(
            results,
            (width, height),
            include_annotated=config.save_output
        )

    def infer_image(
        self,
        image_path: str,
        config: Optional[InferenceConfig] = None
    ) -> InferenceResult:
        """
        Perform inference on a single image file.

        Args:
            image_path: Path to input image.
            config: Inference configuration.

        Returns:
            InferenceResult with detections.

        Raises:
            FileNotFoundError: If image not found.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if config is None:
            config = InferenceConfig()

        img = cv2.imread(image_path)
        result = self.infer_frame(img, config)

        if config.save_output and config.output_path:
            if result.annotated_image is not None:
                cv2.imwrite(config.output_path, result.annotated_image)
                logging.info(f"Annotated image saved to {config.output_path}")

        return result

    def infer_video(
        self,
        video_path: str,
        config: Optional[InferenceConfig] = None,
        callback=None
    ) -> List[InferenceResult]:
        """
        Perform inference on a video file.

        Args:
            video_path: Path to input video.
            config: Inference configuration.
            callback: Optional callback(frame_idx, result) for each frame.

        Returns:
            List of InferenceResult for each frame.

        Raises:
            FileNotFoundError: If video not found.
            ValueError: If video cannot be opened.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        if config is None:
            config = InferenceConfig()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Setup video writer if saving
        out = None
        if config.save_output and config.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(config.output_path, fourcc, fps, (width, height))

        results = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # For video, we need annotated frames if saving
            frame_config = InferenceConfig(
                conf_threshold=config.conf_threshold,
                iou_threshold=config.iou_threshold,
                save_output=config.save_output
            )
            result = self.infer_frame(frame, frame_config)
            results.append(result)

            if out is not None and result.annotated_image is not None:
                out.write(result.annotated_image)

            if callback:
                callback(frame_count, result)

            frame_count += 1
            if frame_count % 100 == 0:
                logging.info(f"Processed {frame_count} frames")

        cap.release()
        if out is not None:
            out.release()
            logging.info(f"Annotated video saved to {config.output_path}")

        return results

    def infer_batch(
        self,
        images: List[np.ndarray],
        config: Optional[InferenceConfig] = None
    ) -> List[InferenceResult]:
        """
        Perform inference on a batch of images.

        Args:
            images: List of input images as numpy arrays.
            config: Inference configuration.

        Returns:
            List of InferenceResult for each image.
        """
        if config is None:
            config = InferenceConfig()

        results = []
        for img in images:
            result = self.infer_frame(img, config)
            results.append(result)

        return results
