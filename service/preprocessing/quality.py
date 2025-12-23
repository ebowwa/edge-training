"""
Image Quality Assessment - Composable quality filters for preprocessing.

This module helps you detect and filter low-quality images before they pollute
your training dataset or waste inference compute on a live stream.

Why does this matter?
---------------------
If you train YOLO on blurry, dark, or washed-out images, the model learns to
associate those artifacts with your target classes. That's bad. Similarly, if
you're running inference on a live stream from glasses or a phone, there's no
point wasting GPU cycles on a frame where the user's hand covered the camera.

What does this module detect?
-----------------------------
- **Blur**: Motion blur from shaky cameras, or focus issues.
- **Brightness**: Too dark (underexposed) or too bright (overexposed).
- **Contrast**: Flat lighting, fog, or haze where everything looks the same.
- **Noise**: High ISO artifacts from low-light phone cameras.

How to use it
-------------
There are two main patterns:

1. **Dataset Cleaning** (delete bad images before training):

    ```python
    from service.preprocessing import BlurDetector, ExposureDetector, PreprocessingPipeline

    # Create a pipeline that removes blurry and poorly-lit images
    pipeline = PreprocessingPipeline(
        cleaners=[
            BlurDetector(threshold=80),      # Lower = more tolerant of blur
            ExposureDetector(min_brightness=50, max_brightness=200),
        ],
        transforms=[],  # No augmentation, just cleaning
    )

    # Run it on your dataset folder
    stats = pipeline.clean("dataset/images/", "dataset/labels/")
    print(f"Removed {stats.images_removed} low-quality images")
    ```

2. **Live Stream Gating** (skip bad frames, don't waste inference):

    ```python
    import cv2
    from service.preprocessing.quality import QualityAssessor

    # Set up the assessor with your thresholds
    assessor = QualityAssessor(
        blur_threshold=100,
        min_brightness=40,
        max_brightness=220,
    )

    cap = cv2.VideoCapture(0)  # Or glasses/phone stream
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check quality BEFORE running YOLO (saves compute)
        metrics = assessor.assess(frame)
        
        if metrics.is_acceptable:
            # Frame is good, run your expensive inference
            detections = yolo_model.predict(frame)
        else:
            # Frame is garbage, skip it and wait for the next one
            print(f"Skipping frame: blur={metrics.blur_score:.1f}, bright={metrics.brightness:.1f}")
            continue
    ```

Individual Cleaners
-------------------
Each quality filter is a separate class that inherits from `BaseCleaner`, so you
can mix and match them in a pipeline. They are:

- `BlurDetector`: Uses Laplacian variance to detect sharpness.
- `ExposureDetector`: Uses mean pixel brightness to detect dark/bright issues.
- `ContrastDetector`: Uses standard deviation to detect flat lighting.

All cleaners have a `remove` flag. Set it to `False` if you just want to log
issues without deleting files (useful for auditing a dataset).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .cleaners import BaseCleaner, CleanResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class QualityMetrics:
    """Quality assessment metrics for a single image."""
    blur_score: float       # Laplacian variance (higher = sharper)
    brightness: float       # Mean pixel value (0-255)
    contrast: float         # Std dev of pixel values
    noise_score: float      # Estimated noise level
    is_acceptable: bool     # Overall quality pass/fail


class BlurDetector(BaseCleaner):
    """
    Detect and filter blurry images using Laplacian variance.
    
    The Laplacian operator calculates the second derivative of the image.
    Sharp edges produce high variance; blur produces low variance.
    
    Threshold Guide:
        - < 50:   Very blurry (motion blur, completely out of focus)
        - 50-100: Slightly blurry (may be acceptable for training)
        - > 100:  Sharp image
    """

    def __init__(self, threshold: float = 100.0, remove: bool = True):
        """
        Args:
            threshold: Minimum acceptable blur score (Laplacian variance).
            remove: If True, delete blurry images. If False, just log.
        """
        self.threshold = threshold
        self.remove = remove

    def calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate blur score (Laplacian variance)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def clean(self, image_path: Path, label_path: Optional[Path]) -> CleanResult:
        result = CleanResult(str(image_path), str(label_path) if label_path else None)
        
        img = cv2.imread(str(image_path))
        if img is None:
            return result
        
        blur_score = self.calculate_blur_score(img)
        
        if blur_score < self.threshold:
            logging.info(f"Blur detected: {image_path.name} (score: {blur_score:.2f} < {self.threshold})")
            if self.remove:
                os.remove(image_path)
                if label_path and label_path.exists():
                    os.remove(label_path)
                result.removed = True
        
        return result


class ExposureDetector(BaseCleaner):
    """
    Detect underexposed (dark) or overexposed (washed out) images.
    
    Uses the mean pixel brightness on grayscale conversion.
    
    Threshold Guide:
        - < 40:   Underexposed (too dark, details lost in shadows)
        - 40-220: Acceptable range
        - > 220:  Overexposed (too bright, details lost in highlights)
    """

    def __init__(
        self, 
        min_brightness: float = 40.0, 
        max_brightness: float = 220.0,
        remove: bool = True
    ):
        """
        Args:
            min_brightness: Minimum acceptable mean pixel value.
            max_brightness: Maximum acceptable mean pixel value.
            remove: If True, delete bad images. If False, just log.
        """
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.remove = remove

    def calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate mean brightness (0-255)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(np.mean(gray))

    def clean(self, image_path: Path, label_path: Optional[Path]) -> CleanResult:
        result = CleanResult(str(image_path), str(label_path) if label_path else None)
        
        img = cv2.imread(str(image_path))
        if img is None:
            return result
        
        brightness = self.calculate_brightness(img)
        
        if brightness < self.min_brightness:
            logging.info(f"Underexposed: {image_path.name} (brightness: {brightness:.2f})")
            if self.remove:
                os.remove(image_path)
                if label_path and label_path.exists():
                    os.remove(label_path)
                result.removed = True
        elif brightness > self.max_brightness:
            logging.info(f"Overexposed: {image_path.name} (brightness: {brightness:.2f})")
            if self.remove:
                os.remove(image_path)
                if label_path and label_path.exists():
                    os.remove(label_path)
                result.removed = True
        
        return result


class ContrastDetector(BaseCleaner):
    """
    Detect low-contrast images (flat lighting, fog, haze).
    
    Uses standard deviation of pixel values. Low std dev means 
    all pixels are similar in value (flat, washed out).
    
    Threshold Guide:
        - < 15:  Very low contrast (foggy, hazy, flat)
        - 15-30: Moderate contrast
        - > 30:  Good contrast
    """

    def __init__(self, min_contrast: float = 20.0, remove: bool = True):
        """
        Args:
            min_contrast: Minimum acceptable standard deviation.
            remove: If True, delete low-contrast images.
        """
        self.min_contrast = min_contrast
        self.remove = remove

    def calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate contrast (std dev of pixel values)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(np.std(gray))

    def clean(self, image_path: Path, label_path: Optional[Path]) -> CleanResult:
        result = CleanResult(str(image_path), str(label_path) if label_path else None)
        
        img = cv2.imread(str(image_path))
        if img is None:
            return result
        
        contrast = self.calculate_contrast(img)
        
        if contrast < self.min_contrast:
            logging.info(f"Low contrast: {image_path.name} (std: {contrast:.2f})")
            if self.remove:
                os.remove(image_path)
                if label_path and label_path.exists():
                    os.remove(label_path)
                result.removed = True
        
        return result


class QualityAssessor:
    """
    Comprehensive image quality assessment.
    
    Use this for real-time streams where you want to SKIP bad frames
    rather than DELETE them. Returns metrics without modifying files.
    
    Example:
        assessor = QualityAssessor()
        metrics = assessor.assess(frame)
        if metrics.is_acceptable:
            run_yolo_inference(frame)
        else:
            skip_frame()
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_contrast: float = 20.0,
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast

    def assess(self, image: np.ndarray) -> QualityMetrics:
        """
        Assess image quality without modifying files.
        
        Args:
            image: BGR image (numpy array from cv2.imread or video capture).
            
        Returns:
            QualityMetrics with all scores and pass/fail status.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # Noise estimation using median absolute deviation
        median = np.median(gray)
        noise_score = float(np.median(np.abs(gray.astype(np.float32) - median)))
        
        is_acceptable = (
            blur_score >= self.blur_threshold and
            self.min_brightness <= brightness <= self.max_brightness and
            contrast >= self.min_contrast
        )
        
        return QualityMetrics(
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            noise_score=noise_score,
            is_acceptable=is_acceptable
        )

    def assess_from_path(self, image_path: str) -> Optional[QualityMetrics]:
        """Assess quality from a file path."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        return self.assess(img)
