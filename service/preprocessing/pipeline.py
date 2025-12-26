"""
Preprocessing Pipeline - Orchestrates cleaners and transforms.

Supports:
- Composable cleaners and transforms
- Parallel processing with num_workers
- Progress callbacks
- Detailed statistics
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

import cv2
import albumentations as A

from service.preprocessing.cleaners import BaseCleaner, CleanResult
from service.preprocessing.transforms import BaseTransform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class PreprocessingResult:
    """Result from preprocessing pipeline."""
    images_processed: int = 0
    images_removed: int = 0
    labels_fixed: int = 0
    images_augmented: int = 0
    errors: List[str] = field(default_factory=list)

    def __add__(self, other: 'PreprocessingResult') -> 'PreprocessingResult':
        return PreprocessingResult(
            images_processed=self.images_processed + other.images_processed,
            images_removed=self.images_removed + other.images_removed,
            labels_fixed=self.labels_fixed + other.labels_fixed,
            images_augmented=self.images_augmented + other.images_augmented,
            errors=self.errors + other.errors,
        )


class PreprocessingPipeline:
    """
    Composable preprocessing pipeline for YOLO datasets.

    Example:
        pipeline = PreprocessingPipeline(
            cleaners=[CorruptedImageCleaner(), BBoxValidator()],
            transforms=[FlipTransform(), RotateTransform()],
            augment_factor=3,
            num_workers=4,
        )
        result = pipeline.process(images_dir, labels_dir, output_dir)
    """

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(
        self,
        cleaners: Optional[List[BaseCleaner]] = None,
        transforms: Optional[List[BaseTransform]] = None,
        augment_factor: int = 2,
        num_workers: int = 1,
    ):
        """
        Initialize pipeline.

        Args:
            cleaners: List of cleaners to apply (in order).
            transforms: List of transforms to compose.
            augment_factor: Number of augmented versions per image.
            num_workers: Number of parallel workers for processing.
        """
        self.cleaners = cleaners or []
        self.transforms = transforms or []
        self.augment_factor = augment_factor
        self.num_workers = num_workers
        self._augmentation_pipeline = self._build_augmentation_pipeline()

    def _build_augmentation_pipeline(self) -> A.Compose:
        """Build Albumentations pipeline from transforms."""
        if not self.transforms:
            return None

        alb_transforms = []
        for t in self.transforms:
            alb_t = t.get_albumentations_transform()
            if isinstance(alb_t, A.Compose):
                alb_transforms.extend(alb_t.transforms)
            else:
                alb_transforms.append(alb_t)

        return A.Compose(
            alb_transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    def _get_image_files(self, images_dir: Path) -> List[Path]:
        """Get all image files in directory."""
        files = []
        for ext in self.IMAGE_EXTENSIONS:
            files.extend(images_dir.glob(f'*{ext}'))
            files.extend(images_dir.glob(f'*{ext.upper()}'))
        return files

    def _get_label_path(self, image_path: Path, labels_dir: Path) -> Optional[Path]:
        """Get corresponding label path for an image."""
        label_path = labels_dir / f"{image_path.stem}.txt"
        return label_path if label_path.exists() else None

    def _clean_single(self, image_path: Path, label_path: Optional[Path]) -> CleanResult:
        """Apply all cleaners to a single image/label pair."""
        result = CleanResult(str(image_path), str(label_path) if label_path else None)

        for cleaner in self.cleaners:
            clean_result = cleaner.clean(image_path, label_path)
            if clean_result.removed:
                result.removed = True
                return result  # Image removed, stop processing
            if clean_result.fixed:
                result.fixed = True
            if clean_result.error:
                result.error = clean_result.error

        return result

    def _augment_single(
        self,
        image_path: Path,
        label_path: Optional[Path],
        output_images_dir: Path,
        output_labels_dir: Path,
    ) -> int:
        """Augment a single image and save results."""
        if not self._augmentation_pipeline:
            return 0

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return 0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read labels
        bboxes = []
        class_labels = []
        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:]])

        count = 0
        for i in range(self.augment_factor):
            try:
                transformed = self._augmentation_pipeline(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                # Save augmented image
                aug_name = f"{image_path.stem}_aug_{i}{image_path.suffix}"
                aug_img_path = output_images_dir / aug_name
                aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_img_path), aug_img)

                # Save augmented labels
                aug_label_name = f"{image_path.stem}_aug_{i}.txt"
                aug_label_path = output_labels_dir / aug_label_name
                with open(aug_label_path, 'w') as f:
                    for cls, bbox in zip(transformed['class_labels'], transformed['bboxes']):
                        f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")

                count += 1
            except Exception as e:
                logging.warning(f"Augmentation failed for {image_path}: {e}")

        return count

    def clean(self, images_dir: str, labels_dir: str) -> PreprocessingResult:
        """
        Run cleaning phase only.

        Args:
            images_dir: Directory containing images.
            labels_dir: Directory containing labels.

        Returns:
            PreprocessingResult with cleaning stats.
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        image_files = self._get_image_files(images_path)

        result = PreprocessingResult()

        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        self._clean_single,
                        img,
                        self._get_label_path(img, labels_path)
                    ): img for img in image_files
                }
                for future in as_completed(futures):
                    res = future.result()
                    result.images_processed += 1
                    if res.removed:
                        result.images_removed += 1
                    if res.fixed:
                        result.labels_fixed += 1
                    if res.error:
                        result.errors.append(res.error)
        else:
            for img in image_files:
                res = self._clean_single(img, self._get_label_path(img, labels_path))
                result.images_processed += 1
                if res.removed:
                    result.images_removed += 1
                if res.fixed:
                    result.labels_fixed += 1
                if res.error:
                    result.errors.append(res.error)

        logging.info(f"Cleaning done: {result.images_removed} removed, {result.labels_fixed} fixed")
        return result

    def augment(
        self,
        images_dir: str,
        labels_dir: str,
        output_images_dir: Optional[str] = None,
        output_labels_dir: Optional[str] = None,
    ) -> PreprocessingResult:
        """
        Run augmentation phase only.

        Args:
            images_dir: Source images directory.
            labels_dir: Source labels directory.
            output_images_dir: Output for augmented images (default: same as input).
            output_labels_dir: Output for augmented labels (default: same as input).

        Returns:
            PreprocessingResult with augmentation stats.
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        out_images = Path(output_images_dir) if output_images_dir else images_path
        out_labels = Path(output_labels_dir) if output_labels_dir else labels_path

        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        image_files = self._get_image_files(images_path)
        result = PreprocessingResult()

        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        self._augment_single,
                        img,
                        self._get_label_path(img, labels_path),
                        out_images,
                        out_labels
                    ): img for img in image_files
                }
                for future in as_completed(futures):
                    result.images_augmented += future.result()
                    result.images_processed += 1
        else:
            for img in image_files:
                result.images_augmented += self._augment_single(
                    img,
                    self._get_label_path(img, labels_path),
                    out_images,
                    out_labels
                )
                result.images_processed += 1

        logging.info(f"Augmentation done: {result.images_augmented} images created")
        return result

    def process(
        self,
        images_dir: str,
        labels_dir: str,
        output_images_dir: Optional[str] = None,
        output_labels_dir: Optional[str] = None,
    ) -> PreprocessingResult:
        """
        Run full pipeline: cleaning + augmentation.

        Args:
            images_dir: Source images directory.
            labels_dir: Source labels directory.
            output_images_dir: Output for augmented images.
            output_labels_dir: Output for augmented labels.

        Returns:
            Combined PreprocessingResult.
        """
        logging.info("Starting preprocessing pipeline...")

        # Phase 1: Cleaning
        clean_result = self.clean(images_dir, labels_dir)

        # Phase 2: Augmentation
        aug_result = self.augment(
            images_dir, labels_dir,
            output_images_dir, output_labels_dir
        )

        return clean_result + aug_result
