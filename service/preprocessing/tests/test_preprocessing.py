#!/usr/bin/env python3
"""
Tests for preprocessing module.

service/preprocessing/
├── cleaners.py
├── transforms.py
├── pipeline.py
├── config.py
└── tests/
    ├── __init__.py
    └── test_preprocessing.py
"""
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2


from service.preprocessing.cleaners import (
    CorruptedImageCleaner,
    AnnotationValidator,
    BBoxValidator,
    CleanResult,
)
from service.preprocessing.transforms import (
    FlipTransform,
    RotateTransform,
    ColorTransform,
    NoiseTransform,
)
from service.preprocessing.pipeline import PreprocessingPipeline, PreprocessingResult
from service.preprocessing.config import CleaningConfig, AugmentationConfig


class TestFixtures:
    """Create test fixtures for preprocessing tests."""

    @staticmethod
    def create_test_dataset(base_dir: Path, num_images: int = 5):
        """Create a test dataset with images and labels."""
        images_dir = base_dir / "images"
        labels_dir = base_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_images):
            # Create random image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / f"img_{i}.jpg"), img)

            # Create label with valid YOLO format
            with open(labels_dir / f"img_{i}.txt", "w") as f:
                f.write(f"0 0.5 0.5 0.3 0.3\n")

        return images_dir, labels_dir


def test_cleaners():
    """Test cleaner components."""
    print("\n[Test] Cleaners...")

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        images_dir, labels_dir = TestFixtures.create_test_dataset(base)

        # Test CorruptedImageCleaner
        cleaner = CorruptedImageCleaner()
        img_path = images_dir / "img_0.jpg"
        label_path = labels_dir / "img_0.txt"
        result = cleaner.clean(img_path, label_path)
        assert not result.removed, "Valid image should not be removed"
        print("  ✓ CorruptedImageCleaner: valid images pass")

        # Test AnnotationValidator
        validator = AnnotationValidator()
        result = validator.clean(img_path, label_path)
        assert not result.fixed, "Valid labels should not need fixing"
        print("  ✓ AnnotationValidator: valid labels pass")

        # Test with invalid annotation
        with open(labels_dir / "img_1.txt", "w") as f:
            f.write("invalid line\n")
            f.write("0 0.5 0.5 0.3 0.3\n")  # valid
        result = validator.clean(images_dir / "img_1.jpg", labels_dir / "img_1.txt")
        assert result.fixed, "Invalid annotation should be fixed"
        print("  ✓ AnnotationValidator: fixes invalid lines")

        # Test BBoxValidator
        bbox_validator = BBoxValidator(min_size=1, max_ratio=0.9)
        result = bbox_validator.clean(img_path, label_path)
        assert not result.fixed, "Valid bbox should pass"
        print("  ✓ BBoxValidator: valid bboxes pass")

    print("  ✓ All cleaner tests passed!")


def test_transforms():
    """Test transform components."""
    print("\n[Test] Transforms...")

    # Test FlipTransform
    flip = FlipTransform(horizontal_p=0.5, vertical_p=0.1)
    assert flip.name == "flip"
    alb = flip.get_albumentations_transform()
    assert alb is not None
    print("  ✓ FlipTransform: instantiates correctly")

    # Test RotateTransform
    rotate = RotateTransform(limit=15, p=0.3)
    assert rotate.name == "rotate"
    alb = rotate.get_albumentations_transform()
    assert alb is not None
    print("  ✓ RotateTransform: instantiates correctly")

    # Test ColorTransform
    color = ColorTransform(brightness=0.2, contrast=0.2, p=0.3)
    assert color.name == "color"
    alb = color.get_albumentations_transform()
    assert alb is not None
    print("  ✓ ColorTransform: instantiates correctly")

    # Test NoiseTransform
    noise = NoiseTransform(blur_limit=3, blur_p=0.1)
    assert noise.name == "noise"
    alb = noise.get_albumentations_transform()
    assert alb is not None
    print("  ✓ NoiseTransform: instantiates correctly")

    print("  ✓ All transform tests passed!")


def test_pipeline():
    """Test preprocessing pipeline."""
    print("\n[Test] Pipeline...")

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        images_dir, labels_dir = TestFixtures.create_test_dataset(base, num_images=3)
        output_images = base / "output" / "images"
        output_labels = base / "output" / "labels"

        # Create pipeline
        pipeline = PreprocessingPipeline(
            cleaners=[CorruptedImageCleaner(), BBoxValidator()],
            transforms=[FlipTransform(horizontal_p=0.5)],
            augment_factor=2,
            num_workers=1,
        )
        print(f"  ✓ Pipeline created: {len(pipeline.cleaners)} cleaners, {len(pipeline.transforms)} transforms")

        # Test cleaning phase
        clean_result = pipeline.clean(str(images_dir), str(labels_dir))
        assert clean_result.images_processed == 3
        assert clean_result.images_removed == 0
        print(f"  ✓ Cleaning phase: processed {clean_result.images_processed} images")

        # Test augmentation phase
        aug_result = pipeline.augment(
            str(images_dir), str(labels_dir),
            str(output_images), str(output_labels)
        )
        assert aug_result.images_augmented > 0
        print(f"  ✓ Augmentation phase: created {aug_result.images_augmented} images")

        # Verify output files exist
        output_files = list(output_images.glob("*.jpg"))
        assert len(output_files) > 0
        print(f"  ✓ Output files: {len(output_files)} augmented images")

    print("  ✓ All pipeline tests passed!")


def test_config():
    """Test configuration dataclasses."""
    print("\n[Test] Config...")

    cleaning = CleaningConfig()
    assert cleaning.remove_corrupted is True
    assert cleaning.min_bbox_size == 1
    print("  ✓ CleaningConfig: defaults work")

    aug = AugmentationConfig()
    assert aug.augment_factor == 2
    assert aug.num_workers == 1
    print("  ✓ AugmentationConfig: defaults work")

    # Test custom values
    cleaning = CleaningConfig(min_bbox_size=5, max_bbox_ratio=0.8)
    assert cleaning.min_bbox_size == 5
    print("  ✓ CleaningConfig: custom values work")

    print("  ✓ All config tests passed!")


def test_result_addition():
    """Test PreprocessingResult addition."""
    print("\n[Test] Result addition...")

    r1 = PreprocessingResult(images_processed=10, images_removed=2, images_augmented=20)
    r2 = PreprocessingResult(images_processed=5, labels_fixed=3, images_augmented=10)

    combined = r1 + r2
    assert combined.images_processed == 15
    assert combined.images_removed == 2
    assert combined.labels_fixed == 3
    assert combined.images_augmented == 30
    print("  ✓ Results combine correctly")

    print("  ✓ All result tests passed!")


def main():
    print("=" * 60)
    print("Testing preprocessing module...")
    print("=" * 60)

    test_config()
    test_cleaners()
    test_transforms()
    test_result_addition()
    test_pipeline()

    print("\n" + "=" * 60)
    print("All preprocessing tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
