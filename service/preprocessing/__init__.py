"""
Preprocessing Module - Composable Data Cleaning and Augmentation

Example:
    from service.preprocessing import (
        PreprocessingPipeline,
        CorruptedImageCleaner,
        BBoxValidator,
        FlipTransform,
    )

    pipeline = PreprocessingPipeline(
        cleaners=[CorruptedImageCleaner(), BBoxValidator()],
        transforms=[FlipTransform(horizontal_p=0.5)],
        augment_factor=2,
        num_workers=4,
    )
    stats = pipeline.process(images_dir, labels_dir, output_dir)
"""

from .cleaners import (
    BaseCleaner,
    CorruptedImageCleaner,
    AnnotationValidator,
    BBoxValidator,
)

from .quality import (
    BlurDetector,
    ExposureDetector,
    ContrastDetector,
    QualityAssessor,
    QualityMetrics,
)

from .transforms import (
    BaseTransform,
    FlipTransform,
    RotateTransform,
    ColorTransform,
    NoiseTransform,
)

from .pipeline import PreprocessingPipeline, PreprocessingResult
from .config import CleaningConfig, AugmentationConfig

__all__ = [
    "BaseCleaner", "CorruptedImageCleaner", "AnnotationValidator", "BBoxValidator",
    "BlurDetector", "ExposureDetector", "ContrastDetector", "QualityAssessor", "QualityMetrics",
    "BaseTransform", "FlipTransform", "RotateTransform", "ColorTransform", "NoiseTransform",
    "PreprocessingPipeline", "PreprocessingResult",
    "CleaningConfig", "AugmentationConfig",
]
