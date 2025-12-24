"""
YOLO Training Service - Composable Service Layer

This package provides a clean, composable service layer for YOLO model
training, inference, validation, and export operations.

Example usage:
    from com.service import (
        DatasetService, TrainingService, InferenceService,
        DatasetConfig, TrainingConfig, InferenceConfig
    )

    # Prepare dataset
    config = DatasetConfig(
        dataset_handle='jocelyndumlao/multi-weather-pothole-detection-mwpd',
        nc=1,
        names=['Potholes']
    )
    yaml_path = DatasetService.prepare(config)

    # Train model
    train_config = TrainingConfig(epochs=60, imgsz=512)
    result = TrainingService.train(yaml_path, train_config)

    # Run inference
    inference = InferenceService(result.best_model_path)
    detections = inference.infer_image('test.jpg')
"""

__version__ = "0.1.0"

# Configuration dataclasses
from .config import (
    ModelRegistry,
    DatasetConfig,
    Detection,
    ExportConfig,
    InferenceConfig,
    InferenceResult,
    PreprocessingConfig,
    TrainingConfig,
    TrainingResult,
    ValidationConfig,
    ValidationResult,
)

# Service classes
from integrations.kaggle.dataset import DatasetService
from .export_service import ExportService
from .inference_service import InferenceService
from .yolo_training import TrainingService
from .yolo.validation import ValidationService
from .slam import SlamService, DevicePose, SpatialAnchor
from .optimization import InferenceOptimizer, optimize_for_inference
from .callbacks import (
    TrainingCallback,
    CallbackManager,
    TensorBoardCallback,
    WandBCallback,
    EarlyStoppingCallback,
    MetricsLoggerCallback,
    create_default_callbacks,
)
from .rfdetr_service import RFDETRService, RFDETRResult, RFDETRDetection

__all__ = [
    # Version
    "__version__",
    # Configs
    "ModelRegistry",
    "DatasetConfig",
    "TrainingConfig",
    "InferenceConfig",
    "ValidationConfig",
    "ExportConfig",
    "PreprocessingConfig",
    # Results
    "Detection",
    "InferenceResult",
    "TrainingResult",
    "ValidationResult",
    # SLAM Types
    "DevicePose",
    "SpatialAnchor",
    # Services
    "DatasetService",
    "TrainingService",
    "InferenceService",
    "ValidationService",
    "ExportService",
    "SlamService",
    # Optimization
    "InferenceOptimizer",
    "optimize_for_inference",
    # Callbacks
    "TrainingCallback",
    "CallbackManager",
    "TensorBoardCallback",
    "WandBCallback",
    "EarlyStoppingCallback",
    "MetricsLoggerCallback",
    "create_default_callbacks",
    # RF-DETR Backend
    "RFDETRService",
    "RFDETRResult",
    "RFDETRDetection",
]
