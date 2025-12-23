"""
Composable Pipeline Module.

Provides chainable, composable pipelines for real-time inference and spatial detection.

Example:
    from pipeline import SpatialInferencePipeline
    
    pipeline = SpatialInferencePipeline(
        model_path="model.pt",
        enable_slam=True,
    )
    
    result = pipeline.process_frame(frame, imu_data)
    # result.detections - YOLO detections
    # result.anchors - Spatially anchored detections
    # result.pose - Current device pose
"""

from .base import (
    PipelineStage,
    Pipeline,
    PipelineContext,
    FunctionStage,
    ConditionalStage,
    ParallelStage,
    ComposedStage,
    PipelineRegistry,
    AttachablePipeline,
)
from .spatial_inference import SpatialInferencePipeline, SpatialInferenceResult

__all__ = [
    # Base abstractions
    "PipelineStage",
    "Pipeline",
    "PipelineContext",
    "FunctionStage",
    "ConditionalStage",
    "ParallelStage",
    "ComposedStage",
    # Attachable/Registry
    "PipelineRegistry",
    "AttachablePipeline",
    # Spatial inference
    "SpatialInferencePipeline",
    "SpatialInferenceResult",
]
