"""
Spatial Inference Pipeline.

Chains YOLO inference with SLAM for real-time spatial detection.
Designed for egocentric video from wearables (glasses, phones).
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Add service path for imports
_pipeline_dir = Path(__file__).parent
_root_dir = _pipeline_dir.parent
_service_dir = _root_dir / "service"
_slam_dir = _service_dir / "slam"
sys.path.insert(0, str(_pipeline_dir))
sys.path.insert(0, str(_service_dir))
sys.path.insert(0, str(_slam_dir))

# Handle both relative and absolute imports
try:
    from .base import Pipeline, PipelineStage, PipelineContext, FunctionStage
except ImportError:
    from base import Pipeline, PipelineStage, PipelineContext, FunctionStage

# Import services
from inference_service import InferenceService
from config import InferenceConfig, Detection
from slam_service import SlamService, DevicePose, SpatialAnchor


@dataclass
class SpatialDetection:
    """A detection with spatial anchoring."""
    # Original detection
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    bbox_normalized: Tuple[float, float, float, float]  # xc, yc, w, h
    
    # Spatial info
    anchor_id: Optional[int] = None
    spatial_coords: Optional[Tuple[float, float]] = None
    is_new: bool = True  # First time seeing this object


@dataclass 
class SpatialInferenceResult:
    """Result from spatial inference pipeline."""
    # Frame info
    frame_id: int
    timestamp: float
    image_size: Tuple[int, int]
    
    # Device pose
    pose: DevicePose
    
    # Detections with spatial info
    detections: List[SpatialDetection] = field(default_factory=list)
    
    # Performance
    inference_time_ms: float = 0.0
    slam_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Spatial map summary
    total_anchors: int = 0


class InferenceStage(PipelineStage):
    """YOLO inference stage."""
    
    def __init__(self, model_path: str, config: Optional[InferenceConfig] = None):
        self.model_path = model_path
        self.config = config or InferenceConfig()
        self._service: Optional[InferenceService] = None
    
    @property
    def name(self) -> str:
        return "inference"
    
    def _get_service(self) -> InferenceService:
        if self._service is None:
            self._service = InferenceService(self.model_path)
        return self._service
    
    def process(self, frame: np.ndarray, context: PipelineContext) -> List[Detection]:
        svc = self._get_service()
        
        start = time.perf_counter()
        result = svc.infer_frame(frame, self.config)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        context.set("inference_time_ms", elapsed_ms)
        context.set("image_size", result.image_size)
        
        return result.detections


class SlamPoseStage(PipelineStage):
    """Update device pose from frame."""
    
    def __init__(self, slam_service: SlamService):
        self.slam = slam_service
    
    @property
    def name(self) -> str:
        return "slam_pose"
    
    def process(self, frame: np.ndarray, context: PipelineContext) -> DevicePose:
        imu_data = context.get("imu_data")
        
        start = time.perf_counter()
        pose = self.slam.update_pose(frame, imu_data)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        context.set("slam_pose_time_ms", elapsed_ms)
        context.set("pose", pose)
        
        return pose


class SlamAnchorStage(PipelineStage):
    """Anchor detections spatially."""
    
    def __init__(self, slam_service: SlamService):
        self.slam = slam_service
    
    @property
    def name(self) -> str:
        return "slam_anchor"
    
    def process(
        self, 
        detections: List[Detection], 
        context: PipelineContext
    ) -> List[SpatialDetection]:
        pose = context.get("pose", DevicePose(timestamp=0.0))
        
        start = time.perf_counter()
        spatial_detections = []
        
        for det in detections:
            anchor = self.slam.anchor_detection(det, pose)
            
            spatial_det = SpatialDetection(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                bbox_normalized=det.bbox_normalized,
                anchor_id=anchor.id,
                spatial_coords=anchor.relative_coords,
                is_new=True,  # TODO: track if seen before
            )
            spatial_detections.append(spatial_det)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        context.set("slam_anchor_time_ms", elapsed_ms)
        
        return spatial_detections


class SpatialInferencePipeline:
    """
    High-level API for spatial inference.
    
    Chains YOLO inference with SLAM for real-time spatial detection.
    
    Example:
        pipeline = SpatialInferencePipeline(
            model_path="weights/pothole.pt",
            enable_slam=True,
            imu_enabled=True,
        )
        
        for frame in video_stream:
            result = pipeline.process_frame(frame, imu_data=sensor_data)
            for det in result.detections:
                print(f"{det.class_name} at {det.spatial_coords}")
    """
    
    def __init__(
        self,
        model_path: str,
        enable_slam: bool = True,
        imu_enabled: bool = False,
        conf_threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.enable_slam = enable_slam
        
        # Initialize services
        self.inference_config = InferenceConfig(conf_threshold=conf_threshold)
        self._inference_service: Optional[InferenceService] = None
        
        if enable_slam:
            self.slam_service = SlamService({"imu_enabled": imu_enabled})
        else:
            self.slam_service = None
        
        self._frame_count = 0
    
    def _get_inference_service(self) -> InferenceService:
        if self._inference_service is None:
            self._inference_service = InferenceService(self.model_path)
        return self._inference_service
    
    def process_frame(
        self,
        frame: np.ndarray,
        imu_data: Optional[Dict[str, float]] = None,
    ) -> SpatialInferenceResult:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
            imu_data: Optional IMU sensor data for pose estimation
            
        Returns:
            SpatialInferenceResult with detections and spatial info
        """
        start_time = time.perf_counter()
        self._frame_count += 1
        
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Step 1: Update pose (if SLAM enabled)
        pose = DevicePose(timestamp=time.time())
        slam_time_ms = 0.0
        
        if self.enable_slam and self.slam_service:
            slam_start = time.perf_counter()
            pose = self.slam_service.update_pose(frame, imu_data)
            slam_time_ms = (time.perf_counter() - slam_start) * 1000
        
        # Step 2: Run YOLO inference
        inference_start = time.perf_counter()
        inference_svc = self._get_inference_service()
        inference_result = inference_svc.infer_frame(frame, self.inference_config)
        inference_time_ms = (time.perf_counter() - inference_start) * 1000
        
        # Step 3: Anchor detections spatially (if SLAM enabled)
        spatial_detections = []
        
        for det in inference_result.detections:
            if self.enable_slam and self.slam_service:
                anchor = self.slam_service.anchor_detection(det, pose)
                spatial_det = SpatialDetection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    bbox_normalized=det.bbox_normalized,
                    anchor_id=anchor.id,
                    spatial_coords=anchor.relative_coords,
                    is_new=True,
                )
            else:
                spatial_det = SpatialDetection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    bbox_normalized=det.bbox_normalized,
                )
            spatial_detections.append(spatial_det)
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Get total anchors
        total_anchors = 0
        if self.enable_slam and self.slam_service:
            total_anchors = len(self.slam_service.get_active_map())
        
        return SpatialInferenceResult(
            frame_id=self._frame_count,
            timestamp=pose.timestamp,
            image_size=(w, h),
            pose=pose,
            detections=spatial_detections,
            inference_time_ms=inference_time_ms,
            slam_time_ms=slam_time_ms,
            total_time_ms=total_time_ms,
            total_anchors=total_anchors,
        )
    
    def reset(self) -> None:
        """Reset the pipeline state (clear SLAM anchors, etc)."""
        self._frame_count = 0
        if self.enable_slam:
            imu_enabled = self.slam_service.imu_enabled if self.slam_service else False
            self.slam_service = SlamService({"imu_enabled": imu_enabled})
    
    def get_spatial_map(self) -> List[SpatialAnchor]:
        """Get all spatially anchored detections."""
        if self.slam_service:
            return self.slam_service.get_active_map()
        return []


# Composable stage factories for advanced usage
def create_inference_stage(model_path: str, conf_threshold: float = 0.5) -> InferenceStage:
    """Create an inference stage."""
    config = InferenceConfig(conf_threshold=conf_threshold)
    return InferenceStage(model_path, config)


def create_slam_stages(imu_enabled: bool = False) -> Tuple[SlamPoseStage, SlamAnchorStage]:
    """Create SLAM pose and anchor stages sharing a service."""
    slam = SlamService({"imu_enabled": imu_enabled})
    return SlamPoseStage(slam), SlamAnchorStage(slam)
