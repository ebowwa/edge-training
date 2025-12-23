"""
FastAPI routes for YOLO Training API.
"""

import sys
import os
import tempfile
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks

# Service imports
from service import config as service_config
from service import dataset_service, training_service, inference_service, validation_service, export_service

# Preprocessing imports
from service.preprocessing import pipeline as preprocessing_pipeline
from service.preprocessing import cleaners, transforms

# SLAM imports
from service.slam import SlamService, DevicePose, SpatialAnchor

from .schemas import (
    DatasetRequest, DatasetResponse,
    TrainingRequest, TrainingResponse,
    InferenceRequest, InferenceResponse, DetectionResult,
    ValidationRequest, ValidationResponse,
    ExportRequest, ExportResponse,
    PreprocessingRequest, PreprocessingResponse,
    SlamPoseRequest, DevicePoseResponse,
    AnchorDetectionRequest, SpatialAnchorResponse,
    SlamMapResponse, SlamConfigRequest, SlamStatusResponse,
    ImuData, HealthResponse,
)

router = APIRouter()


# === Health ===

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health."""
    return HealthResponse()


# === Dataset ===
# Kaggle now, need new import methods huggingface, gitlfs, etc
@router.post("/datasets/prepare", response_model=DatasetResponse, tags=["Dataset"])
async def prepare_dataset(request: DatasetRequest):
    """Download and prepare a Kaggle dataset."""
    try:
        config = service_config.DatasetConfig(
            dataset_handle=request.dataset_handle,
            nc=request.nc,
            names=request.names,
        )
        path = dataset_service.DatasetService.download(config)
        paths, path = dataset_service.DatasetService.detect_structure(path)
        
        if not paths:
            raise HTTPException(400, "No valid dataset structure found")
        
        yaml_path = dataset_service.DatasetService.create_yaml(
            path, paths, config.nc, config.names
        )
        
        return DatasetResponse(
            yaml_path=yaml_path,
            dataset_path=path,
            splits={k.replace('_images', ''): v for k, v in paths.items() if '_images' in k}
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# === Training ===

# Store for background training jobs
_training_jobs = {}

@router.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest):
    """Train a YOLO model (synchronous)."""
    try:
        config = service_config.TrainingConfig(
            epochs=request.epochs,
            imgsz=request.imgsz,
            batch=request.batch,
            device=request.device,
            project=request.project,
            name=request.name,
            weights=request.weights,
            base_model=request.base_model,
        )
        result = training_service.TrainingService.train(request.yaml_path, config)
        
        return TrainingResponse(
            best_model_path=result.best_model_path,
            last_model_path=result.last_model_path,
            epochs_completed=result.epochs_completed,
            metrics=result.metrics,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/train/resume", response_model=TrainingResponse, tags=["Training"])
async def resume_training(project: str = "runs/train", name: str = "yolo_train"):
    """Resume training from last checkpoint."""
    try:
        result = training_service.TrainingService.resume(project, name)
        return TrainingResponse(
            best_model_path=result.best_model_path,
            last_model_path=result.last_model_path,
            epochs_completed=result.epochs_completed,
            metrics=result.metrics,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# === Inference ===

# Cached inference services
_inference_cache = {}

def _get_inference_service(model_path: str):
    """Get or create cached inference service."""
    if model_path not in _inference_cache:
        _inference_cache[model_path] = inference_service.InferenceService(model_path)
    return _inference_cache[model_path]


@router.post("/infer/image", response_model=InferenceResponse, tags=["Inference"])
async def infer_image(
    model_path: str,
    image: UploadFile = File(...),
    conf_threshold: float = 0.5,
):
    """Run inference on an uploaded image."""
    try:
        svc = _get_inference_service(model_path)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            config = service_config.InferenceConfig(conf_threshold=conf_threshold)
            result = svc.infer_image(tmp_path, config)
        finally:
            os.unlink(tmp_path)
        
        return InferenceResponse(
            detections=[
                DetectionResult(
                    class_id=d.class_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox=d.bbox,
                    bbox_normalized=d.bbox_normalized,
                )
                for d in result.detections
            ],
            inference_time_ms=result.inference_time_ms,
            image_size=list(result.image_size),
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# === Validation ===

@router.post("/validate", response_model=ValidationResponse, tags=["Validation"])
async def validate_model(request: ValidationRequest):
    """Validate a trained model."""
    try:
        config = service_config.ValidationConfig(imgsz=request.imgsz, split=request.split)
        result = validation_service.ValidationService.validate(
            request.model_path, request.yaml_path, config
        )
        
        return ValidationResponse(
            map50=result.map50,
            map50_95=result.map50_95,
            precision=result.precision,
            recall=result.recall,
            metrics=result.metrics,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# === Export ===

@router.post("/export", response_model=ExportResponse, tags=["Export"])
async def export_model(request: ExportRequest):
    """Export a trained model to deployment format."""
    try:
        config = service_config.ExportConfig(
            format=request.format,
            imgsz=request.imgsz,
            half=request.half,
        )
        export_path = export_service.ExportService.export(request.model_path, config)
        
        return ExportResponse(
            export_path=export_path,
            format=request.format,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# === Preprocessing ===

@router.post("/preprocess", response_model=PreprocessingResponse, tags=["Preprocessing"])
async def preprocess_dataset(request: PreprocessingRequest):
    """Run preprocessing (cleaning + augmentation) on a dataset."""
    try:
        cleaner_list = []
        transform_list = []
        
        if request.clean:
            cleaner_list = [cleaners.CorruptedImageCleaner(), cleaners.BBoxValidator()]
        
        if request.augment:
            transform_list = [
                transforms.FlipTransform(horizontal_p=0.5),
                transforms.RotateTransform(limit=15, p=0.3),
                transforms.ColorTransform(p=0.3),
            ]
        
        pipeline = preprocessing_pipeline.PreprocessingPipeline(
            cleaners=cleaner_list,
            transforms=transform_list,
            augment_factor=request.augment_factor,
            num_workers=request.num_workers,
        )
        
        result = pipeline.process(
            request.images_dir,
            request.labels_dir,
            request.output_images_dir,
            request.output_labels_dir,
        )
        
        return PreprocessingResponse(
            images_processed=result.images_processed,
            images_removed=result.images_removed,
            labels_fixed=result.labels_fixed,
            images_augmented=result.images_augmented,
            errors=result.errors,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# === SLAM ===

# Global SLAM service instance
_slam_service: SlamService = None


def _get_slam_service(config: dict = None) -> SlamService:
    """Get or create the SLAM service instance."""
    global _slam_service
    if _slam_service is None:
        _slam_service = SlamService(config or {})
    return _slam_service


@router.post("/slam/init", response_model=SlamStatusResponse, tags=["SLAM"])
async def initialize_slam(request: SlamConfigRequest):
    """Initialize the SLAM service with configuration."""
    global _slam_service
    try:
        config = {"imu_enabled": request.imu_enabled}
        _slam_service = SlamService(config)
        return SlamStatusResponse(
            initialized=True,
            imu_enabled=request.imu_enabled,
            active_anchors=0,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/slam/status", response_model=SlamStatusResponse, tags=["SLAM"])
async def get_slam_status():
    """Get current SLAM service status."""
    svc = _get_slam_service()
    return SlamStatusResponse(
        initialized=True,
        imu_enabled=svc.imu_enabled,
        active_anchors=len(svc.active_anchors),
    )


@router.post("/slam/pose", response_model=DevicePoseResponse, tags=["SLAM"])
async def update_pose(
    frame: UploadFile = File(...),
    request: SlamPoseRequest = None,
):
    """Update device pose from frame (and optional IMU data)."""
    import numpy as np
    from PIL import Image
    import io
    
    try:
        svc = _get_slam_service()
        
        # Read frame
        content = await frame.read()
        image = Image.open(io.BytesIO(content))
        frame_array = np.array(image)
        
        # Convert IMU data if provided
        imu_dict = None
        if request and request.imu_data:
            imu_dict = {
                "accel_x": request.imu_data.accel_x,
                "accel_y": request.imu_data.accel_y,
                "accel_z": request.imu_data.accel_z,
                "gyro_x": request.imu_data.gyro_x,
                "gyro_y": request.imu_data.gyro_y,
                "gyro_z": request.imu_data.gyro_z,
            }
        
        pose = svc.update_pose(frame_array, imu_dict)
        
        return DevicePoseResponse(
            timestamp=pose.timestamp,
            delta_x=pose.delta_x,
            delta_y=pose.delta_y,
            rotation_deg=pose.rotation_deg,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/slam/anchor", response_model=SpatialAnchorResponse, tags=["SLAM"])
async def create_anchor(request: AnchorDetectionRequest):
    """Create a spatial anchor from a detection."""
    try:
        svc = _get_slam_service()
        
        # Create a mock detection object for the service
        class MockDetection:
            def __init__(self, class_name, confidence, bbox):
                self.class_name = class_name
                self.confidence = confidence
                self.bbox = bbox
        
        detection = MockDetection(
            request.class_name,
            request.confidence,
            request.bbox,
        )
        
        # Use current pose (or default)
        pose = DevicePose(timestamp=0.0)
        anchor = svc.anchor_detection(detection, pose)
        
        return SpatialAnchorResponse(
            id=anchor.id,
            label=anchor.label,
            relative_coords=list(anchor.relative_coords),
            confidence=anchor.confidence,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/slam/map", response_model=SlamMapResponse, tags=["SLAM"])
async def get_spatial_map():
    """Get the current spatial map of all anchored detections."""
    try:
        svc = _get_slam_service()
        anchors = svc.get_active_map()
        
        return SlamMapResponse(
            anchors=[
                SpatialAnchorResponse(
                    id=a.id,
                    label=a.label,
                    relative_coords=list(a.relative_coords),
                    confidence=a.confidence,
                )
                for a in anchors
            ],
            anchor_count=len(anchors),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/slam/reset", response_model=SlamStatusResponse, tags=["SLAM"])
async def reset_slam():
    """Reset the SLAM service and clear all anchors."""
    global _slam_service
    try:
        if _slam_service:
            imu_enabled = _slam_service.imu_enabled
            _slam_service = SlamService({"imu_enabled": imu_enabled})
        else:
            _slam_service = SlamService({})
        
        return SlamStatusResponse(
            initialized=True,
            imu_enabled=_slam_service.imu_enabled,
            active_anchors=0,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
