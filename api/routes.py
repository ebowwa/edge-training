"""
FastAPI routes for YOLO Training API.
"""

import sys
import os
import tempfile
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks

# Resolve paths
_api_dir = Path(__file__).parent
_root_dir = _api_dir.parent
_service_dir = _root_dir / "service"
_preprocessing_dir = _service_dir / "preprocessing"

# Import service modules
sys.path.insert(0, str(_service_dir))
import config as service_config
import dataset_service
import training_service
import inference_service
import validation_service
import export_service

# Import preprocessing modules  
sys.path.insert(0, str(_preprocessing_dir))
import pipeline as preprocessing_pipeline
import cleaners
import transforms

from schemas import (
    DatasetRequest, DatasetResponse,
    TrainingRequest, TrainingResponse,
    InferenceRequest, InferenceResponse, DetectionResult,
    ValidationRequest, ValidationResponse,
    ExportRequest, ExportResponse,
    PreprocessingRequest, PreprocessingResponse,
    HealthResponse,
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
