"""
Pydantic schemas for API request/response models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# === Dataset Schemas ===

class DatasetRequest(BaseModel):
    """Request to prepare a dataset."""
    dataset_handle: str = Field(..., description="Kaggle dataset handle")
    nc: int = Field(..., description="Number of classes")
    names: List[str] = Field(..., description="Class names")


class DatasetResponse(BaseModel):
    """Response from dataset preparation."""
    yaml_path: str
    dataset_path: str
    splits: Dict[str, str]  # e.g., {"train": "path", "val": "path"}


# === Training Schemas ===

class TrainingRequest(BaseModel):
    """Request to train a model."""
    yaml_path: str = Field(..., description="Path to data.yaml")
    epochs: int = Field(60, ge=1, le=1000)
    imgsz: int = Field(512, ge=32, le=1280)
    batch: int = Field(32, ge=1, le=256)
    device: str = Field("0", description="GPU index or 'cpu'")
    project: str = Field("runs/train")
    name: str = Field("yolo_train")
    weights: Optional[str] = Field(None, description="Custom weights path")
    base_model: str = Field("yolov8m.pt")


class TrainingResponse(BaseModel):
    """Response from training."""
    best_model_path: str
    last_model_path: str
    epochs_completed: int
    metrics: Dict[str, Any] = {}


# === Inference Schemas ===

class InferenceRequest(BaseModel):
    """Request for inference."""
    model_path: str = Field(..., description="Path to trained model")
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)


class DetectionResult(BaseModel):
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    bbox_normalized: List[float]  # [x_center, y_center, width, height]


class InferenceResponse(BaseModel):
    """Response from inference."""
    detections: List[DetectionResult]
    inference_time_ms: float
    image_size: List[int]  # [width, height]


# === Validation Schemas ===

class ValidationRequest(BaseModel):
    """Request to validate a model."""
    model_path: str = Field(..., description="Path to trained model")
    yaml_path: str = Field(..., description="Path to data.yaml")
    imgsz: int = Field(512)
    split: str = Field("test", description="Dataset split to validate on")


class ValidationResponse(BaseModel):
    """Response from validation."""
    map50: float
    map50_95: float
    precision: float
    recall: float
    metrics: Dict[str, Any] = {}


# === Export Schemas ===

class ExportRequest(BaseModel):
    """Request to export a model."""
    model_path: str = Field(..., description="Path to trained model")
    format: str = Field("ncnn", description="Export format: ncnn, onnx, coreml, tflite")
    imgsz: int = Field(512)
    half: bool = Field(False, description="FP16 quantization")


class ExportResponse(BaseModel):
    """Response from export."""
    export_path: str
    format: str


# === Preprocessing Schemas ===

class PreprocessingRequest(BaseModel):
    """Request to run preprocessing."""
    images_dir: str
    labels_dir: str
    output_images_dir: Optional[str] = None
    output_labels_dir: Optional[str] = None
    augment_factor: int = Field(2, ge=1, le=10)
    num_workers: int = Field(1, ge=1, le=16)
    clean: bool = Field(True)
    augment: bool = Field(True)


class PreprocessingResponse(BaseModel):
    """Response from preprocessing."""
    images_processed: int
    images_removed: int
    labels_fixed: int
    images_augmented: int
    errors: List[str] = []


# === Health Check ===

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.1.0"
