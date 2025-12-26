"""
Pydantic schemas for API request/response models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from service.config import ModelRegistry


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


class RoboflowDatasetRequest(BaseModel):
    """Request to download Roboflow dataset."""
    url: str = Field(..., description="Roboflow export URL (https://app.roboflow.com/ds/KEY?style=zip)")
    output_dir: Optional[str] = Field("datasets/roboflow", description="Output directory")


class RoboflowDatasetResponse(BaseModel):
    """Response from Roboflow dataset download."""
    yaml_path: str
    dataset_path: str
    info: Dict[str, Any] = {}  # Dataset statistics


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
    base_model: str = Field(default=ModelRegistry.get_default_path())


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


# === SLAM Schemas ===

class ImuData(BaseModel):
    """IMU sensor data for pose estimation."""
    accel_x: float = Field(0.0, description="Accelerometer X")
    accel_y: float = Field(0.0, description="Accelerometer Y")
    accel_z: float = Field(0.0, description="Accelerometer Z")
    gyro_x: float = Field(0.0, description="Gyroscope X")
    gyro_y: float = Field(0.0, description="Gyroscope Y")
    gyro_z: float = Field(0.0, description="Gyroscope Z")


class SlamPoseRequest(BaseModel):
    """Request to update device pose from a frame."""
    imu_data: Optional[ImuData] = None


class DevicePoseResponse(BaseModel):
    """Device pose (position/rotation) response."""
    timestamp: float
    delta_x: float
    delta_y: float
    rotation_deg: float


class AnchorDetectionRequest(BaseModel):
    """Request to create a spatial anchor from a detection."""
    class_name: str = Field(..., description="Class name of detection")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")


class SpatialAnchorResponse(BaseModel):
    """Spatial anchor response."""
    id: int
    label: str
    relative_coords: List[float]  # [x, y]
    confidence: float


class SlamMapResponse(BaseModel):
    """Response with current spatial map."""
    anchors: List[SpatialAnchorResponse]
    anchor_count: int


class SlamConfigRequest(BaseModel):
    """Configuration for SLAM service initialization."""
    imu_enabled: bool = Field(False, description="Enable IMU sensor fusion")


class SlamStatusResponse(BaseModel):
    """SLAM service status."""
    initialized: bool
    imu_enabled: bool
    active_anchors: int


# === Health Check ===

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.1.0"


# === Planner Schemas ===

class PlannerRequest(BaseModel):
    """Request for training resource planning."""
    num_images: int = Field(..., ge=1, description="Number of training images")
    resolution_width: int = Field(640, ge=32, le=4096, description="Image width")
    resolution_height: int = Field(640, ge=32, le=4096, description="Image height")
    model_variant: str = Field("m", pattern="^(n|s|m|l|x)$", description="YOLO variant")
    epochs: int = Field(60, ge=1, le=1000)
    batch_size: int = Field(32, ge=1, le=256)
    augment_factor: int = Field(1, ge=1, le=10, description="Augmentation multiplier")
    gpu_name: Optional[str] = Field(None, description="Specific GPU for cost estimation")


class StorageEstimate(BaseModel):
    """Storage requirement estimates."""
    dataset: str
    dataset_augmented: str
    model_weights: str
    checkpoints: str
    cache: str
    total: str


class ComputeEstimate(BaseModel):
    """Compute requirement estimates."""
    vram: str
    recommended_gpu: str
    training_hours: str
    cpu_hours: Optional[str] = None
    recommended_cpu: Optional[str] = None
    ram: Optional[str] = None
    estimated_cost: Optional[str] = None


class PlannerResponse(BaseModel):
    """Response from training resource planning."""
    storage: StorageEstimate
    compute: ComputeEstimate
    specs: Dict[str, str]
