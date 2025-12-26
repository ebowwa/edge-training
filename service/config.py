"""
Configuration dataclasses for YOLO Training Service.
Type-safe configuration for all service operations.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """
    Centralized registry for YOLO and RF-DETR models.

    All model path references should go through this registry to ensure
    consistency and make model management easier.

    Models are loaded from models.json in the same directory as this file.

    Usage:
        from service.config import ModelRegistry

        # Get a specific model
        path = ModelRegistry.get_path("yolov8m.pt")  # -> "resources/yolov8m.pt"

        # Use the default model
        path = ModelRegistry.get_path(ModelRegistry.DEFAULT)

        # List available models
        models = ModelRegistry.list_available()

        # Get model info
        info = ModelRegistry.get_model_info("yolov8m.pt")

        # Auto-download if missing (inspired by RF-DETR)
        path = ModelRegistry.download_if_missing("yolov8n.pt")
    """

    MODELS_DIR = "resources"
    DEFAULT = "yolov8m.pt"
    _MODELS_JSON: Optional[Dict[str, Any]] = None
    _JSON_PATH = Path(__file__).parent / "models.json"

    @classmethod
    def _load_models_json(cls) -> Dict[str, Any]:
        """Load models from JSON file (cached)."""
        if cls._MODELS_JSON is None:
            if not cls._JSON_PATH.exists():
                raise FileNotFoundError(
                    f"Models configuration file not found: {cls._JSON_PATH}"
                )
            with open(cls._JSON_PATH, "r") as f:
                cls._MODELS_JSON = json.load(f)
        return cls._MODELS_JSON

    @classmethod
    def _reload_models(cls) -> None:
        """Force reload of models JSON file."""
        cls._MODELS_JSON = None
        cls._load_models_json()

    @classmethod
    def get_yolo_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get all YOLO models from JSON config."""
        data = cls._load_models_json()
        return data.get("yolo", {}).get("models", {})

    @classmethod
    def get_rfdetr_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get all RF-DETR models from JSON config."""
        data = cls._load_models_json()
        return data.get("rfdetr", {}).get("models", {})

    @classmethod
    def get_yolo_url(cls, model_name: str) -> Optional[str]:
        """Get download URL for a YOLO model."""
        models = cls.get_yolo_models()
        model_info = models.get(model_name, {})
        return model_info.get("url")

    @classmethod
    def get_rfdetr_info(cls, model_name: str) -> Dict[str, Any]:
        """Get info for an RF-DETR model."""
        models = cls.get_rfdetr_models()
        return models.get(model_name, {})
    
    @classmethod
    def get_path(cls, model_name: str) -> str:
        """
        Get the full path for a model by name.
        
        Args:
            model_name: Name of the model file (e.g., "yolov8m.pt")
            
        Returns:
            Full path to the model (e.g., "resources/yolov8m.pt")
        """
        # If already a full path, return as-is
        if "/" in model_name or "\\" in model_name:
            return model_name
        return f"{cls.MODELS_DIR}/{model_name}"
    
    @classmethod
    def get_default_path(cls) -> str:
        """Get the path to the default model."""
        return cls.get_path(cls.DEFAULT)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available .pt model files in the models directory.
        
        Returns:
            List of model filenames (e.g., ["yolov8m.pt", "pothole.pt"])
        """
        models_path = Path(cls.MODELS_DIR)
        if not models_path.exists():
            return []
        return [f.name for f in models_path.glob("*.pt")]
    
    @classmethod
    def list_downloadable(cls) -> List[str]:
        """List all YOLO models available for download."""
        return list(cls.get_yolo_models().keys())
    
    @classmethod
    def exists(cls, model_name: str) -> bool:
        """Check if a model exists in the registry."""
        return Path(cls.get_path(model_name)).exists()
    
    @classmethod
    def download_if_missing(cls, model_name: str, force: bool = False) -> str:
        """
        Download a YOLO model if it's not present locally.

        Inspired by RF-DETR's download_pretrain_weights pattern.

        Args:
            model_name: Name of the model to download (e.g., "yolov8n.pt")
            force: If True, re-download even if file exists

        Returns:
            Path to the downloaded model

        Raises:
            ValueError: If model is not in YOLO models registry
        """
        import logging
        import urllib.request

        model_path = cls.get_path(model_name)

        # Already exists and not forcing re-download
        if cls.exists(model_name) and not force:
            logging.info(f"Model already exists: {model_path}")
            return model_path

        # Get download URL from JSON
        url = cls.get_yolo_url(model_name)
        if url is None:
            available = ", ".join(cls.get_yolo_models().keys())
            raise ValueError(
                f"Model '{model_name}' not found in YOLO registry. "
                f"Available: {available}"
            )

        # Ensure directory exists
        Path(cls.MODELS_DIR).mkdir(parents=True, exist_ok=True)

        logging.info(f"Downloading {model_name} from {url}...")

        try:
            urllib.request.urlretrieve(url, model_path)
            logging.info(f"Successfully downloaded: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_name}: {e}")

        return model_path
    
    @classmethod
    def is_rfdetr_model(cls, model_name: str) -> bool:
        """Check if model name is an RF-DETR variant."""
        return model_name.startswith("rfdetr-") or model_name in cls.get_rfdetr_models()
    
    @classmethod
    def get_backend(cls, model_name: str) -> str:
        """
        Determine which backend to use for a given model name.
        
        Returns:
            "rfdetr" or "yolo"
        """
        if cls.is_rfdetr_model(model_name):
            return "rfdetr"
        return "yolo"
    
    @classmethod
    def list_all_models(cls) -> dict:
        """
        List all available models (YOLO + RF-DETR).

        Returns:
            Dict with "yolo" and "rfdetr" keys containing model lists
        """
        return {
            "yolo": list(cls.get_yolo_models().keys()),
            "rfdetr": list(cls.get_rfdetr_models().keys()),
        }

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.

        Args:
            model_name: Name of the model (e.g., "yolov8m.pt", "rfdetr-base")

        Returns:
            Dict with model information (url, variant, family, params, etc.)
        """
        # Check YOLO models first
        yolo_models = cls.get_yolo_models()
        if model_name in yolo_models:
            return yolo_models[model_name]

        # Check RF-DETR models
        rfdetr_models = cls.get_rfdetr_models()
        if model_name in rfdetr_models:
            return rfdetr_models[model_name]

        return {}


@dataclass
class DatasetConfig:
    """Configuration for dataset operations."""
    dataset_handle: str  # Kaggle dataset handle
    nc: int  # Number of classes
    names: List[str]  # Class names
    cache_dir: Optional[str] = None  # Optional custom cache directory


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 60
    imgsz: int = 512
    batch: int = 32
    device: str = "0"  # GPU index or "cpu"
    project: str = "runs/train"
    name: str = "yolo_train"
    weights: Optional[str] = None  # Custom pretrained weights path
    base_model: str = None  # Base model, defaults to ModelRegistry.DEFAULT
    
    def __post_init__(self):
        if self.base_model is None:
            self.base_model = ModelRegistry.get_default_path()


@dataclass
class InferenceConfig:
    """Configuration for inference operations."""
    conf_threshold: float = 0.5
    save_output: bool = False
    output_path: Optional[str] = None
    iou_threshold: float = 0.45  # NMS IoU threshold


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    imgsz: int = 512
    split: str = "test"  # Dataset split to validate on
    project: str = "runs/val"
    name: str = "validation"


@dataclass
class ExportConfig:
    """Configuration for model export."""
    format: str = "ncnn"  # Export format: ncnn, onnx, torchscript, etc.
    output_dir: Optional[str] = None
    imgsz: int = 512
    half: bool = False  # FP16 quantization
    dynamic: bool = False  # Dynamic axes for ONNX


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    config_path: Optional[str] = None  # Path to preprocessing config YAML
    augment_factor: int = 2  # Number of augmented versions per image
    clean: bool = True  # Run cleaning before augmentation


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixels
    bbox_normalized: List[float]  # [x_center, y_center, width, height] normalized


@dataclass
class InferenceResult:
    """Result from inference operation."""
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_size: tuple = (0, 0)  # (width, height)
    annotated_image: Optional[any] = None  # numpy array if requested


@dataclass
class TrainingResult:
    """Result from training operation."""
    best_model_path: str
    last_model_path: str
    metrics: dict = field(default_factory=dict)
    epochs_completed: int = 0


@dataclass
class ValidationResult:
    """Result from validation operation."""
    map50: float = 0.0  # mAP@0.5
    map50_95: float = 0.0  # mAP@0.5:0.95
    precision: float = 0.0
    recall: float = 0.0
    metrics: dict = field(default_factory=dict)
