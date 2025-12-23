#!/usr/bin/env python3
"""
Tests for API Pydantic schemas.
Validates field validation, defaults, and serialization.
"""

import pytest
import sys
from pathlib import Path

# Add api directory to path
api_dir = Path(__file__).parent.parent
sys.path.insert(0, str(api_dir))

from schemas import (
    DatasetRequest, DatasetResponse,
    TrainingRequest, TrainingResponse,
    InferenceRequest, InferenceResponse, DetectionResult,
    ValidationRequest, ValidationResponse,
    ExportRequest, ExportResponse,
    PreprocessingRequest, PreprocessingResponse,
    HealthResponse,
)


class TestDatasetSchemas:
    """Tests for Dataset request/response schemas."""

    def test_dataset_request_required_fields(self):
        """Test DatasetRequest requires all fields."""
        request = DatasetRequest(
            dataset_handle="user/pothole-detection",
            nc=2,
            names=["pothole", "crack"]
        )
        assert request.dataset_handle == "user/pothole-detection"
        assert request.nc == 2
        assert request.names == ["pothole", "crack"]

    def test_dataset_request_missing_field_raises(self):
        """Test DatasetRequest raises on missing required fields."""
        with pytest.raises(ValueError):
            DatasetRequest(dataset_handle="test/data")  # missing nc and names

    def test_dataset_response(self):
        """Test DatasetResponse creation."""
        response = DatasetResponse(
            yaml_path="/data/data.yaml",
            dataset_path="/data/dataset",
            splits={"train": "/data/train", "val": "/data/val"}
        )
        assert response.yaml_path == "/data/data.yaml"
        assert "train" in response.splits


class TestTrainingSchemas:
    """Tests for Training request/response schemas."""

    def test_training_request_defaults(self):
        """Test TrainingRequest field defaults."""
        request = TrainingRequest(yaml_path="/data/data.yaml")
        assert request.epochs == 60
        assert request.imgsz == 512
        assert request.batch == 32
        assert request.device == "0"
        assert request.project == "runs/train"
        assert request.name == "yolo_train"
        assert request.weights is None
        assert request.base_model == "yolov8m.pt"

    def test_training_request_custom_values(self):
        """Test TrainingRequest with custom values."""
        request = TrainingRequest(
            yaml_path="/data/data.yaml",
            epochs=100,
            imgsz=640,
            batch=64,
            device="cuda:0",
            weights="/custom/weights.pt"
        )
        assert request.epochs == 100
        assert request.imgsz == 640
        assert request.batch == 64
        assert request.weights == "/custom/weights.pt"

    def test_training_request_epoch_validation(self):
        """Test epochs field validation bounds."""
        # Valid edge cases
        request_min = TrainingRequest(yaml_path="/data/data.yaml", epochs=1)
        assert request_min.epochs == 1
        
        request_max = TrainingRequest(yaml_path="/data/data.yaml", epochs=1000)
        assert request_max.epochs == 1000
        
        # Invalid cases
        with pytest.raises(ValueError):
            TrainingRequest(yaml_path="/data/data.yaml", epochs=0)
        with pytest.raises(ValueError):
            TrainingRequest(yaml_path="/data/data.yaml", epochs=1001)

    def test_training_request_batch_validation(self):
        """Test batch field validation bounds."""
        with pytest.raises(ValueError):
            TrainingRequest(yaml_path="/data/data.yaml", batch=0)
        with pytest.raises(ValueError):
            TrainingRequest(yaml_path="/data/data.yaml", batch=257)

    def test_training_response(self):
        """Test TrainingResponse creation."""
        response = TrainingResponse(
            best_model_path="/runs/train/best.pt",
            last_model_path="/runs/train/last.pt",
            epochs_completed=60,
            metrics={"mAP50": 0.85}
        )
        assert response.best_model_path == "/runs/train/best.pt"
        assert response.epochs_completed == 60
        assert response.metrics["mAP50"] == 0.85


class TestInferenceSchemas:
    """Tests for Inference request/response schemas."""

    def test_inference_request_defaults(self):
        """Test InferenceRequest defaults."""
        request = InferenceRequest(model_path="/models/best.pt")
        assert request.conf_threshold == 0.5
        assert request.iou_threshold == 0.45

    def test_inference_request_threshold_validation(self):
        """Test threshold bounds (0.0 to 1.0)."""
        # Valid edge cases
        InferenceRequest(model_path="/m.pt", conf_threshold=0.0, iou_threshold=0.0)
        InferenceRequest(model_path="/m.pt", conf_threshold=1.0, iou_threshold=1.0)
        
        # Invalid cases
        with pytest.raises(ValueError):
            InferenceRequest(model_path="/m.pt", conf_threshold=-0.1)
        with pytest.raises(ValueError):
            InferenceRequest(model_path="/m.pt", conf_threshold=1.1)
        with pytest.raises(ValueError):
            InferenceRequest(model_path="/m.pt", iou_threshold=-0.1)

    def test_detection_result(self):
        """Test DetectionResult creation."""
        detection = DetectionResult(
            class_id=0,
            class_name="pothole",
            confidence=0.95,
            bbox=[100.0, 200.0, 300.0, 400.0],
            bbox_normalized=[0.5, 0.5, 0.2, 0.2]
        )
        assert detection.class_id == 0
        assert detection.class_name == "pothole"
        assert detection.confidence == 0.95
        assert len(detection.bbox) == 4

    def test_inference_response(self):
        """Test InferenceResponse with multiple detections."""
        detections = [
            DetectionResult(
                class_id=0, class_name="pothole", confidence=0.9,
                bbox=[10, 20, 30, 40], bbox_normalized=[0.2, 0.3, 0.1, 0.1]
            ),
            DetectionResult(
                class_id=1, class_name="crack", confidence=0.8,
                bbox=[50, 60, 70, 80], bbox_normalized=[0.4, 0.5, 0.1, 0.1]
            ),
        ]
        response = InferenceResponse(
            detections=detections,
            inference_time_ms=15.5,
            image_size=[640, 480]
        )
        assert len(response.detections) == 2
        assert response.inference_time_ms == 15.5
        assert response.image_size == [640, 480]


class TestValidationSchemas:
    """Tests for Validation request/response schemas."""

    def test_validation_request_defaults(self):
        """Test ValidationRequest defaults."""
        request = ValidationRequest(
            model_path="/models/best.pt",
            yaml_path="/data/data.yaml"
        )
        assert request.imgsz == 512
        assert request.split == "test"

    def test_validation_request_custom_split(self):
        """Test ValidationRequest with custom split."""
        request = ValidationRequest(
            model_path="/models/best.pt",
            yaml_path="/data/data.yaml",
            split="val"
        )
        assert request.split == "val"

    def test_validation_response(self):
        """Test ValidationResponse metrics."""
        response = ValidationResponse(
            map50=0.85,
            map50_95=0.65,
            precision=0.88,
            recall=0.82,
            metrics={"confusion_matrix": [[50, 5], [3, 42]]}
        )
        assert response.map50 == 0.85
        assert response.map50_95 == 0.65
        assert response.precision == 0.88
        assert response.recall == 0.82


class TestExportSchemas:
    """Tests for Export request/response schemas."""

    def test_export_request_defaults(self):
        """Test ExportRequest defaults."""
        request = ExportRequest(model_path="/models/best.pt")
        assert request.format == "ncnn"
        assert request.imgsz == 512
        assert request.half is False

    def test_export_request_formats(self):
        """Test various export formats."""
        for fmt in ["ncnn", "onnx", "coreml", "tflite"]:
            request = ExportRequest(model_path="/m.pt", format=fmt)
            assert request.format == fmt

    def test_export_request_half_precision(self):
        """Test FP16 quantization flag."""
        request = ExportRequest(model_path="/m.pt", half=True)
        assert request.half is True

    def test_export_response(self):
        """Test ExportResponse creation."""
        response = ExportResponse(
            export_path="/exports/model_ncnn",
            format="ncnn"
        )
        assert response.export_path == "/exports/model_ncnn"
        assert response.format == "ncnn"


class TestPreprocessingSchemas:
    """Tests for Preprocessing request/response schemas."""

    def test_preprocessing_request_defaults(self):
        """Test PreprocessingRequest defaults."""
        request = PreprocessingRequest(
            images_dir="/data/images",
            labels_dir="/data/labels"
        )
        assert request.augment_factor == 2
        assert request.num_workers == 1
        assert request.clean is True
        assert request.augment is True
        assert request.output_images_dir is None
        assert request.output_labels_dir is None

    def test_preprocessing_request_validation(self):
        """Test PreprocessingRequest field validation."""
        # Valid edge cases
        PreprocessingRequest(
            images_dir="/i", labels_dir="/l", augment_factor=1, num_workers=1
        )
        PreprocessingRequest(
            images_dir="/i", labels_dir="/l", augment_factor=10, num_workers=16
        )
        
        # Invalid cases
        with pytest.raises(ValueError):
            PreprocessingRequest(
                images_dir="/i", labels_dir="/l", augment_factor=0
            )
        with pytest.raises(ValueError):
            PreprocessingRequest(
                images_dir="/i", labels_dir="/l", augment_factor=11
            )
        with pytest.raises(ValueError):
            PreprocessingRequest(
                images_dir="/i", labels_dir="/l", num_workers=0
            )

    def test_preprocessing_response(self):
        """Test PreprocessingResponse creation."""
        response = PreprocessingResponse(
            images_processed=100,
            images_removed=5,
            labels_fixed=3,
            images_augmented=190,
            errors=["error1"]
        )
        assert response.images_processed == 100
        assert response.images_removed == 5
        assert response.labels_fixed == 3
        assert response.images_augmented == 190
        assert len(response.errors) == 1


class TestHealthSchema:
    """Tests for Health check schema."""

    def test_health_response_defaults(self):
        """Test HealthResponse defaults."""
        response = HealthResponse()
        assert response.status == "ok"
        assert response.version == "0.1.0"

    def test_health_response_custom(self):
        """Test HealthResponse with custom values."""
        response = HealthResponse(status="degraded", version="0.2.0")
        assert response.status == "degraded"
        assert response.version == "0.2.0"


class TestSchemaSerialization:
    """Tests for schema serialization/deserialization."""

    def test_training_request_to_dict(self):
        """Test TrainingRequest model_dump."""
        request = TrainingRequest(yaml_path="/data/data.yaml", epochs=50)
        data = request.model_dump()
        assert data["yaml_path"] == "/data/data.yaml"
        assert data["epochs"] == 50

    def test_detection_result_json(self):
        """Test DetectionResult JSON serialization."""
        detection = DetectionResult(
            class_id=0,
            class_name="pothole",
            confidence=0.95,
            bbox=[100, 200, 300, 400],
            bbox_normalized=[0.5, 0.5, 0.2, 0.2]
        )
        json_str = detection.model_dump_json()
        assert "pothole" in json_str
        assert "0.95" in json_str

    def test_inference_response_roundtrip(self):
        """Test InferenceResponse serialization roundtrip."""
        original = InferenceResponse(
            detections=[
                DetectionResult(
                    class_id=0, class_name="test", confidence=0.5,
                    bbox=[0, 0, 10, 10], bbox_normalized=[0.5, 0.5, 0.1, 0.1]
                )
            ],
            inference_time_ms=10.0,
            image_size=[640, 480]
        )
        data = original.model_dump()
        reconstructed = InferenceResponse(**data)
        assert len(reconstructed.detections) == 1
        assert reconstructed.detections[0].class_name == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
