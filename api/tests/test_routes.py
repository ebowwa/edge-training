#!/usr/bin/env python3
"""
Tests for API routes.
Uses FastAPI TestClient without requiring real services.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO

# Add api and service directories to path
api_dir = Path(__file__).parent.parent
root_dir = api_dir.parent
service_dir = root_dir / "service"
preprocessing_dir = service_dir / "preprocessing"

sys.path.insert(0, str(api_dir))
sys.path.insert(0, str(service_dir))
sys.path.insert(0, str(preprocessing_dir))


@pytest.fixture
def mock_services():
    """Mock all service modules before importing routes."""
    # Create mock modules
    mock_config = MagicMock()
    mock_dataset = MagicMock()
    mock_training = MagicMock()
    mock_inference = MagicMock()
    mock_validation = MagicMock()
    mock_export = MagicMock()
    mock_pipeline = MagicMock()
    mock_cleaners = MagicMock()
    mock_transforms = MagicMock()
    
    with patch.dict(sys.modules, {
        'config': mock_config,
        'service_config': mock_config,
        'dataset_service': mock_dataset,
        'training_service': mock_training,
        'inference_service': mock_inference,
        'validation_service': mock_validation,
        'export_service': mock_export,
        'pipeline': mock_pipeline,
        'preprocessing_pipeline': mock_pipeline,
        'cleaners': mock_cleaners,
        'transforms': mock_transforms,
    }):
        yield {
            'config': mock_config,
            'dataset_service': mock_dataset,
            'training_service': mock_training,
            'inference_service': mock_inference,
            'validation_service': mock_validation,
            'export_service': mock_export,
            'pipeline': mock_pipeline,
            'cleaners': mock_cleaners,
            'transforms': mock_transforms,
        }


@pytest.fixture
def client(mock_services):
    """Create test client with mocked services."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    # Import routes after mocking
    from routes import router
    
    app = FastAPI()
    app.include_router(router)
    
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestDatasetEndpoint:
    """Tests for /datasets/prepare endpoint."""

    def test_prepare_dataset_success(self, client, mock_services):
        """Test successful dataset preparation."""
        # Setup mocks
        mock_services['config'].DatasetConfig = MagicMock(return_value=MagicMock())
        mock_services['dataset_service'].DatasetService.download.return_value = "/data/dataset"
        mock_services['dataset_service'].DatasetService.detect_structure.return_value = (
            {"train_images": "/data/train", "val_images": "/data/val"},
            "/data/dataset"
        )
        mock_services['dataset_service'].DatasetService.create_yaml.return_value = "/data/data.yaml"
        
        response = client.post("/datasets/prepare", json={
            "dataset_handle": "user/dataset",
            "nc": 2,
            "names": ["class1", "class2"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "yaml_path" in data
        assert "dataset_path" in data
        assert "splits" in data

    def test_prepare_dataset_invalid_structure(self, client, mock_services):
        """Test dataset with invalid structure returns 400."""
        mock_services['config'].DatasetConfig = MagicMock(return_value=MagicMock())
        mock_services['dataset_service'].DatasetService.download.return_value = "/data"
        mock_services['dataset_service'].DatasetService.detect_structure.return_value = ({}, "/data")
        
        response = client.post("/datasets/prepare", json={
            "dataset_handle": "user/invalid",
            "nc": 1,
            "names": ["test"]
        })
        
        assert response.status_code == 400

    def test_prepare_dataset_missing_fields(self, client):
        """Test dataset request with missing fields returns 422."""
        response = client.post("/datasets/prepare", json={
            "dataset_handle": "user/dataset"
            # missing nc and names
        })
        assert response.status_code == 422


class TestTrainingEndpoint:
    """Tests for /train endpoint."""

    def test_train_model_success(self, client, mock_services):
        """Test successful model training."""
        mock_result = MagicMock()
        mock_result.best_model_path = "/runs/train/best.pt"
        mock_result.last_model_path = "/runs/train/last.pt"
        mock_result.epochs_completed = 60
        mock_result.metrics = {"mAP50": 0.85}
        
        mock_services['config'].TrainingConfig = MagicMock(return_value=MagicMock())
        mock_services['training_service'].TrainingService.train.return_value = mock_result
        
        response = client.post("/train", json={
            "yaml_path": "/data/data.yaml",
            "epochs": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["best_model_path"] == "/runs/train/best.pt"
        assert data["epochs_completed"] == 60

    def test_train_invalid_epochs(self, client):
        """Test training with invalid epochs validation."""
        response = client.post("/train", json={
            "yaml_path": "/data/data.yaml",
            "epochs": 0  # invalid: must be >= 1
        })
        assert response.status_code == 422

    def test_resume_training_success(self, client, mock_services):
        """Test resume training endpoint."""
        mock_result = MagicMock()
        mock_result.best_model_path = "/runs/train/best.pt"
        mock_result.last_model_path = "/runs/train/last.pt"
        mock_result.epochs_completed = 80
        mock_result.metrics = {}
        
        mock_services['training_service'].TrainingService.resume.return_value = mock_result
        
        response = client.post("/train/resume", params={
            "project": "runs/train",
            "name": "yolo_train"
        })
        
        assert response.status_code == 200

    def test_resume_training_not_found(self, client, mock_services):
        """Test resume training when checkpoint not found."""
        mock_services['training_service'].TrainingService.resume.side_effect = (
            FileNotFoundError("No checkpoint found")
        )
        
        response = client.post("/train/resume")
        assert response.status_code == 404


class TestInferenceEndpoint:
    """Tests for /infer/image endpoint."""

    def test_infer_image_success(self, client, mock_services):
        """Test successful image inference."""
        # Setup mock detection result
        mock_detection = MagicMock()
        mock_detection.class_id = 0
        mock_detection.class_name = "pothole"
        mock_detection.confidence = 0.95
        mock_detection.bbox = [100, 200, 300, 400]
        mock_detection.bbox_normalized = [0.5, 0.5, 0.2, 0.2]
        
        mock_result = MagicMock()
        mock_result.detections = [mock_detection]
        mock_result.inference_time_ms = 15.5
        mock_result.image_size = (640, 480)
        
        mock_inference_svc = MagicMock()
        mock_inference_svc.infer_image.return_value = mock_result
        mock_services['inference_service'].InferenceService.return_value = mock_inference_svc
        mock_services['config'].InferenceConfig = MagicMock(return_value=MagicMock())
        
        # Create test image file
        image_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        files = {"image": ("test.jpg", BytesIO(image_content), "image/jpeg")}
        
        response = client.post(
            "/infer/image",
            params={"model_path": "/models/best.pt"},
            files=files
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "inference_time_ms" in data

    def test_infer_image_model_not_found(self, client, mock_services):
        """Test inference with non-existent model."""
        mock_services['inference_service'].InferenceService.side_effect = (
            FileNotFoundError("Model not found")
        )
        
        image_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        files = {"image": ("test.jpg", BytesIO(image_content), "image/jpeg")}
        
        response = client.post(
            "/infer/image",
            params={"model_path": "/nonexistent/model.pt"},
            files=files
        )
        
        assert response.status_code == 404


class TestValidationEndpoint:
    """Tests for /validate endpoint."""

    def test_validate_model_success(self, client, mock_services):
        """Test successful model validation."""
        mock_result = MagicMock()
        mock_result.map50 = 0.85
        mock_result.map50_95 = 0.65
        mock_result.precision = 0.88
        mock_result.recall = 0.82
        mock_result.metrics = {}
        
        mock_services['config'].ValidationConfig = MagicMock(return_value=MagicMock())
        mock_services['validation_service'].ValidationService.validate.return_value = mock_result
        
        response = client.post("/validate", json={
            "model_path": "/models/best.pt",
            "yaml_path": "/data/data.yaml"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["map50"] == 0.85
        assert data["precision"] == 0.88

    def test_validate_model_not_found(self, client, mock_services):
        """Test validation with non-existent model."""
        mock_services['config'].ValidationConfig = MagicMock(return_value=MagicMock())
        mock_services['validation_service'].ValidationService.validate.side_effect = (
            FileNotFoundError("Model not found")
        )
        
        response = client.post("/validate", json={
            "model_path": "/nonexistent/model.pt",
            "yaml_path": "/data/data.yaml"
        })
        
        assert response.status_code == 404


class TestExportEndpoint:
    """Tests for /export endpoint."""

    def test_export_ncnn_success(self, client, mock_services):
        """Test successful NCNN export."""
        mock_services['config'].ExportConfig = MagicMock(return_value=MagicMock())
        mock_services['export_service'].ExportService.export.return_value = "/exports/model_ncnn"
        
        response = client.post("/export", json={
            "model_path": "/models/best.pt",
            "format": "ncnn"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "ncnn"
        assert "export_path" in data

    def test_export_invalid_format(self, client, mock_services):
        """Test export with invalid format."""
        mock_services['config'].ExportConfig = MagicMock(return_value=MagicMock())
        mock_services['export_service'].ExportService.export.side_effect = (
            ValueError("Unsupported format")
        )
        
        response = client.post("/export", json={
            "model_path": "/models/best.pt",
            "format": "invalid"
        })
        
        assert response.status_code == 400

    def test_export_model_not_found(self, client, mock_services):
        """Test export with non-existent model."""
        mock_services['config'].ExportConfig = MagicMock(return_value=MagicMock())
        mock_services['export_service'].ExportService.export.side_effect = (
            FileNotFoundError("Model not found")
        )
        
        response = client.post("/export", json={
            "model_path": "/nonexistent/model.pt",
            "format": "onnx"
        })
        
        assert response.status_code == 404


class TestPreprocessingEndpoint:
    """Tests for /preprocess endpoint."""

    def test_preprocess_success(self, client, mock_services):
        """Test successful preprocessing."""
        mock_result = MagicMock()
        mock_result.images_processed = 100
        mock_result.images_removed = 5
        mock_result.labels_fixed = 3
        mock_result.images_augmented = 190
        mock_result.errors = []
        
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = mock_result
        mock_services['pipeline'].PreprocessingPipeline.return_value = mock_pipeline
        
        mock_services['cleaners'].CorruptedImageCleaner.return_value = MagicMock()
        mock_services['cleaners'].BBoxValidator.return_value = MagicMock()
        mock_services['transforms'].FlipTransform.return_value = MagicMock()
        mock_services['transforms'].RotateTransform.return_value = MagicMock()
        mock_services['transforms'].ColorTransform.return_value = MagicMock()
        
        response = client.post("/preprocess", json={
            "images_dir": "/data/images",
            "labels_dir": "/data/labels"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["images_processed"] == 100
        assert data["images_augmented"] == 190

    def test_preprocess_clean_only(self, client, mock_services):
        """Test preprocessing with cleaning only (no augmentation)."""
        mock_result = MagicMock()
        mock_result.images_processed = 50
        mock_result.images_removed = 2
        mock_result.labels_fixed = 1
        mock_result.images_augmented = 0
        mock_result.errors = []
        
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = mock_result
        mock_services['pipeline'].PreprocessingPipeline.return_value = mock_pipeline
        
        mock_services['cleaners'].CorruptedImageCleaner.return_value = MagicMock()
        mock_services['cleaners'].BBoxValidator.return_value = MagicMock()
        
        response = client.post("/preprocess", json={
            "images_dir": "/data/images",
            "labels_dir": "/data/labels",
            "clean": True,
            "augment": False
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["images_augmented"] == 0

    def test_preprocess_invalid_augment_factor(self, client):
        """Test preprocessing with invalid augment factor."""
        response = client.post("/preprocess", json={
            "images_dir": "/data/images",
            "labels_dir": "/data/labels",
            "augment_factor": 15  # max is 10
        })
        assert response.status_code == 422


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_internal_server_error(self, client, mock_services):
        """Test 500 error handling."""
        mock_services['config'].TrainingConfig = MagicMock(return_value=MagicMock())
        mock_services['training_service'].TrainingService.train.side_effect = (
            RuntimeError("GPU out of memory")
        )
        
        response = client.post("/train", json={
            "yaml_path": "/data/data.yaml"
        })
        
        assert response.status_code == 500
        assert "GPU out of memory" in response.json()["detail"]

    def test_validation_error_details(self, client):
        """Test validation error returns details."""
        response = client.post("/train", json={
            "yaml_path": "/data/data.yaml",
            "epochs": "not_a_number"  # invalid type
        })
        assert response.status_code == 422
        assert "detail" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
