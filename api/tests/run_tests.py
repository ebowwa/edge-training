#!/usr/bin/env python3
"""
Standalone test runner for API tests.
Works without pytest - just uses built-in unittest assertions.
"""

import sys
import os
from pathlib import Path

# Add api directory to path
api_dir = Path(__file__).parent.parent
sys.path.insert(0, str(api_dir))

print("=" * 60)
print("Testing API schemas...")
print("=" * 60)

from schemas import (
    DatasetRequest, DatasetResponse,
    TrainingRequest, TrainingResponse,
    InferenceRequest, InferenceResponse, DetectionResult,
    ValidationRequest, ValidationResponse,
    ExportRequest, ExportResponse,
    PreprocessingRequest, PreprocessingResponse,
    HealthResponse,
)

passed = 0
failed = 0


def test(name, func):
    """Run a test and track results."""
    global passed, failed
    try:
        func()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1


# === Dataset Schema Tests ===
print("\n[1/7] Testing Dataset schemas...")


def test_dataset_request():
    req = DatasetRequest(
        dataset_handle="user/pothole-detection",
        nc=2,
        names=["pothole", "crack"]
    )
    assert req.dataset_handle == "user/pothole-detection"
    assert req.nc == 2
    assert req.names == ["pothole", "crack"]


test("DatasetRequest with all fields", test_dataset_request)


def test_dataset_response():
    resp = DatasetResponse(
        yaml_path="/data/data.yaml",
        dataset_path="/data/dataset",
        splits={"train": "/data/train", "val": "/data/val"}
    )
    assert resp.yaml_path == "/data/data.yaml"
    assert "train" in resp.splits


test("DatasetResponse creation", test_dataset_response)


# === Training Schema Tests ===
print("\n[2/7] Testing Training schemas...")


def test_training_request_defaults():
    req = TrainingRequest(yaml_path="/data/data.yaml")
    assert req.epochs == 60
    assert req.imgsz == 512
    assert req.batch == 32
    assert req.device == "0"
    assert req.base_model == "yolov8m.pt"


test("TrainingRequest defaults", test_training_request_defaults)


def test_training_request_custom():
    req = TrainingRequest(
        yaml_path="/data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=64
    )
    assert req.epochs == 100
    assert req.imgsz == 640


test("TrainingRequest custom values", test_training_request_custom)


def test_training_request_validation():
    try:
        TrainingRequest(yaml_path="/data/data.yaml", epochs=0)
        raise AssertionError("Should have raised validation error")
    except ValueError:
        pass  # Expected


test("TrainingRequest epochs validation", test_training_request_validation)


def test_training_response():
    resp = TrainingResponse(
        best_model_path="/runs/train/best.pt",
        last_model_path="/runs/train/last.pt",
        epochs_completed=60,
        metrics={"mAP50": 0.85}
    )
    assert resp.best_model_path == "/runs/train/best.pt"
    assert resp.metrics["mAP50"] == 0.85


test("TrainingResponse creation", test_training_response)


# === Inference Schema Tests ===
print("\n[3/7] Testing Inference schemas...")


def test_inference_request_defaults():
    req = InferenceRequest(model_path="/models/best.pt")
    assert req.conf_threshold == 0.5
    assert req.iou_threshold == 0.45


test("InferenceRequest defaults", test_inference_request_defaults)


def test_inference_threshold_validation():
    try:
        InferenceRequest(model_path="/m.pt", conf_threshold=1.5)
        raise AssertionError("Should have raised validation error")
    except ValueError:
        pass


test("InferenceRequest threshold validation", test_inference_threshold_validation)


def test_detection_result():
    det = DetectionResult(
        class_id=0,
        class_name="pothole",
        confidence=0.95,
        bbox=[100.0, 200.0, 300.0, 400.0],
        bbox_normalized=[0.5, 0.5, 0.2, 0.2]
    )
    assert det.class_name == "pothole"
    assert det.confidence == 0.95
    assert len(det.bbox) == 4


test("DetectionResult creation", test_detection_result)


def test_inference_response():
    detections = [
        DetectionResult(
            class_id=0, class_name="pothole", confidence=0.9,
            bbox=[10, 20, 30, 40], bbox_normalized=[0.2, 0.3, 0.1, 0.1]
        )
    ]
    resp = InferenceResponse(
        detections=detections,
        inference_time_ms=15.5,
        image_size=[640, 480]
    )
    assert len(resp.detections) == 1
    assert resp.inference_time_ms == 15.5


test("InferenceResponse creation", test_inference_response)


# === Validation Schema Tests ===
print("\n[4/7] Testing Validation schemas...")


def test_validation_request():
    req = ValidationRequest(
        model_path="/models/best.pt",
        yaml_path="/data/data.yaml"
    )
    assert req.imgsz == 512
    assert req.split == "test"


test("ValidationRequest defaults", test_validation_request)


def test_validation_response():
    resp = ValidationResponse(
        map50=0.85,
        map50_95=0.65,
        precision=0.88,
        recall=0.82
    )
    assert resp.map50 == 0.85
    assert resp.precision == 0.88


test("ValidationResponse creation", test_validation_response)


# === Export Schema Tests ===
print("\n[5/7] Testing Export schemas...")


def test_export_request_defaults():
    req = ExportRequest(model_path="/models/best.pt")
    assert req.format == "ncnn"
    assert req.imgsz == 512
    assert req.half is False


test("ExportRequest defaults", test_export_request_defaults)


def test_export_response():
    resp = ExportResponse(
        export_path="/exports/model_ncnn",
        format="ncnn"
    )
    assert resp.format == "ncnn"


test("ExportResponse creation", test_export_response)


# === Preprocessing Schema Tests ===
print("\n[6/7] Testing Preprocessing schemas...")


def test_preprocessing_request_defaults():
    req = PreprocessingRequest(
        images_dir="/data/images",
        labels_dir="/data/labels"
    )
    assert req.augment_factor == 2
    assert req.num_workers == 1
    assert req.clean is True
    assert req.augment is True


test("PreprocessingRequest defaults", test_preprocessing_request_defaults)


def test_preprocessing_request_validation():
    try:
        PreprocessingRequest(
            images_dir="/i", labels_dir="/l", augment_factor=15
        )
        raise AssertionError("Should have raised validation error")
    except ValueError:
        pass


test("PreprocessingRequest validation", test_preprocessing_request_validation)


def test_preprocessing_response():
    resp = PreprocessingResponse(
        images_processed=100,
        images_removed=5,
        labels_fixed=3,
        images_augmented=190
    )
    assert resp.images_processed == 100
    assert resp.images_augmented == 190


test("PreprocessingResponse creation", test_preprocessing_response)


# === Health Schema Tests ===
print("\n[7/7] Testing Health schema...")


def test_health_response_defaults():
    resp = HealthResponse()
    assert resp.status == "ok"
    assert resp.version == "0.1.0"


test("HealthResponse defaults", test_health_response_defaults)


# === Serialization Tests ===
print("\n[BONUS] Testing serialization...")


def test_model_dump():
    req = TrainingRequest(yaml_path="/data/data.yaml", epochs=50)
    data = req.model_dump()
    assert data["yaml_path"] == "/data/data.yaml"
    assert data["epochs"] == 50


test("TrainingRequest model_dump", test_model_dump)


def test_json_roundtrip():
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


test("InferenceResponse roundtrip", test_json_roundtrip)


# === Summary ===
print("\n" + "=" * 60)
if failed == 0:
    print(f"✓ All {passed} tests passed!")
else:
    print(f"✗ {passed} passed, {failed} failed")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
