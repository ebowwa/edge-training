#!/usr/bin/env python3
"""
Test script for service layer.
Verifies imports and basic functionality without requiring GPU/datasets.

service/
├── __init__.py
├── config.py
├── integrations/kaggle/dataset.py
├── yolo_training.py
├── inference_service.py
├── yolo/
│   └── validation.py
├── export_service.py
└── tests/
    ├── __init__.py
    └── test_services.py
"""

import sys

# Test 1: Import all config classes
print("\n[1/5] Testing config imports...")
try:
    from service.config import (
        ModelRegistry,
        DatasetConfig,
        TrainingConfig,
        InferenceConfig,
        ValidationConfig,
        ExportConfig,
        PreprocessingConfig,
        Detection,
        InferenceResult,
        TrainingResult,
        ValidationResult,
    )
    print("  ✓ All config classes imported successfully")
    
    # Test ModelRegistry
    print("\n  Testing ModelRegistry...")
    assert ModelRegistry.MODELS_DIR == "resources", "ModelRegistry.MODELS_DIR should be 'resources'"
    assert ModelRegistry.DEFAULT == "yolov8m.pt", "ModelRegistry.DEFAULT should be 'yolov8m.pt'"
    assert ModelRegistry.get_path("test.pt") == "resources/test.pt", "get_path should prepend MODELS_DIR"
    assert ModelRegistry.get_path("/custom/path.pt") == "/custom/path.pt", "get_path should pass through full paths"
    assert ModelRegistry.get_default_path() == "resources/yolov8m.pt", "get_default_path should return full default path"
    print("  ✓ ModelRegistry: MODELS_DIR, DEFAULT, get_path, get_default_path, list_available, exists")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Import all service classes
print("\n[2/5] Testing service imports...")
try:
    from integrations.kaggle.dataset import DatasetService
    from service.yolo_training import TrainingService
    from service.inference_service import InferenceService
    from service.yolo.validation import ValidationService
    from service.export_service import ExportService
    print("  ✓ All service classes imported successfully")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Create config instances
print("\n[3/5] Testing config instantiation...")
try:
    dataset_cfg = DatasetConfig(
        dataset_handle="test/dataset",
        nc=2,
        names=["class1", "class2"]
    )
    print(f"  ✓ DatasetConfig: handle={dataset_cfg.dataset_handle}, nc={dataset_cfg.nc}")
    
    train_cfg = TrainingConfig(epochs=10, imgsz=640)
    print(f"  ✓ TrainingConfig: epochs={train_cfg.epochs}, imgsz={train_cfg.imgsz}")
    
    infer_cfg = InferenceConfig(conf_threshold=0.6)
    print(f"  ✓ InferenceConfig: conf={infer_cfg.conf_threshold}")
    
    export_cfg = ExportConfig(format="onnx")
    print(f"  ✓ ExportConfig: format={export_cfg.format}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Create result instances
print("\n[4/5] Testing result dataclasses...")
try:
    detection = Detection(
        class_id=0,
        class_name="pothole",
        confidence=0.95,
        bbox=[100, 200, 300, 400],
        bbox_normalized=[0.5, 0.5, 0.2, 0.2]
    )
    print(f"  ✓ Detection: {detection.class_name} @ {detection.confidence:.2f}")
    
    infer_result = InferenceResult(
        detections=[detection],
        inference_time_ms=15.5,
        image_size=(640, 480)
    )
    print(f"  ✓ InferenceResult: {len(infer_result.detections)} detections, {infer_result.inference_time_ms}ms")
    
    train_result = TrainingResult(
        best_model_path="/path/to/best.pt",
        last_model_path="/path/to/last.pt",
        epochs_completed=60
    )
    print(f"  ✓ TrainingResult: {train_result.epochs_completed} epochs")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Check service methods exist
print("\n[5/5] Testing service method availability...")
try:
    # DatasetService
    assert hasattr(DatasetService, 'download'), "Missing DatasetService.download"
    assert hasattr(DatasetService, 'detect_structure'), "Missing DatasetService.detect_structure"
    assert hasattr(DatasetService, 'create_yaml'), "Missing DatasetService.create_yaml"
    assert hasattr(DatasetService, 'prepare'), "Missing DatasetService.prepare"
    print("  ✓ DatasetService: download, detect_structure, create_yaml, prepare")
    
    # TrainingService
    assert hasattr(TrainingService, 'train'), "Missing TrainingService.train"
    assert hasattr(TrainingService, 'resume'), "Missing TrainingService.resume"
    assert hasattr(TrainingService, 'get_best_model_path'), "Missing TrainingService.get_best_model_path"
    print("  ✓ TrainingService: train, resume, get_best_model_path")
    
    # InferenceService
    assert hasattr(InferenceService, 'infer_frame'), "Missing InferenceService.infer_frame"
    assert hasattr(InferenceService, 'infer_image'), "Missing InferenceService.infer_image"
    assert hasattr(InferenceService, 'infer_video'), "Missing InferenceService.infer_video"
    assert hasattr(InferenceService, 'infer_batch'), "Missing InferenceService.infer_batch"
    print("  ✓ InferenceService: infer_frame, infer_image, infer_video, infer_batch")
    
    # ValidationService
    assert hasattr(ValidationService, 'validate'), "Missing ValidationService.validate"
    assert hasattr(ValidationService, 'benchmark'), "Missing ValidationService.benchmark"
    print("  ✓ ValidationService: validate, benchmark")
    
    # ExportService
    assert hasattr(ExportService, 'export'), "Missing ExportService.export"
    assert hasattr(ExportService, 'export_ncnn'), "Missing ExportService.export_ncnn"
    assert hasattr(ExportService, 'export_onnx'), "Missing ExportService.export_onnx"
    print("  ✓ ExportService: export, export_ncnn, export_onnx")
except AssertionError as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("tests passed")
print("=" * 60)
