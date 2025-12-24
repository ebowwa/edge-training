# Edge Training & Inference Platform

A modular, multi-backend ML platform for training, validating, and deploying object detection models on edge devices. Supports **YOLO (v5, v8, v9, v10, v11, YOLO-World)** and **RF-DETR** (transformer-based SOTA). Includes composable spatial inference pipelines for real-time detection on wearables.

## 🌐 Hybrid AI: Cloud + Local Edge

This system is designed to bridge the gap between high-power cloud computing and low-latency edge devices (glasses, phones, IoT).

- **Cloud/Server Role (The Brain)**: 
  - **Heavy Training**: Run multi-GPU training jobs for large backbone models.
  - **High-Accuracy Inference**: Serve complex models via the FastAPI `/api/v1/infer` endpoint for remote devices.
  - **PEFT Training** *(TODO)*: Stub for LoRA/BitFit fine-tuning - not yet integrated.
- **Local Edge Role (The Reflex)**:
  - **Lightweight SLAM**: Track 3D world coordinates and camera pose locally without internet latency.
  - **Quantized Export**: Use the `ExportService` to shrink models into **NCNN, TFLite, or CoreML** for 8-bit on-device execution.
  - **Spatial Anchoring**: "Lock" cloud detections to the 3D world so they remain stable even when the user looks away.

## 🏗 Architecture

The project follows a strict three-layer architecture to ensure testability and separation of concerns:

- **Service Layer (`service/`)**: Pure business logic. Handles YOLO training (via Ultralytics), inference operations, SLAM spatial tracking, and dataset structure detection.
- **API Layer (`api/`)**: HTTP endpoints, request/response validation using Pydantic schemas, and serialization.
- **Server Layer (`server/`)**: Application bootstrap, FastAPI configuration, CORS middleware, and Uvicorn entrypoint.
- **Pipeline Layer (`pipeline/`)**: Composable, chainable pipelines for real-time inference with spatial awareness.

For more details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 🚀 Key Features

- **Multi-Backend Support**: YOLO family (via Ultralytics) and RF-DETR (transformer detection).
- **2D Spatial Tracking**: Optical flow-based camera motion estimation for wearable devices (glasses, phones).
- **Automated Dataset Prep**: Integration with Kaggle for dataset downloads and automatic YOLO structure detection.
- **Preprocessing Pipeline**: Composable cleaners (corrupted image detection, bbox validation) and transforms (augmentation).
- **Deployment-Ready Exports**: Export trained models to NCNN, ONNX, CoreML, and TFLite formats for edge hardware.
- **Inference Optimization**: JIT tracing for up to 2x speedup on PyTorch models.
- **Training Callbacks**: TensorBoard, W&B logging, and early stopping built-in.
- **MobileNet Convolutions**: Depthwise separable convolutions for ~8x computational savings.

## 🛠 Setup

### Prerequisites
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA-compatible environment (optional, but recommended for training)

### Installation with uv (Recommended)
```bash
cd edge-training

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Installation with pip (Alternative)
```bash
cd edge-training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🚦 Running the Service

Start the FastAPI server:
```bash
python server/main.py
```

The API will be available at `http://localhost:8000`.
- **Interactive Docs (Swagger)**: `http://localhost:8000/docs`
- **Alternative Docs (Redoc)**: `http://localhost:8000/redoc`

## 🔗 Composable Pipelines

The `pipeline/` module provides chainable, attachable pipelines for real-time inference:

```python
from pipeline import SpatialInferencePipeline, AttachablePipeline, FunctionStage

# High-level: YOLO + SLAM spatial detection
pipeline = SpatialInferencePipeline(
    model_path="resources/yolov8m.pt",  # or yolov11, yolov5, etc.
    enable_slam=True,
    imu_enabled=True,
)

for frame in video_stream:
    result = pipeline.process_frame(frame, imu_data=sensor_data)
    for det in result.detections:
        print(f"{det.class_name} at {det.spatial_coords}")

# Low-level: Compose custom pipelines
pipeline = AttachablePipeline()
pipeline.attach("normalize", FunctionStage("norm", lambda x, ctx: x / 255.0))
pipeline.attach("inference", InferenceStage("model.pt"))
# Dynamically attach/detach stages at runtime
```

## 🧪 Testing

The project includes both standalone and `pytest`-based tests:

### All Tests
```bash
# Run all tests
python service/slam/tests/test_slam_service.py
python pipeline/tests/test_base.py
python pipeline/tests/test_spatial_inference.py
python api/tests/run_tests.py
```

### With pytest
```bash
pytest api/tests/ -v
pytest service/preprocessing/tests/ -v
```

## 📂 Project Structure

```text
edge-training/
├── api/                # HTTP Layer (FastAPI)
│   ├── routes.py       # API Endpoints
│   ├── schemas.py      # Pydantic Models
│   └── tests/          # API Test Suite
├── server/             # Bootstrap Layer
│   └── main.py         # Entry point
├── service/            # Core Logic Layer
│   ├── preprocessing/  # Data Cleaning & Augmentation
│   ├── slam/           # Spatial tracking (SLAM)
│   ├── training_service.py
│   ├── inference_service.py
│   └── tests/
├── pipeline/           # Composable Pipelines
│   ├── base.py         # Pipeline abstractions
│   ├── spatial_inference.py  # YOLO + SLAM pipeline
│   └── tests/
└── requirements.txt
```
