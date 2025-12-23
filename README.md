# YOLO Training & Inference Service

A modular, three-layer ML service for training, validating, and deploying YOLO models. Supports **YOLOv5, v8, v9, v10, v11, and YOLO-World** via the Ultralytics library. Includes a composable spatial inference pipeline for real-time detection on wearables.

## 🏗 Architecture

The project follows a strict three-layer architecture to ensure testability and separation of concerns:

- **Service Layer (`service/`)**: Pure business logic. Handles YOLO training (via Ultralytics), inference operations, SLAM spatial tracking, and dataset structure detection.
- **API Layer (`api/`)**: HTTP endpoints, request/response validation using Pydantic schemas, and serialization.
- **Server Layer (`server/`)**: Application bootstrap, FastAPI configuration, CORS middleware, and Uvicorn entrypoint.
- **Pipeline Layer (`pipeline/`)**: Composable, chainable pipelines for real-time inference with spatial awareness.

For more details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 🚀 Key Features

- **Multi-YOLO Support**: Works with YOLOv5, v8, v9, v10, v11, and YOLO-World models.
- **Automated Dataset Prep**: Integration with Kaggle for dataset downloads and automatic YOLO structure detection.
- **Preprocessing Pipeline**: Composable cleaners (corrupted image detection, bbox validation) and transforms (augmentation).
- **Training Management**: Synchronous and resumable training with custom weight support.
- **Inference API**: High-performance image inference with configurable confidence and IOU thresholds.
- **Spatial Inference (SLAM)**: Chain YOLO inference with lightweight SLAM for egocentric video from glasses/phones.
- **Deployment-Ready Exports**: Export trained models to NCNN, ONNX, CoreML, and TFLite formats.

## 🛠 Setup

### Prerequisites
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA-compatible environment (optional, but recommended for training)

### Installation with uv (Recommended)
```bash
cd yolo-training

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Installation with pip (Alternative)
```bash
cd yolo-training
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
    model_path="weights/yolov8m.pt",  # or yolov11, yolov5, etc.
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
yolo-training/
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
