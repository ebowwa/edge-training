# YOLO Training & Inference Service

A modular, three-layer ML service for training, validating, and deploying YOLOv8 models. This service provides a robust API for dataset preparation, automated preprocessing, model training, and low-latency inference.

## 🏗 Architecture

The project follows a strict three-layer architecture to ensure testability and separation of concerns:

- **Service Layer (`service/`)**: Pure business logic. Handles YOLO training (via Ultralytics), inference operations, and dataset structure detection.
- **API Layer (`api/`)**: HTTP endpoints, request/response validation using Pydantic schemas, and serialization.
- **Server Layer (`server/`)**: Application bootstrap, FastAPI configuration, CORS middleware, and Uvicorn entrypoint.

For more details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 🚀 Key Features

- **Automated Dataset Prep**: Integration with Kaggle for dataset downloads and automatic YOLO structure detection.
- **Preprocessing Pipeline**: Composable cleaners (corrupted image detection, bbox validation) and transforms (augmentation).
- **Training Management**: Synchronous and resumable training with custom weight support.
- **Inference API**: High-performance image inference with configurable confidence and IOU thresholds.
- **Deployment-Ready Exports**: Export trained models to NCNN, ONNX, CoreML, and TFLite formats.

## 🛠 Setup

### Prerequisites
- Python 3.9+
- CUDA-compatible environment (optional, but recommended for training)

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   cd yolo-training
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Running the Service

Start the FastAPI server:
```bash
python3 server/main.py
```

The API will be available at `http://localhost:8000`.
- **Interactive Docs (Swagger)**: `http://localhost:8000/docs`
- **Alternative Docs (Redoc)**: `http://localhost:8000/redoc`

## 🧪 Testing

The project includes both standalone and `pytest`-based tests:

### API & Schema Tests
```bash
# Run standalone API schema tests
python3 api/tests/run_tests.py

# Run all API tests with pytest
pytest api/tests/ -v
```

### Service Smoke Tests
```bash
# Verify core service functionality
python3 service/tests/test_services.py
```

### Preprocessing Tests
```bash
# Test the preprocessing pipeline
python3 service/preprocessing/tests/test_preprocessing.py
```

## 📂 Project Structure

```text
yolo-training/
├── api/            # HTTP Layer (FastAPI)
│   ├── routes.py   # API Endpoints
│   ├── schemas.py  # Pydantic Models
│   └── tests/      # API Test Suite
├── server/         # Bootstrap Layer
│   └── main.py     # Entry point
├── service/        # Core Logic Layer
│   ├── preprocessing/ # Data Cleaning & Augmentation
│   ├── training_service.py
│   ├── inference_service.py
│   └── tests/      # Service Test Suite
└── requirements.txt
```
