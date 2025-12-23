"""
YOLO Training API

FastAPI routes for training, inference, validation, and export.

Example:
    from api import router
    app = FastAPI()
    app.include_router(router)
"""

from routes import router
from schemas import (
    DatasetRequest,
    TrainingRequest,
    InferenceRequest,
    ValidationRequest,
    ExportRequest,
    TrainingResponse,
    InferenceResponse,
    ValidationResponse,
)

__all__ = [
    "router",
    "DatasetRequest",
    "TrainingRequest",
    "InferenceRequest",
    "ValidationRequest",
    "ExportRequest",
    "TrainingResponse",
    "InferenceResponse",
    "ValidationResponse",
]
