"""
FastAPI dependency functions for route handlers.

Provides cached service instances using FastAPI's dependency injection.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from service.inference_service import InferenceService
from service.slam.slam_service import SlamService


@lru_cache(maxsize=32)
def get_inference_service(model_path: str) -> InferenceService:
    """
    Get or create a cached InferenceService for the given model path.

    Caches up to 32 unique model paths. Cache is per-process.
    """
    return InferenceService(model_path)


def get_slam_service(
    request: Annotated[object, "starlette.requests.Request"]
) -> SlamService:
    """
    Get the SLAM service from app.state (managed by lifespan).

    The SLAM service is initialized once at startup and shared across requests.
    """
    return request.app.state.slam
