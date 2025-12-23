"""
Main entry point for the YOLO Training API server.
"""

import sys
import os
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Resolve paths
_server_dir = Path(__file__).parent
_root_dir = _server_dir.parent
_api_dir = _root_dir / "api"
_service_dir = _root_dir / "service"

# Add directories to path for imports
sys.path.insert(0, str(_root_dir))
sys.path.insert(0, str(_api_dir))
sys.path.insert(0, str(_service_dir))

from api import router as api_router

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="YOLO Training API",
        description="API for managing YOLO model training, inference, and deployment.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/")
    async def root():
        """Root endpoint redirecting to docs."""
        return {
            "message": "YOLO Training API",
            "docs": "/docs",
            "health": "/api/v1/health"
        }

    return app

app = create_app()

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting YOLO Training API on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
