"""
Roboflow Configuration

Supports Doppler: doppler run -- python script.py
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RoboflowConfig:
    """
    Roboflow config. API key auto-reads from ROBOFLOW_API_KEY env var.
    
    Example:
        config = RoboflowConfig(model_id="usd-classification/1")
    """
    api_key: str = ""
    api_url: str = "https://serverless.roboflow.com"
    model_id: Optional[str] = None

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY env var or api_key required")
