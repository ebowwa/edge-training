"""
RunPod Configuration

Supports Doppler: doppler run -- python your_script.py
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from integrations.base import GPUProviderConfig


@dataclass
class RunPodConfig(GPUProviderConfig):
    """
    RunPod config. API key auto-reads from RUNPOD_API_KEY env var.
    
    Example:
        # With Doppler (auto-reads env)
        config = RunPodConfig(training_endpoint_id="abc123")
        
        # Or explicit
        config = RunPodConfig(api_key="xxx", training_endpoint_id="abc123")
    """
    api_key: str = ""
    training_endpoint_id: Optional[str] = None
    inference_endpoint_id: Optional[str] = None
    webhook_url: Optional[str] = None
    s3_config: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY env var or api_key required")
