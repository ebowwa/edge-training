"""Modal-specific configuration."""

from dataclasses import dataclass, field
from typing import Dict, Any
from ..base import GPUProviderConfig


@dataclass
class ModalConfig(GPUProviderConfig):
    """Configuration for Modal provider."""
    
    # GPU settings
    gpu_type: str = "A10G"  # T4, L4, A10G, A100
    gpu_count: int = 1
    
    # Container settings
    python_version: str = "3.11"
    timeout_seconds: int = 3600
    
    # Upload settings
    max_upload_retries: int = 3
    chunk_size_mb: int = 100
    upload_timeout_per_chunk: int = 300
    
    # Secrets (will reference Modal secrets)
    secret_names: list[str] = field(default_factory=list)
    
    # Custom pip packages
    pip_packages: list[str] = field(default_factory=list)
    
    # Environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass  
class ModalJobConfig:
    """Base config for Modal jobs."""
    app_name: str
    function_name: str
    gpu_config: ModalConfig
    args: Dict[str, Any] = field(default_factory=dict)
