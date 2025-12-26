"""
Hetzner Cloud Configuration

API key auto-reads from HETZNER_API_TOKEN env var or from config.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from integrations.base import GPUProviderConfig


@dataclass
class HetznerConfig(GPUProviderConfig):
    """
    Hetzner Cloud GPU configuration.

    Example:
        # With env var (HETZNER_API_TOKEN)
        config = HetznerConfig(region="fsn1")

        # Or explicit
        config = HetznerConfig(api_key="abc123...", region="fsn1")
    """
    api_key: str = ""
    region: str = "fsn1"  # Falkenstein (default) or "hil" (Hilversum)
    ssh_keys: List[int] = field(default_factory=list)  # SSH key IDs from Hetzner
    user_data: Optional[str] = None  # Cloud-init user data script

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("HETZNER_API_TOKEN", "")
        if not self.api_key:
            raise ValueError("HETZNER_API_TOKEN env var or api_key required")

    # Valid Hetzner regions
    VALID_REGIONS = ["fsn1", "hil", "nbg1", "ash"]

    def __setattr__(self, name, value):
        if name == "region" and value and value not in self.VALID_REGIONS:
            raise ValueError(f"Invalid region: {value}. Valid: {self.VALID_REGIONS}")
        super().__setattr__(name, value)
