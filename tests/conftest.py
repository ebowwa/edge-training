"""
Pytest configuration and fixtures for edge-training tests.
"""

import sys
from pathlib import Path

# Add project root to Python path for direct imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add service directory to path to allow importing callbacks directly
service_path = project_root / "service"
sys.path.insert(0, str(service_path))
