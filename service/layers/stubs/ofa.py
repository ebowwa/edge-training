# TODO: PENDING INTEGRATION - Once-for-All networks not wired to training pipeline
# This module implements OFA elastic networks (arXiv:1908.09791)
# but is not currently used by any training code.
# Enable when hardware-aware NAS is needed.

"""
=== COMMENTED OUT: NOT INTEGRATED ===

The following code is complete but not wired into the training pipeline.
To enable:
1. Build supernet with ElasticKernel/ElasticWidth blocks
2. Train with progressive shrinking schedule
3. Extract sub-networks for target hardware

Original docstring:
Once-for-All (OFA) Networks Module.

Based on: "Once-for-All: Train One Network and Specialize it for Efficient
Deployment" (arXiv:1908.09791)

Key Techniques:
- Elastic kernel sizes (3, 5, 7)
- Elastic width (channel multipliers)
- Elastic depth (layer skipping)
- Progressive shrinking training
- Sub-network extraction without retraining

# Full implementation was ~400 lines
# See git history for complete code

=== END COMMENTED OUT CODE ===
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# Stub exports to prevent import errors
@dataclass
class OFAConfig:
    """Stub: OFA supernet config (not implemented)."""
    max_depth: List[int] = field(default_factory=lambda: [4, 4, 6, 4])
    supported_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    supported_width_multipliers: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    supported_resolutions: List[int] = field(default_factory=lambda: [128, 160, 192, 224])
    
    def __post_init__(self):
        raise NotImplementedError("OFA module not integrated. See TODO at top of file.")


@dataclass
class SubNetworkConfig:
    """Stub: Sub-network config (not implemented)."""
    kernel_sizes: List[int] = field(default_factory=list)
    depths: List[int] = field(default_factory=list)
    width_multiplier: float = 1.0
    resolution: int = 224


class ElasticKernel(nn.Module):
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("OFA module not integrated.")


class ElasticWidth(nn.Module):
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("OFA module not integrated.")


class ElasticBlock(nn.Module):
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("OFA module not integrated.")


class OFASubNetworkExtractor:
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("OFA module not integrated.")


def create_progressive_shrinking_schedule(config, total_epochs: int = 100) -> List[Dict]:
    """Stub: Not implemented."""
    raise NotImplementedError("OFA module not integrated. See TODO at top of file.")
