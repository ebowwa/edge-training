"""
Feature Fusion Neck: PAN-FPN (Path Aggregation Network + Feature Pyramid Network).
Used for multi-scale feature integration.
"""

import torch
import torch.nn as nn
from typing import List

try:
    from .common import Conv, C2f, Concat
except ImportError:
    from common import Conv, C2f, Concat


class PANFPN(nn.Module):
    """
    PAN-FPN feature fusion module.
    Fuses features from P3, P4, P5 scales of the backbone.
    
    Architecture:
    1. FPN (Top-down): semantic enrichment of high-res features.
    2. PAN (Bottom-up): spatial enrichment of high-semantic features.
    """
    def __init__(self, c_p3: int, c_p4: int, c_p5: int, n: int = 3, shortcut: bool = False):
        """
        Args:
            c_p3, c_p4, c_p5: Channels of input features from backbone (usually strides 8, 16, 32).
            n: Number of bottleneck layers in C2f.
            shortcut: Whether to use shortcut in C2f.
        """
        super().__init__()
        
        # --- TOP-DOWN (FPN) ---
        # P5 -> P4
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fpn_c2f_p4 = C2f(c_p5 + c_p4, c_p4, n, shortcut)
        
        # P4 -> P3
        self.fpn_c2f_p3 = C2f(c_p4 + c_p3, c_p3, n, shortcut)
        
        # --- BOTTOM-UP (PAN) ---
        # N3 -> N4
        self.downsample_n3 = Conv(c_p3, c_p3, 3, 2)  # Stride 2 conv for downsampling
        self.pan_c2f_n4 = C2f(c_p3 + c_p4, c_p4, n, shortcut)
        
        # N4 -> N5
        self.downsample_n4 = Conv(c_p4, c_p4, 3, 2)
        self.pan_c2f_n5 = C2f(c_p4 + c_p5, c_p5, n, shortcut)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> List[torch.Tensor]:
        """
        Input: [P3, P4, P5] features from backbone.
        Output: [N3, N4, N5] fused features for head prediction.
        """
        # --- Top-Down Pathway (FPN) ---
        # fpn_p4: Upsample P5 and concat with P4
        up_p5 = self.upsample(p5)
        fpn_p4 = self.fpn_c2f_p4(torch.cat([up_p5, p4], 1))
        
        # fpn_p3: Upsample fpn_p4 and concat with P3
        up_p4 = self.upsample(fpn_p4)
        n3 = self.fpn_c2f_p3(torch.cat([up_p4, p3], 1))  # Highest resolution output
        
        # --- Bottom-Up Pathway (PAN) ---
        # pan_n4: Downsample n3 and concat with fpn_p4
        dn_n3 = self.downsample_n3(n3)
        n4 = self.pan_c2f_n4(torch.cat([dn_n3, fpn_p4], 1))
        
        # pan_n5: Downsample n4 and concat with P5
        dn_n4 = self.downsample_n4(n4)
        n5 = self.pan_c2f_n5(torch.cat([dn_n4, p5], 1))  # Highest semantic output
        
        return [n3, n4, n5]
