"""
Detection Heads for YOLO-style architectures.
Supports anchor-free detection with LTRB (Left, Top, Right, Bottom) regression.
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple

try:
    from .common import Conv
except ImportError:
    from common import Conv


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss: https://arxiv.org/abs/2006.04388
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """
        Input shape: (bs, 4*c1, n_anchors)
        Output shape: (bs, 4, n_anchors)
        """
        b, c, a = x.shape
        # Softmax over the distribution, then integral
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    """
    YOLOv8 Style Decoupled Detection Head.
    Predicts:
    1. Classification (Cls): Probability per class.
    2. Regression (Box): Distance to edges (LTRB) using DFL.
    """
    def __init__(self, nc=80, ch=(128, 256, 512)):
        """
        Args:
            nc: Number of classes.
            ch: Input channels from neck (P3, P4, P5).
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels per edge
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        # Build Cls and Box branches for each scale
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate predictions from all scales.
        Each scale i has shape (bs, no, hi, wi).
        """
        shape = x[0].shape  # (bs, c, h, w)
        for i in range(self.nl):
            # Apply cls and box heads
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        if self.training:
            return x
            
        # Inference mode: concatenate and return formatted tensor
        # (bs, no, hi*wi + ...)
        x = [xi.view(shape[0], self.no, -1) for xi in x]
        return torch.cat(x, 2)

    def decode_bboxes(self, bboxes, anchors):
        """
        Decode LTRB (distances) to XYXY (coordinates).
        """
        return dist2bbox(bboxes, anchors, xywh=False, dim=1)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance (ltrb) to box (xywh or xyxy)."""
    lt, rb = torch.chunk(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)
