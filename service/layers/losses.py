"""
Loss functions for object detection training.
Includes CIoU loss for bounding box regression and focal loss for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, 
             GIoU: bool = False, DIoU: bool = False, CIoU: bool = False, 
             eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate IoU between box1 and box2.
    
    Args:
        box1: (N, 4) tensor of boxes
        box2: (M, 4) tensor of boxes
        xywh: If True, boxes are (x, y, w, h), else (x1, y1, x2, y2)
        GIoU, DIoU, CIoU: Use respective IoU variants
        eps: Small value for numerical stability
    
    Returns:
        IoU tensor of shape (N, M) or (N,) if boxes aligned
    """
    # Convert xywh to xyxy
    if xywh:
        x1, y1, w1, h1 = box1.chunk(4, dim=-1)
        x2, y2, w2, h2 = box2.chunk(4, dim=-1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if CIoU or DIoU or GIoU:
        # Convex (smallest enclosing box) width/height
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        
        if CIoU or DIoU:
            # Diagonal distance squared
            c2 = cw ** 2 + ch ** 2 + eps
            # Center distance squared
            rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 +
                    (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
            if CIoU:
                # Aspect ratio consistency
                v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2  # DIoU
        # GIoU
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    
    return iou


class CIoULoss(nn.Module):
    """Complete IoU Loss for bounding box regression."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted boxes (N, 4) in xywh format
            target: Target boxes (N, 4) in xywh format
        """
        iou = bbox_iou(pred, target, xywh=True, CIoU=True)
        loss = 1.0 - iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for classification.
    Addresses class imbalance by down-weighting easy examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C) logits
            target: (N,) class indices
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BCEWithLogitsFocalLoss(nn.Module):
    """
    BCE with logits + focal weighting for multi-label classification.
    Used in YOLO-style detection heads.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C) logits
            target: (N, C) binary targets
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DetectionLoss(nn.Module):
    """
    Combined loss for object detection.
    - Box loss: CIoU
    - Class loss: BCE with focal weighting
    - Objectness loss: BCE
    """
    
    def __init__(self, nc: int = 80, box_weight: float = 7.5, 
                 cls_weight: float = 0.5, obj_weight: float = 1.0):
        super().__init__()
        self.nc = nc
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        
        self.ciou_loss = CIoULoss(reduction='mean')
        self.cls_loss = BCEWithLogitsFocalLoss(reduction='mean')
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(self, pred: torch.Tensor, targets: dict) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: Detection head output
            targets: Dict with 'boxes', 'classes', 'obj_mask' keys
        
        Returns:
            total_loss, loss_dict
        """
        # Placeholder - full implementation requires target assignment
        # This shows the structure
        loss_box = self.ciou_loss(pred[..., :4], targets['boxes'])
        loss_cls = self.cls_loss(pred[..., 4:4+self.nc], targets['classes'])
        loss_obj = self.obj_loss(pred[..., -1], targets['obj_mask'])
        
        total = (self.box_weight * loss_box + 
                 self.cls_weight * loss_cls + 
                 self.obj_weight * loss_obj)
        
        return total, {
            'box_loss': loss_box.item(),
            'cls_loss': loss_cls.item(),
            'obj_loss': loss_obj.item(),
            'total': total.item(),
        }


class DensityLoss(nn.Module):
    """
    Loss for density map counting.
    Uses MSE on density maps.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred_density: torch.Tensor, gt_density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_density: Predicted density map (B, 1, H, W)
            gt_density: Ground truth density map (B, 1, H, W)
        """
        return self.mse(pred_density, gt_density)


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for training student models.
    Based on: https://arxiv.org/abs/2307.07483 (Multimodal Distillation)
    
    Combines:
    - Soft targets (KL divergence with temperature)
    - Hard targets (standard task loss)
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        Args:
            temperature: Softmax temperature for soft targets (higher = softer)
            alpha: Weight for distillation loss (1-alpha for hard target loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                hard_targets: Optional[torch.Tensor] = None,
                hard_loss_fn: Optional[nn.Module] = None) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            student_logits: Student model output (B, C, ...)
            teacher_logits: Teacher model output (B, C, ...) - detached
            hard_targets: Optional ground truth labels
            hard_loss_fn: Optional loss function for hard targets
        
        Returns:
            total_loss, loss_dict
        """
        T = self.temperature
        
        # Flatten spatial dimensions if needed
        if student_logits.dim() > 2:
            student_flat = student_logits.view(student_logits.size(0), student_logits.size(1), -1)
            teacher_flat = teacher_logits.view(teacher_logits.size(0), teacher_logits.size(1), -1)
        else:
            student_flat = student_logits
            teacher_flat = teacher_logits
        
        # Soft targets: KL divergence with temperature
        soft_student = F.log_softmax(student_flat / T, dim=1)
        soft_teacher = F.softmax(teacher_flat.detach() / T, dim=1)
        
        distill_loss = F.kl_div(
            soft_student, soft_teacher, 
            reduction='batchmean'
        ) * (T ** 2)  # Scale by T^2 as per Hinton et al.
        
        # Optional hard targets
        if hard_targets is not None and hard_loss_fn is not None:
            hard_loss = hard_loss_fn(student_logits, hard_targets)
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
            return total_loss, {
                'distill_loss': distill_loss.item(),
                'hard_loss': hard_loss.item(),
                'total': total_loss.item(),
            }
        
        return distill_loss, {
            'distill_loss': distill_loss.item(),
            'total': distill_loss.item(),
        }


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation for intermediate representations.
    Useful for transferring spatial knowledge from teacher to student.
    """
    
    def __init__(self, student_channels: List[int], teacher_channels: List[int]):
        """
        Args:
            student_channels: Channel dims for student features [P3, P4, P5]
            teacher_channels: Channel dims for teacher features [P3, P4, P5]
        """
        super().__init__()
        # Projection layers to align student features to teacher
        self.projections = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, 1) if s_ch != t_ch else nn.Identity()
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])
    
    def forward(self, student_features: List[torch.Tensor], 
                teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            student_features: List of student feature maps [P3, P4, P5]
            teacher_features: List of teacher feature maps [P3, P4, P5]
        """
        total_loss = 0.0
        for proj, s_feat, t_feat in zip(self.projections, student_features, teacher_features):
            # Project student to teacher dimension
            s_projected = proj(s_feat)
            # L2 loss on normalized features
            s_norm = F.normalize(s_projected, dim=1)
            t_norm = F.normalize(t_feat.detach(), dim=1)
            total_loss += F.mse_loss(s_norm, t_norm)
        
        return total_loss / len(self.projections)
