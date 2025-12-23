#!/usr/bin/env python3
"""
Training script for custom detection architecture.
Trains ResNet backbone + PAN-FPN neck + Detect head.

Now integrates with the callbacks system for TensorBoard, W&B, and early stopping.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Layer imports
try:
    from service.layers.backbone import ResNetBackbone
    from service.layers.neck import PANFPN
    from service.layers.head import Detect, DensityHead
    from service.layers.losses import CIoULoss, BCEWithLogitsFocalLoss, DensityLoss
    from service.callbacks import (
        CallbackManager,
        TensorBoardCallback,
        WandBCallback,
        EarlyStoppingCallback,
        MetricsLoggerCallback,
        create_default_callbacks,
    )
except ImportError:
    from layers.backbone import ResNetBackbone
    from layers.neck import PANFPN
    from layers.head import Detect, DensityHead
    from layers.losses import CIoULoss, BCEWithLogitsFocalLoss, DensityLoss
    # Callbacks not available in fallback mode
    CallbackManager = None
    create_default_callbacks = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DetectionModel(nn.Module):
    """
    Full detection model: Backbone + Neck + Head.
    """
    def __init__(self, nc: int = 80, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        self.backbone = ResNetBackbone(backbone, pretrained=pretrained)
        self.neck = PANFPN(*self.backbone.out_channels)
        self.head = Detect(nc, self.backbone.out_channels)
        self.nc = nc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        fused = self.neck(*features)
        return self.head(fused)


class CountingModel(nn.Module):
    """
    Full counting model: Backbone + Neck + DensityHead.
    """
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        self.backbone = ResNetBackbone(backbone, pretrained=pretrained)
        self.neck = PANFPN(*self.backbone.out_channels)
        self.head = DensityHead(self.backbone.out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        fused = self.neck(*features)
        return self.head.count(fused)

class YOLODataset(Dataset):
    """
    YOLO format dataset loader.
    Expects directory structure:
        images/
            img1.jpg
            img2.jpg
        labels/
            img1.txt  (class x_center y_center width height per line)
            img2.txt
    """
    
    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 640, 
                 augment: bool = True, nc: int = 80):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        self.nc = nc
        
        # Collect image paths
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.images_dir.glob(ext)))
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        logger.info(f"Found {len(self.image_paths)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image."""
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # To tensor (HWC -> CHW, normalize)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img
    
    def _load_labels(self, img_path: Path) -> dict:
        """Load YOLO format labels."""
        label_path = self.labels_dir / (img_path.stem + '.txt')
        
        boxes = []
        classes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        classes.append(cls_id)
                        boxes.append([x_center, y_center, width, height])
        
        if len(boxes) == 0:
            # No objects - return empty tensors
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'classes': torch.zeros((0,), dtype=torch.long),
            }
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'classes': torch.tensor(classes, dtype=torch.long),
        }
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = self._load_image(img_path)
        targets = self._load_labels(img_path)
        return img, targets


def collate_fn(batch):
    """Custom collate for variable number of objects."""
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


def compute_loss(outputs: torch.Tensor, targets: list, model: nn.Module,
                 box_loss_fn, cls_loss_fn, device: torch.device) -> Tuple[torch.Tensor, dict]:
    """
    Compute detection loss with target assignment.
    
    Args:
        outputs: Model output tensor (bs, no, n_anchors) where no = nc + reg_max*4
        targets: List of dicts with 'boxes' and 'classes' per image
        model: The detection model (for nc and reg_max info)
        box_loss_fn: CIoU loss
        cls_loss_fn: BCE focal loss
        device: Target device
    """
    bs = len(targets)
    nc = model.nc
    
    # For training mode, outputs is list of feature maps
    # For inference mode, it's concatenated
    if isinstance(outputs, list):
        # Training mode - concatenate predictions
        outputs_cat = torch.cat([o.view(o.shape[0], o.shape[1], -1) for o in outputs], dim=2)
    else:
        outputs_cat = outputs
    
    total_box_loss = torch.tensor(0.0, device=device)
    total_cls_loss = torch.tensor(0.0, device=device)
    n_targets = 0
    
    for i, target in enumerate(targets):
        boxes = target['boxes'].to(device)  # (n, 4) xywh normalized
        classes = target['classes'].to(device)  # (n,)
        
        if len(boxes) == 0:
            continue
        
        n_targets += len(boxes)
        
        # Simplified target assignment: use prediction centers closest to GT centers
        # In production, use proper SimOTA or TAL assignment
        pred = outputs_cat[i]  # (no, n_anchors)
        n_anchors = pred.shape[1]
        
        # Create one-hot class targets
        cls_targets = torch.zeros(len(boxes), nc, device=device)
        for j, cls_id in enumerate(classes):
            if cls_id < nc:
                cls_targets[j, cls_id] = 1.0
        
        # Compute losses (simplified - using first n predictions as matched)
        # Real impl needs proper anchor-to-GT assignment
        n_matched = min(len(boxes), n_anchors)
        
        # Box regression loss (using first 64 channels for reg_max * 4)
        pred_boxes = pred[:64, :n_matched].T  # (n_matched, 64)
        # Convert to xywh format (simplified)
        pred_xywh = pred_boxes[:, :4]  # Just use first 4 for simplified loss
        total_box_loss += box_loss_fn(pred_xywh, boxes[:n_matched])
        
        # Classification loss
        pred_cls = pred[64:64+nc, :n_matched].T  # (n_matched, nc)
        total_cls_loss += cls_loss_fn(pred_cls, cls_targets[:n_matched])
    
    # Normalize by number of targets
    n_targets = max(1, n_targets)
    box_loss = total_box_loss / n_targets
    cls_loss = total_cls_loss / n_targets
    
    total_loss = 7.5 * box_loss + 0.5 * cls_loss
    
    return total_loss, {
        'box_loss': box_loss.item(),
        'cls_loss': cls_loss.item(),
        'total': total_loss.item(),
    }


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: optim.Optimizer, device: torch.device,
                box_loss_fn, cls_loss_fn) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_box = 0.0
    total_cls = 0.0
    
    for batch_idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(imgs)
        
        # Compute loss with target assignment
        loss, loss_dict = compute_loss(
            outputs, targets, model, box_loss_fn, cls_loss_fn, device
        )
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss_dict['total']
        total_box += loss_dict['box_loss']
        total_cls += loss_dict['cls_loss']
        
        if batch_idx % 10 == 0:
            logger.info(f"  Batch {batch_idx}/{len(dataloader)}, "
                       f"Loss: {loss_dict['total']:.4f} "
                       f"(box: {loss_dict['box_loss']:.4f}, cls: {loss_dict['cls_loss']:.4f})")
    
    n_batches = len(dataloader)
    return {
        'train_loss': total_loss / n_batches,
        'box_loss': total_box / n_batches,
        'cls_loss': total_cls / n_batches,
    }


def train(args):
    """Main training loop with callbacks support."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Model
    model = DetectionModel(
        nc=args.nc,
        backbone=args.backbone,
        pretrained=not args.scratch,
    ).to(device)
    
    logger.info(f"Model: {args.backbone} backbone, {args.nc} classes")
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {params:,}")
    
    # Dataset
    dataset = YOLODataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        img_size=args.imgsz,
        nc=args.nc,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss functions
    box_loss = CIoULoss()
    cls_loss = BCEWithLogitsFocalLoss()
    
    # Setup callbacks
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = None
    if create_default_callbacks is not None:
        callbacks = create_default_callbacks(
            output_dir=str(output_dir),
            tensorboard=args.tensorboard,
            wandb_project=args.wandb_project if hasattr(args, 'wandb_project') else None,
            early_stopping=args.early_stopping if hasattr(args, 'early_stopping') else False,
            early_stopping_patience=args.patience if hasattr(args, 'patience') else 10,
        )
        callbacks.on_train_start({"epochs": args.epochs, "nc": args.nc})
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        metrics = train_epoch(model, dataloader, optimizer, device, box_loss, cls_loss)
        metrics['epoch'] = epoch + 1
        metrics['lr'] = scheduler.get_last_lr()[0]
        scheduler.step()
        
        logger.info(f"Epoch {epoch + 1} metrics: {metrics}")
        
        # Call callbacks
        if callbacks:
            callbacks.on_epoch_end(metrics)
        
        # Save checkpoint
        if metrics['train_loss'] < best_loss:
            best_loss = metrics['train_loss']
            checkpoint_path = output_dir / 'best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
    
    # Finalize callbacks
    if callbacks:
        callbacks.on_train_end({"best_loss": best_loss, "epochs": args.epochs})
    
    logger.info("Training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train custom detection model')
    
    # Data
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--labels-dir', type=str, required=True,
                        help='Path to labels directory (YOLO format)')
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34'])
    parser.add_argument('--nc', type=int, default=80, help='Number of classes')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch')
    
    # Training
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/train', help='Project dir')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    
    # Callbacks
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--wandb-project', type=str, default=None, help='W&B project name')
    parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
