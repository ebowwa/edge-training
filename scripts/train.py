#!/usr/bin/env python3
"""
Training script for custom detection architecture.
Trains ResNet backbone + PAN-FPN neck + Detect head.
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
except ImportError:
    from layers.backbone import ResNetBackbone
    from layers.neck import PANFPN
    from layers.head import Detect, DensityHead
    from layers.losses import CIoULoss, BCEWithLogitsFocalLoss, DensityLoss

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


class DummyDataset(Dataset):
    """Placeholder dataset for testing training loop."""
    def __init__(self, size: int = 100, img_size: int = 640, nc: int = 80):
        self.size = size
        self.img_size = img_size
        self.nc = nc
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random image
        img = torch.randn(3, self.img_size, self.img_size)
        # Random targets (simplified)
        n_objs = torch.randint(1, 10, (1,)).item()
        boxes = torch.rand(n_objs, 4)  # xywh normalized
        classes = torch.randint(0, self.nc, (n_objs,))
        return img, {'boxes': boxes, 'classes': classes}


def collate_fn(batch):
    """Custom collate for variable number of objects."""
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: optim.Optimizer, device: torch.device,
                box_loss_fn, cls_loss_fn) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(imgs)
        
        # Simplified loss (real impl needs target assignment)
        # This is a placeholder to test the training loop
        if model.training:
            # Use a simple loss for testing
            loss = outputs.abs().mean()  # Placeholder
        else:
            loss = outputs.abs().mean()
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logger.info(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return {'train_loss': total_loss / len(dataloader)}


def train(args):
    """Main training loop."""
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
    dataset = DummyDataset(size=args.samples, img_size=args.imgsz, nc=args.nc)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss functions
    box_loss = CIoULoss()
    cls_loss = BCEWithLogitsFocalLoss()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        metrics = train_epoch(model, dataloader, optimizer, device, box_loss, cls_loss)
        scheduler.step()
        
        logger.info(f"Epoch {epoch + 1} metrics: {metrics}")
        
        # Save checkpoint
        if metrics['train_loss'] < best_loss:
            best_loss = metrics['train_loss']
            checkpoint_path = Path(args.project) / args.name / 'best.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
    
    logger.info("Training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train custom detection model')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34'])
    parser.add_argument('--nc', type=int, default=80, help='Number of classes')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--project', type=str, default='runs/train', help='Project dir')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch')
    parser.add_argument('--samples', type=int, default=100, help='Dummy dataset size')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
