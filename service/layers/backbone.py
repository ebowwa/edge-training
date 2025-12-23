"""
ResNet backbones for detection.
Wraps torchvision models to extract multi-scale feature maps.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class ResNetBackbone(nn.Module):
    """
    Wrapper for ResNet backbones to extract P3, P4, P5 features.
    
    P3: Stride 8 (layer1 output)
    P4: Stride 16 (layer2 output)
    P5: Stride 32 (layer3 output)
    """
    def __init__(self, variant: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        
        if variant == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            self.channels = [64, 128, 256, 512]  # channels for [layer1, layer2, layer3, layer4]
        elif variant == 'resnet34':
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
            self.channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        # Components
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # stride 4
        self.layer2 = backbone.layer2  # stride 8 (P3)
        self.layer3 = backbone.layer3  # stride 16 (P4)
        self.layer4 = backbone.layer4  # stride 32 (P5)
        
        # Define P3, P4, P5 channel counts for external compatibility
        self.out_channels = [self.channels[1], self.channels[2], self.channels[3]]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features.
        """
        x = self.stem(x)
        x = self.layer1(x)  # stride 4
        p3 = self.layer2(x) # stride 8
        p4 = self.layer3(p3) # stride 16
        p5 = self.layer4(p4) # stride 32
        
        return [p3, p4, p5]
