"""
Parameter-Efficient Fine-Tuning (PEFT) wrappers.
Implements LoRA, BitFit, and adapter patterns for memory-efficient training.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Union


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear.
    Freezes original weights and adds trainable low-rank decomposition.
    
    Based on: https://arxiv.org/abs/2106.09685
    """
    
    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        """
        Args:
            original: The linear layer to wrap.
            rank: Rank of the low-rank decomposition.
            alpha: Scaling factor for LoRA updates.
        """
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original.in_features
        out_features = original.out_features
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
        
        # Low-rank trainable matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original(x)
        # LoRA delta: x @ A^T @ B^T * scaling
        lora_delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_delta
    
    def merge(self) -> nn.Linear:
        """Merge LoRA weights into original for inference."""
        with torch.no_grad():
            merged = nn.Linear(
                self.original.in_features, 
                self.original.out_features,
                bias=self.original.bias is not None
            )
            merged.weight.copy_(
                self.original.weight + (self.lora_B @ self.lora_A) * self.scaling
            )
            if self.original.bias is not None:
                merged.bias.copy_(self.original.bias)
        return merged


class Adapter(nn.Module):
    """
    Adapter layer: bottleneck MLP inserted after attention/FFN.
    
    Based on: https://arxiv.org/abs/1902.00751
    """
    
    def __init__(self, dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)
        
        # Initialize up projection to near-zero for residual identity start
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


def apply_lora(
    module: nn.Module, 
    rank: int = 8, 
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None
) -> int:
    """
    Apply LoRA to all Linear layers in a module.
    
    Args:
        module: The module to modify.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: Optional list of module name patterns to target.
                       If None, applies to all Linear layers.
    
    Returns:
        Number of layers wrapped with LoRA.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if target_modules is None or any(t in name for t in target_modules):
                setattr(module, name, LoRALinear(child, rank, alpha))
                count += 1
        else:
            count += apply_lora(child, rank, alpha, target_modules)
    return count


def apply_bitfit(module: nn.Module) -> int:
    """
    Apply BitFit: freeze all weights, only train biases.
    
    Based on: https://arxiv.org/abs/2106.10199
    
    Args:
        module: The module to modify.
    
    Returns:
        Number of bias parameters left trainable.
    """
    trainable_params = 0
    for name, param in module.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    return trainable_params


def freeze_except(module: nn.Module, patterns: List[str]) -> int:
    """
    Freeze all parameters except those matching patterns.
    
    Args:
        module: The module to modify.
        patterns: List of patterns to keep trainable.
    
    Returns:
        Number of trainable parameters.
    """
    trainable = 0
    for name, param in module.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    return trainable


def count_parameters(module: nn.Module) -> dict:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_pct': 100 * trainable / max(1, total),
    }


def get_trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Get list of trainable parameters for optimizer."""
    return [p for p in module.parameters() if p.requires_grad]


class GradientCheckpointWrapper(nn.Module):
    """
    Wrapper to apply gradient checkpointing to a module.
    Trades compute for memory during training.
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self.module, *args, use_reentrant=False, **kwargs
            )
        return self.module(*args, **kwargs)
