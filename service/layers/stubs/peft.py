# TODO: PENDING INTEGRATION - PEFT/LoRA not wired to training pipeline
# This module implements parameter-efficient fine-tuning techniques
# but is not currently used by any training code.
# Enable when few-shot personalization is needed.

"""
=== COMMENTED OUT: NOT INTEGRATED ===

The following code is complete but not wired into the training pipeline.
To enable:
1. Add PEFT mode to training_service.py
2. Inject LoRA/BitFit adapters into model layers
3. Freeze base model, train only adapters

Original docstring:
Parameter-Efficient Fine-Tuning (PEFT) Module.

Implements LoRA (Low-Rank Adaptation) and BitFit for efficient
fine-tuning with minimal trainable parameters.

Key Techniques:
- LoRA: Low-rank decomposition of weight updates
- BitFit: Train only bias terms
- Adapter layers: Small bottleneck modules

References:
- LoRA: https://arxiv.org/abs/2106.09685
- BitFit: https://arxiv.org/abs/2106.10199

# Full implementation was ~200 lines
# See git history for complete code

=== END COMMENTED OUT CODE ===
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


# Stub exports to prevent import errors
@dataclass
class LoRAConfig:
    """Stub: LoRA config (not implemented)."""
    r: int = 8  # Rank of low-rank matrices
    alpha: float = 16.0  # Scaling factor
    dropout: float = 0.1
    target_modules: List[str] = None  # Which modules to apply LoRA to
    
    def __post_init__(self):
        raise NotImplementedError("PEFT module not integrated. See TODO at top of file.")


class LoRALinear(nn.Module):
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("PEFT module not integrated.")


class LoRAConv2d(nn.Module):
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("PEFT module not integrated.")


def apply_lora(model: nn.Module, config: Optional['LoRAConfig'] = None, 
               target_modules: Optional[List[str]] = None) -> nn.Module:
    """Stub: Not implemented."""
    raise NotImplementedError("PEFT module not integrated. See TODO at top of file.")


def apply_bitfit(model: nn.Module) -> nn.Module:
    """Stub: Not implemented - freezes all params except biases."""
    raise NotImplementedError("PEFT module not integrated. See TODO at top of file.")


def get_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters (this one actually works)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters in a model (this one actually works)."""
    for param in model.parameters():
        param.requires_grad = False
