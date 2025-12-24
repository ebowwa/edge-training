# TODO: PENDING INTEGRATION - Knowledge distillation not wired to training pipeline
# This module implements patient knowledge distillation (arXiv:2012.06785)
# but is not currently used by any training code.
# Enable when teacher-student training is implemented.

"""
=== COMMENTED OUT: NOT INTEGRATED ===

The following code is complete but not wired into the training pipeline.
To enable:
1. Add distillation mode to training_service.py
2. Create teacher/student model pairs
3. Wire up DistillationLoss in training loop

Original docstring:
Knowledge Distillation Module.

Based on: "Patient Knowledge Distillation for BERT Model Compression"
(arXiv:2012.06785) and general distillation literature.

Key Techniques:
- Soft label distillation with temperature scaling
- Patient (multi-layer) feature matching
- PKD-Last: distill from last K teacher layers
- PKD-Skip: distill from every K-th teacher layer

# Full implementation was ~420 lines
# See git history for complete code

=== END COMMENTED OUT CODE ===
"""

from enum import Enum
from dataclasses import dataclass


# Stub exports to prevent import errors
class DistillationStrategy(Enum):
    """Stub: Distillation strategies (not implemented)."""
    RESPONSE = "response"
    FEATURE = "feature"
    PATIENT = "patient"
    ATTENTION = "attention"


@dataclass
class DistillationConfig:
    """Stub: Distillation config (not implemented)."""
    temperature: float = 4.0
    alpha: float = 0.5
    strategy: DistillationStrategy = DistillationStrategy.PATIENT
    
    def __post_init__(self):
        raise NotImplementedError("Distillation module not integrated. See TODO at top of file.")


class SoftLabelLoss:
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Distillation module not integrated.")


class FeatureMatchingLoss:
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Distillation module not integrated.")


class PatientDistillationLoss:
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Distillation module not integrated.")


class AttentionDistillationLoss:
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Distillation module not integrated.")


class DistillationLoss:
    """Stub: Not implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Distillation module not integrated.")


def create_distillation_trainer(*args, **kwargs):
    """Stub: Not implemented."""
    raise NotImplementedError("Distillation module not integrated.")
