# Edge Optimization Guide

This guide documents the edge optimization techniques integrated into the edge-training repository, based on cutting-edge research papers.

## Overview

The edge-training platform now includes advanced optimization modules for deploying efficient models on edge devices (glasses, phones, IoT):

| Module | Technique | arXiv Paper | Status |
|--------|-----------|-------------|--------|
| `depthwise_separable` | MobileNet Convolutions | [1704.04861](https://arxiv.org/abs/1704.04861) | ✅ Implemented |
| `pruning` | Attention Head Pruning | [2201.08071](https://arxiv.org/abs/2201.08071) | ⚠️ TODO |
| `ofa` | Once-for-All Networks | [1908.09791](https://arxiv.org/abs/1908.09791) | ⚠️ TODO |
| `distillation` | Patient Knowledge Distillation | [2012.06785](https://arxiv.org/abs/2012.06785) | ⚠️ TODO |

Additional relevant research:
- **RT-DETR** ([2304.08069](https://arxiv.org/abs/2304.08069)) - Already integrated in `rfdetr_service.py`
- **Deformable DETR** ([2010.04159](https://arxiv.org/abs/2010.04159)) - Deformable attention patterns

---

<!-- 
## Attention Head Pruning [NOT IMPLEMENTED]

TODO: The following section documents a planned feature. The pruning.py module
has not been created yet. Uncomment when implemented.

> Based on "Layer-wise Pruning of Transformer Attention Heads for Efficient Language Modeling" (arXiv:2201.08071)

Remove redundant attention heads from transformer models like RT-DETR to reduce computation without significant accuracy loss.

### Quick Start

```python
from service.layers import (
    PrunedMultiHeadAttention,
    AttentionHeadPruner,
    PruningConfig,
    apply_head_pruning,
)

# Replace standard attention with prunable attention
attention = PrunedMultiHeadAttention(
    embed_dim=512,
    num_heads=8,
)

# Configure pruning
config = PruningConfig(
    sparsity=0.3,  # Prune 30% of heads
    strategy="importance",
    use_gradients=True,
)

# Prune the model
pruner = AttentionHeadPruner(model, config)
pruned_heads = pruner.prune()

print(pruner.get_stats())
# {'total_heads': 48, 'active_heads': 34, 'sparsity': 0.29}
```

### Key Classes

| Class | Description |
|-------|-------------|
| `PrunedMultiHeadAttention` | Drop-in MHA replacement with pruning support |
| `HeadImportanceScorer` | Compute importance scores from gradients/activations |
| `AttentionHeadPruner` | Orchestrate pruning across model layers |
-->

---

## Depthwise Separable Convolutions

> Based on "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (arXiv:1704.04861)

Replace standard convolutions with depthwise separable variants for **8-9x computational savings**.

### Quick Start

```python
from service.layers import (
    DepthwiseSeparableConv2d,
    InvertedResidual,
    apply_width_multiplier,
    replace_conv_with_depthwise_separable,
    compute_flops_reduction,
)

# Use depthwise separable blocks directly
conv = DepthwiseSeparableConv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    activation="relu6",
)

# Or inverted residual (MobileNetV2 style)
block = InvertedResidual(
    in_channels=64,
    out_channels=64,
    stride=1,
    expand_ratio=6.0,
)

# Auto-replace convolutions in existing model
num_replaced = replace_conv_with_depthwise_separable(model, min_channels=32)
print(f"Replaced {num_replaced} convolutions")

# Calculate theoretical speedup
stats = compute_flops_reduction(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    spatial_size=56,
)
print(f"Speedup: {stats['reduction_ratio']:.1f}x")  # ~8.4x
```

### Width Multiplier

Scale network width uniformly for memory/speed tradeoffs:

```python
# Apply 0.75x width multiplier
new_channels = apply_width_multiplier(channels=64, multiplier=0.75)
# Result: 48 channels
```

---

<!-- 
## Once-for-All (OFA) Networks [NOT IMPLEMENTED]

TODO: The following section documents a planned feature. The ofa.py module
has not been created yet. Uncomment when implemented.

> Based on "Once-for-All: Train One Network and Specialize it for Efficient Deployment" (arXiv:1908.09791)

Train a single "supernet" that contains **>10^19 sub-networks**. Extract specialized models for any hardware without retraining.

### Quick Start

```python
from service.layers import (
    OFAConfig,
    SubNetworkConfig,
    ElasticKernel,
    ElasticWidth,
    ElasticBlock,
    OFASubNetworkExtractor,
    create_progressive_shrinking_schedule,
)

# Build an OFA supernet with elastic components
elastic_conv = ElasticKernel(
    in_channels=64,
    out_channels=128,
    max_kernel_size=7,
    supported_sizes=[3, 5, 7],
)

# Configure supernet
ofa_config = OFAConfig(
    max_depth=[4, 4, 6, 4],
    supported_kernel_sizes=[3, 5, 7],
    supported_resolutions=[128, 160, 192, 224],
)

# Extract sub-networks
extractor = OFASubNetworkExtractor(supernet, ofa_config)

# Sample configurations
smallest = extractor.sample_smallest()
largest = extractor.sample_largest()
random_config = extractor.sample_random()

# Deploy specific sub-network
extractor.set_config(smallest)
output = supernet(input_tensor)

# Estimate complexity
stats = extractor.get_complexity(smallest)
print(f"FLOPs: {stats['estimated_flops']:.2e}")
```

### Progressive Shrinking Training

```python
# Create training schedule
schedule = create_progressive_shrinking_schedule(ofa_config, total_epochs=100)

for phase in schedule:
    print(f"Epochs {phase['epoch_start']}-{phase['epoch_end']}:")
    print(f"  Sample kernel: {phase['sample_kernel']}")
    print(f"  Sample depth: {phase['sample_depth']}")
    print(f"  Sample width: {phase['sample_width']}")
```
-->

---

<!-- 
## Patient Knowledge Distillation [NOT IMPLEMENTED]

TODO: The following section documents a planned feature. The distillation.py module
has not been created yet. Uncomment when implemented.

> Based on "Patient Knowledge Distillation for BERT Model Compression" (arXiv:2012.06785)

Transfer knowledge from a large teacher model to a smaller student, learning from **multiple intermediate layers** for better generalization.

### Quick Start

```python
from service.layers import (
    DistillationConfig,
    DistillationStrategy,
    PatientDistillationLoss,
    create_distillation_trainer,
)

# Configure distillation
config = DistillationConfig(
    temperature=4.0,
    alpha=0.5,  # 50% distill loss, 50% task loss
    strategy=DistillationStrategy.PATIENT,
    num_teacher_layers=12,
    pkd_variant="skip",  # or "last"
)

# Create distillation setup
trainer = create_distillation_trainer(
    teacher=teacher_model,
    student=student_model,
    config=config,
)

# Training loop
for batch in dataloader:
    # Forward pass
    student_out = student(batch)
    with torch.no_grad():
        teacher_out = teacher(batch)
    
    # Compute loss
    distill_loss = trainer["loss_fn"](student_out, teacher_out)
    task_loss = criterion(student_out["logits"], labels)
    
    total_loss = config.alpha * distill_loss + (1 - config.alpha) * task_loss
    total_loss.backward()
```

### PKD Variants

| Variant | Description | Example (12→6 layers) |
|---------|-------------|----------------------|
| PKD-Last | Student matches last K teacher layers | Student [1-6] ↔ Teacher [7-12] |
| PKD-Skip | Student matches every K-th teacher layer | Student [1-6] ↔ Teacher [2,4,6,8,10,12] |
-->

---

## Integration with Existing Services

### Optimized Export Pipeline

```python
from service.export_service import ExportService
from service.layers.depthwise_separable import replace_conv_with_depthwise_separable

# 1. Replace standard convolutions with efficient depthwise separable
# (Gives ~8x computational savings)
replaced = replace_conv_with_depthwise_separable(model, min_channels=32)
print(f"Replaced {replaced} convolutions")

# 2. Export to edge format
ExportService.export_ncnn(model_path)
ExportService.export_coreml(model_path, half=True)
ExportService.export_tflite(model_path)
```

### With Spatial Inference Pipeline

```python
from pipeline import SpatialInferencePipeline

# Use optimized model
pipeline = SpatialInferencePipeline(
    model_path="resources/pruned_yolov8m.pt",
    optimize=True,  # Enable JIT optimization
    enable_slam=True,
)
```

---

## Benchmarks

Expected improvements from optimization techniques:

| Technique | FLOPs Reduction | Latency Improvement | Accuracy Impact |
|-----------|-----------------|---------------------|-----------------|
| 30% head pruning | ~25% | ~20% faster | -0.5% mAP |
| Depthwise separable | ~85% | ~8x faster | -1-2% mAP |
| OFA (smallest) | ~90% | ~10x faster | -3% mAP |
| Distillation (6L) | ~50% | ~2x faster | -1% mAP |

*Actual results vary by model and dataset.*

---

## Further Reading

- [README.md](../README.md) - Project overview
- [RESEARCH_PAPERS.md](RESEARCH_PAPERS.md) - Full paper citations
- [RTDETR_TRAINING.md](RTDETR_TRAINING.md) - RT-DETR training guide
