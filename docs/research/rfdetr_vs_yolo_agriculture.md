# RF-DETR vs YOLOv12: Agricultural Detection Comparison

> **Paper**: "RF-DETR Object Detection vs YOLOv12: A Study of Transformer-based and CNN-based Architectures for Single-Class and Multi-Class Greenfruit Detection in Complex Orchard Environments Under Label Ambiguity"  
> **Source**: [arXiv:2504.13099](https://arxiv.org/abs/2504.13099)  
> **Domain**: Precision Agriculture  
> **Task**: Greenfruit detection in orchards

---

## 🎯 Study Overview

This paper provides a **head-to-head comparison** of RF-DETR and YOLOv12 on a real agricultural dataset with challenging conditions:

| Challenge | Description |
|-----------|-------------|
| **Label Ambiguity** | Uncertain boundaries between fruit and foliage |
| **Occlusions** | Fruits hidden behind leaves/branches |
| **Background Blending** | Green fruits against green leaves |
| **Complex Scenes** | Cluttered orchard environments |

---

## 📊 Benchmark Results

### Single-Class Detection (Greenfruit)

| Model | mAP@50 | mAP@50:95 | Best For |
|-------|--------|-----------|----------|
| **RF-DETR** | **0.9464** ✅ | - | Complex spatial scenarios |
| YOLOv12N | - | **0.7620** ✅ | Strict localization |
| YOLOv12 (other) | Lower | Lower | Edge deployment |

**Key Finding**: RF-DETR achieved the **highest mAP@50 (94.64%)** for single-class greenfruit detection.

### Multi-Class Detection (Occluded vs Non-Occluded)

| Model | mAP@50 | mAP@50:95 | Best For |
|-------|--------|-----------|----------|
| **RF-DETR** | **0.8298** ✅ | - | Occlusion differentiation |
| YOLOv12L | - | **0.6622** ✅ | Detailed classification |

**Key Finding**: RF-DETR better distinguishes between occluded and non-occluded fruits.

---

## 🔬 Architecture Comparison

### RF-DETR (Transformer-based)

```
┌─────────────────────────────────────────────┐
│              RF-DETR                         │
├─────────────────────────────────────────────┤
│  DINOv2 Backbone                            │
│       ↓                                      │
│  Deformable Attention                       │
│       ↓                                      │
│  Global Context Modeling                    │
│       ↓                                      │
│  Object Queries → Detections                │
└─────────────────────────────────────────────┘

Strengths:
✅ Global context modeling
✅ Handles partial occlusions
✅ Resolves ambiguous boundaries
✅ Swift convergence (~10 epochs)
```

### YOLOv12 (CNN + Attention)

```
┌─────────────────────────────────────────────┐
│              YOLOv12                         │
├─────────────────────────────────────────────┤
│  CNN Backbone                               │
│       ↓                                      │
│  Area Attention Mechanism                   │
│       ↓                                      │
│  Local Feature Extraction                   │
│       ↓                                      │
│  Multi-scale Predictions                    │
└─────────────────────────────────────────────┘

Strengths:
✅ Computational efficiency
✅ Edge deployment ready
✅ Fast inference
✅ Better mAP@50:95 in some cases
```

---

## 📈 Training Dynamics

### Convergence Comparison

| Model | Epochs to Plateau | Convergence Speed |
|-------|-------------------|-------------------|
| RF-DETR (single-class) | ~10 epochs | **Very fast** ✅ |
| RF-DETR (multi-class) | ~15-20 epochs | Fast |
| YOLOv12 | ~30-50 epochs | Standard |

**Key Finding**: RF-DETR's transformer architecture adapts faster to new visual domains.

### Why RF-DETR Converges Faster

1. **Pre-trained DINOv2 backbone**: Strong general visual features
2. **Deformable attention**: Efficiently focuses on relevant regions
3. **Object queries**: Learn object prototypes quickly
4. **End-to-end training**: No anchor tuning needed

---

## 🎯 When to Use Which Model

### Use RF-DETR When:

```
✅ Complex spatial scenarios (orchards, forests, crowds)
✅ Label ambiguity exists (unclear boundaries)
✅ Occlusion handling is critical
✅ You have GPU resources
✅ You need fast convergence on new domains
✅ mAP@50 is your primary metric
```

### Use YOLOv12 When:

```
✅ Edge deployment required (mobile, embedded)
✅ Real-time on resource-constrained hardware
✅ mAP@50:95 (strict localization) matters more
✅ Training infrastructure is limited
✅ Well-defined object boundaries
```

---

## 🌾 Precision Agriculture Implications

### Detection Challenges in Orchards

| Challenge | RF-DETR Approach | YOLO Approach |
|-----------|------------------|---------------|
| **Green-on-green** | Global context helps distinguish | Local features may confuse |
| **Clustered fruits** | Object queries handle overlaps | NMS may suppress valid detections |
| **Partial occlusion** | Deformable attention "sees through" | Harder without full object visible |
| **Varying lighting** | DINOv2 pre-training helps | Data augmentation needed |

### Practical Applications

1. **Yield Estimation**: Count fruits before harvest
2. **Selective Harvesting**: Identify ripe vs unripe
3. **Disease Detection**: Spot infected fruits early
4. **Robotics**: Guide picking arms to fruit locations

---

## 🔧 Integration with yolo-training

### Dataset Structure for Agricultural Detection

```python
# Custom dataset config for greenfruit-like detection
class AgriculturalDatasetConfig:
    """
    Configuration for orchard/agricultural detection datasets.
    """
    
    # Single-class mode
    SINGLE_CLASS = {
        "nc": 1,
        "names": ["greenfruit"],
    }
    
    # Multi-class mode (with occlusion)
    MULTI_CLASS = {
        "nc": 2,
        "names": ["greenfruit_visible", "greenfruit_occluded"],
    }
    
    # Extended mode (maturity stages)
    MATURITY_CLASS = {
        "nc": 4,
        "names": ["unripe", "ripening", "ripe", "overripe"],
    }
```

### Handling Label Ambiguity

```python
# service/preprocessing/transforms.py (proposed extension)

class LabelAmbiguityHandler:
    """
    Strategies for handling ambiguous labels in agricultural datasets.
    """
    
    def __init__(self, strategy: str = "soft_labels"):
        self.strategy = strategy
    
    def apply(self, labels, confidences=None):
        """
        Process ambiguous labels.
        
        Args:
            labels: Original hard labels
            confidences: Optional annotation confidence scores
        """
        if self.strategy == "soft_labels":
            # Convert hard labels to probability distributions
            return self._soft_label_transform(labels, confidences)
        
        elif self.strategy == "ignore_ambiguous":
            # Mark ambiguous samples for exclusion from loss
            return self._mark_ambiguous(labels, confidences)
        
        elif self.strategy == "multi_label":
            # Allow multiple class assignments
            return self._to_multi_label(labels)
    
    def _soft_label_transform(self, labels, confidences):
        """
        Create soft labels based on annotation confidence.
        
        Example: If annotator was 70% confident it's "occluded"
        and 30% confident it's "visible", encode as [0.3, 0.7]
        """
        if confidences is None:
            return labels  # No confidence info, return as-is
        
        soft_labels = []
        for label, conf in zip(labels, confidences):
            # Create soft label distribution
            soft = [0.0] * self.num_classes
            soft[label] = conf
            soft[1 - label] = 1 - conf  # Binary case
            soft_labels.append(soft)
        
        return soft_labels
```

### Model Selection Stage for Pipeline

```python
# pipeline/stages/model_selector.py (proposed)

from pipeline import PipelineStage, PipelineContext

class AdaptiveModelSelector(PipelineStage):
    """
    Dynamically select detection model based on scene complexity.
    
    Uses RF-DETR for complex scenes, YOLO for simple ones.
    """
    
    name = "adaptive_model_selector"
    
    def __init__(
        self,
        rfdetr_model: str = "rf-detr-base",
        yolo_model: str = "yolov12m.pt",
        complexity_threshold: float = 0.5,
    ):
        self.rfdetr = self._load_rfdetr(rfdetr_model)
        self.yolo = self._load_yolo(yolo_model)
        self.complexity_threshold = complexity_threshold
    
    def process(self, frame, context: PipelineContext):
        # Estimate scene complexity
        complexity = self._estimate_complexity(frame)
        context.set("scene_complexity", complexity)
        
        # Select model based on complexity
        if complexity > self.complexity_threshold:
            context.set("model_used", "rf-detr")
            return self.rfdetr.predict(frame)
        else:
            context.set("model_used", "yolo")
            return self.yolo.predict(frame)
    
    def _estimate_complexity(self, frame):
        """
        Estimate scene complexity using simple heuristics.
        
        Factors:
        - Edge density (cluttered = high edges)
        - Color variance (uniform = low variance)
        - Texture energy (complex textures = high energy)
        """
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        # Color variance
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_var = np.std(hsv[:, :, 0]) / 180.0  # Normalize hue variance
        
        # Combined complexity score
        complexity = 0.6 * edge_density + 0.4 * color_var
        
        return complexity
```

---

## 📊 Key Takeaways

### For This Project

| Insight | Application |
|---------|-------------|
| **RF-DETR excels at ambiguity** | Use for tricky detection scenarios |
| **YOLOv12 wins on edge** | Keep YOLO for mobile/wearable deployment |
| **Fast convergence** | Fine-tune RF-DETR with fewer epochs |
| **Multi-class occlusion** | Consider occlusion as a class, not just noise |

### Model Selection Matrix

```
                    Simple Scene          Complex Scene
                    ─────────────────────────────────────
Edge Device    │    YOLOv12N ✅         YOLOv12N (accept tradeoff)
               │
GPU Available  │    YOLOv12M            RF-DETR ✅
               │
Accuracy       │    Either              RF-DETR ✅
Priority       │
```

---

## 🔗 Related Research

- **BuzzSet** (in this repo): Small object detection, similar challenges
- **RF-DETR Architecture** (in this repo): Foundation model details
- **Precision Agriculture Reviews**: Survey papers on crop detection

---

## 📖 Citation

```bibtex
@article{rfdetr_vs_yolov12_agriculture,
  title={RF-DETR Object Detection vs YOLOv12: A Study of Transformer-based and 
         CNN-based Architectures for Single-Class and Multi-Class Greenfruit 
         Detection in Complex Orchard Environments Under Label Ambiguity},
  author={...},
  journal={arXiv preprint arXiv:2504.13099},
  year={2025}
}
```

---

## 🎯 Action Items

- [ ] Add scene complexity estimator to preprocessing
- [ ] Implement `AdaptiveModelSelector` stage
- [ ] Create soft label support for ambiguous annotations
- [ ] Add multi-class occlusion handling
- [ ] Benchmark RF-DETR vs YOLO on custom agricultural data
