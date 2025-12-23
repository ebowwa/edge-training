# BuzzSet: Large-Scale Pollinator Detection Dataset

> **Paper**: "BuzzSet: A Large-Scale Dataset for Automated Pollinator Monitoring"  
> **Focus**: Small object detection for ecological computer vision  
> **Models**: YOLOv12 (initial labeling), RF-DETR (final evaluation)

---

## 📋 Dataset Overview

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Images | 7,856 |
| Total Annotations | 8,000+ instances |
| Image Tile Size | 256 × 256 pixels |
| Classes | 3 |
| Collection Context | Real agricultural field conditions |

### Class Distribution

| Class | Description | Notes |
|-------|-------------|-------|
| **Honeybees** | Apis mellifera | Primary target, highest F1 |
| **Bumblebees** | Bombus spp. | Secondary target, strong detection |
| **Unidentified** | Other insects | Challenging due to label ambiguity |

---

## 🎯 Why This Dataset Matters

### Ecological Importance
- Pollinator populations are **declining globally** due to:
  - Pesticide use
  - Habitat loss
  - Climate change
  - Disease
- Automated monitoring enables **scalable, non-invasive** population tracking
- Supports agricultural planning and conservation efforts

### Computer Vision Challenges
1. **Small Object Detection**: Insects are tiny relative to image size
2. **Class Ambiguity**: Many insects look similar; "unidentified" class is intentionally noisy
3. **Field Conditions**: Variable lighting, backgrounds, motion blur
4. **Imbalanced Classes**: Unidentified insects have lower frequency

---

## 🔬 Methodology

### Annotation Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Raw Images     │────▶│  YOLOv12        │────▶│  Human          │
│  (high-res)     │     │  Pre-labeling   │     │  Verification   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Final Labels   │
                                               │  (verified)     │
                                               └─────────────────┘
```

1. **Pre-labeling with YOLOv12**: Trained on external pollinator data
2. **Human Verification**: Using open-source labeling tools (e.g., Label Studio, CVAT)
3. **Quality Control**: Manual review to minimize label noise

### Image Preprocessing

The key innovation is **tiling high-resolution images into 256×256 patches**:

```python
# Pseudocode for tiling approach
def tile_image(image, tile_size=256, overlap=0):
    """
    Split large image into smaller tiles for small object detection.
    
    Why 256x256?
    - Typical insect: ~20-50 pixels in original image
    - After tiling: insect becomes ~10-20% of tile area
    - Better feature extraction at detection scale
    """
    tiles = []
    h, w = image.shape[:2]
    
    for y in range(0, h - tile_size + 1, tile_size - overlap):
        for x in range(0, w - tile_size + 1, tile_size - overlap):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append((tile, x, y))
    
    return tiles
```

**Benefits of Tiling:**
- Increases effective resolution of small objects
- Enables batch processing of large images
- Reduces GPU memory requirements per inference
- Allows standard object detectors to work on small targets

---

## 🤖 Model Baselines

### RF-DETR (Real-Time Detection Transformer)

RF-DETR is used as the primary evaluation model. Key characteristics:

| Property | Details |
|----------|---------|
| Architecture | DETR-based transformer |
| Backbone | Typically ResNet or Swin |
| Detection Head | Transformer decoder with learned queries |
| Strengths | Strong global context, no NMS needed |
| Inference | Real-time capable |

### Comparison: YOLO vs DETR-based

| Aspect | YOLO Family | DETR/RF-DETR |
|--------|-------------|--------------|
| Architecture | CNN + Detection Head | CNN + Transformer |
| Object Queries | Grid-based anchors | Learned queries |
| Post-processing | NMS required | End-to-end, no NMS |
| Small Objects | Requires multi-scale | Better with attention |
| Training | Faster convergence | Needs more epochs |
| Real-time | Yes (v8, v11, etc.) | Yes (RF-DETR, RT-DETR) |

### Results on BuzzSet

| Metric | Honeybee | Bumblebee | Unidentified |
|--------|----------|-----------|--------------|
| **F1-Score** | 0.94 | 0.92 | Lower (challenging) |
| **Confusion** | Minimal | Minimal | More errors |

**Overall Detection:**
- Best **mAP@0.50**: 0.559
- Strong class separation between honeybees and bumblebees
- Unidentified class provides robustness evaluation (noisy labels)

---

## 🔧 Integration with yolo-training

### Option 1: Add BuzzSet Dataset Loader

```python
# service/dataset_service.py (proposed extension)

class BuzzSetLoader:
    """
    Loader for BuzzSet pollinator detection dataset.
    
    Expected structure:
    buzzset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
    """
    
    CLASSES = ["honeybee", "bumblebee", "unidentified"]
    
    @staticmethod
    def create_yaml(data_path: str) -> str:
        """Generate data.yaml for YOLO training."""
        yaml_content = {
            "path": data_path,
            "train": "images/train",
            "val": "images/val", 
            "test": "images/test",
            "nc": 3,
            "names": BuzzSetLoader.CLASSES,
        }
        # Write and return path...
```

### Option 2: Add Tiling Transform to Preprocessing

```python
# service/preprocessing/transforms.py (proposed extension)

class TilingTransform:
    """
    Tile large images for small object detection.
    
    Inspired by BuzzSet's 256x256 tiling approach.
    """
    
    def __init__(self, tile_size: int = 256, overlap: int = 32):
        self.tile_size = tile_size
        self.overlap = overlap
    
    def __call__(self, image: np.ndarray, labels: np.ndarray):
        """
        Split image into tiles, adjusting labels accordingly.
        
        Args:
            image: Full-resolution image (H, W, 3)
            labels: YOLO format labels (class, x_center, y_center, w, h)
            
        Returns:
            List of (tile_image, tile_labels) tuples
        """
        tiles = []
        h, w = image.shape[:2]
        step = self.tile_size - self.overlap
        
        for y in range(0, h - self.tile_size + 1, step):
            for x in range(0, w - self.tile_size + 1, step):
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                tile_labels = self._adjust_labels(labels, x, y, w, h)
                if len(tile_labels) > 0:  # Only keep tiles with objects
                    tiles.append((tile, tile_labels))
        
        return tiles
```

### Option 3: Add RF-DETR/RT-DETR Support

While the current pipeline uses Ultralytics YOLO, transformer-based detectors can be added:

```python
# service/inference_service.py (proposed extension)

class RTDETRInferenceService:
    """
    Real-Time DETR inference service.
    
    Uses Ultralytics RT-DETR implementation:
    https://docs.ultralytics.com/models/rtdetr/
    """
    
    def __init__(self, model_path: str = "rtdetr-l.pt"):
        from ultralytics import RTDETR
        self.model = RTDETR(model_path)
    
    def infer_frame(self, frame, config):
        results = self.model(frame, conf=config.conf_threshold)
        # Process results...
```

---

## 📚 Key Concepts for Exploration

### 1. Small Object Detection Strategies

| Strategy | Description | Used in BuzzSet |
|----------|-------------|-----------------|
| **Image Tiling** | Split into smaller patches | ✅ 256×256 tiles |
| **Multi-scale Training** | Train at various resolutions | Common in YOLO |
| **Feature Pyramid Networks** | Multi-resolution features | Built into most detectors |
| **Attention Mechanisms** | Focus on small regions | RF-DETR strength |
| **Super-resolution** | Upscale images | Not mentioned |

### 2. Handling Label Noise

The "unidentified" class intentionally contains ambiguous samples:

```python
# Strategies for noisy labels
class LabelNoiseStrategies:
    """
    Techniques for training with label uncertainty.
    """
    
    # 1. Label Smoothing: Soften hard labels
    def label_smoothing(self, labels, epsilon=0.1):
        return labels * (1 - epsilon) + epsilon / num_classes
    
    # 2. Confidence Weighting: Down-weight uncertain samples
    def confidence_weighted_loss(self, predictions, labels, confidence):
        loss = cross_entropy(predictions, labels)
        return (loss * confidence).mean()
    
    # 3. Class Rebalancing: Handle imbalanced classes
    def focal_loss(self, predictions, labels, gamma=2.0):
        ce = cross_entropy(predictions, labels)
        pt = torch.exp(-ce)
        return ((1 - pt) ** gamma * ce).mean()
```

### 3. Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | % of detections that are correct |
| **Recall** | TP / (TP + FN) | % of ground truth objects found |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of P and R |
| **mAP@0.50** | Mean AP at IoU ≥ 0.50 | Overall detection quality |
| **mAP@0.50:0.95** | Mean over IoU thresholds | Stricter localization |

---

## 🔗 Related Research & Resources

### Datasets
- **iNaturalist**: Large-scale species classification
- **IP102**: Insect pest dataset (102 classes)
- **ArTaxOr**: Arthropod taxonomy dataset

### Models
- **YOLOv11/v12**: Latest YOLO iterations (Ultralytics)
- **RT-DETR**: Real-time transformer detector (Ultralytics)
- **DINO**: Self-distillation with no labels
- **Grounding DINO**: Open-vocabulary detection

### Techniques
- **SAHI**: Slicing Aided Hyper Inference (for small objects)
  - GitHub: https://github.com/obss/sahi
  - Tiles images at inference time, merges results

---

## 🎯 Action Items

To fully integrate BuzzSet concepts into this project:

- [ ] Add `TilingTransform` to preprocessing pipeline
- [ ] Create `BuzzSetLoader` dataset class
- [ ] Add RT-DETR inference option alongside YOLO
- [ ] Implement SAHI-style inference for small objects
- [ ] Add label noise handling strategies
- [ ] Create example notebook for pollinator detection

---

## 📖 Citation

```bibtex
@article{buzzset2025,
  title={BuzzSet: A Large-Scale Dataset for Automated Pollinator Monitoring},
  author={...},
  journal={arXiv preprint arXiv:2508.19762},
  year={2025}
}
```
