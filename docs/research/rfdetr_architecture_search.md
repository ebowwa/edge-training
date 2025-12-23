# RF-DETR: Real-Time Detection Transformer with Neural Architecture Search

> **Paper**: "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers"  
> **Authors**: Roboflow Research Team  
> **Source**: [arXiv:2511.09554](https://arxiv.org/abs/2511.09554)  
> **Code**: [github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr)  
> **License**: Apache 2.0

---

## 🎯 TL;DR

RF-DETR is **the first real-time object detector to surpass 60 AP on COCO**. It uses Neural Architecture Search (NAS) to find optimal accuracy-latency tradeoffs for any target dataset, outperforming both YOLO and prior DETR variants.

---

## 📊 Benchmark Results

### COCO Performance

| Model | AP | AP50 | Latency | Notes |
|-------|-----|------|---------|-------|
| **RF-DETR-2XL** | **60.0+** | - | Real-time | First RT model >60 AP |
| RF-DETR-L | 56.8 | - | Real-time | |
| RF-DETR-M | 54.7 | - | Real-time | |
| RF-DETR-S | 53.0 | - | Real-time | Beats YOLO11-X (51.2) |
| RF-DETR-N (nano) | 48.0 | - | Real-time | +5.3 AP vs D-FINE nano |
| RF-DETR-Base | 53.3 | - | Real-time | |

### Roboflow100-VL (Domain Generalization)

| Model | mAP | Speed vs GroundingDINO |
|-------|-----|------------------------|
| RF-DETR-2XL | 86.7 | **20x faster** |
| GroundingDINO (tiny) | 85.5 | Baseline |

---

## 🏗 Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         RF-DETR                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │  DINOv2     │────▶│   LW-DETR   │────▶│   NAS-      │       │
│   │  Backbone   │     │   Encoder   │     │   Tuned     │       │
│   │  (frozen)   │     │             │     │   Head      │       │
│   └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
│   Pre-trained          Lightweight         Architecture         │
│   Vision               Transformer         Search               │
│   Features             Encoder             Optimized            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Backbone** | DINOv2 (frozen/fine-tuned) | Strong pre-trained features |
| **Encoder** | LW-DETR (Lightweight DETR) | Efficient transformer layers |
| **Detection Head** | Learned object queries | No anchors, no NMS |
| **Optimization** | Weight-sharing NAS | Find Pareto-optimal configs |

---

## 🔬 Neural Architecture Search (NAS)

### What is Weight-Sharing NAS?

Traditional NAS trains thousands of models from scratch. Weight-sharing NAS:

1. **Train once**: Fine-tune a "supernet" with all possible configurations
2. **Evaluate many**: Sample different sub-networks without retraining
3. **Find Pareto curve**: Map accuracy vs latency tradeoffs

```python
# Pseudocode for weight-sharing NAS
class WeightSharingNAS:
    """
    Discover optimal network configurations without retraining.
    """
    
    def __init__(self, supernet):
        self.supernet = supernet  # Contains all possible sub-networks
        
    def search(self, dataset, latency_targets):
        """
        Find Pareto-optimal configurations.
        
        Args:
            dataset: Target dataset for evaluation
            latency_targets: List of target inference times
            
        Returns:
            List of (config, accuracy, latency) tuples on Pareto frontier
        """
        pareto_configs = []
        
        for config in self.sample_configurations(n=1000):
            # Extract sub-network from supernet
            subnet = self.supernet.extract(config)
            
            # Evaluate without retraining (weights are shared)
            accuracy = self.evaluate(subnet, dataset)
            latency = self.measure_latency(subnet)
            
            if self.is_pareto_optimal(accuracy, latency, pareto_configs):
                pareto_configs.append((config, accuracy, latency))
        
        return pareto_configs
    
    def sample_configurations(self, n):
        """Sample network configurations (depth, width, etc.)"""
        for _ in range(n):
            yield {
                "encoder_layers": random.choice([2, 4, 6, 8]),
                "hidden_dim": random.choice([128, 256, 384, 512]),
                "num_queries": random.choice([100, 300, 500]),
                "backbone_scale": random.choice(["small", "base", "large"]),
            }
```

### Tunable Knobs in RF-DETR

| Knob | Options | Effect |
|------|---------|--------|
| **Encoder Depth** | 2, 4, 6, 8 layers | More layers = higher accuracy, slower |
| **Hidden Dimension** | 128-512 | Wider = more capacity |
| **Object Queries** | 100-500 | More queries = detect more objects |
| **Backbone Scale** | nano, small, base, large | Larger = better features |
| **Resolution** | 480, 640, 800... | Higher = better small objects |

---

## ⚡ RF-DETR vs YOLO: When to Use Which?

### Feature Comparison

| Feature | RF-DETR | YOLO (v8/v11) |
|---------|---------|---------------|
| **Architecture** | Transformer | CNN + Detection Head |
| **NMS Required** | ❌ No (end-to-end) | ✅ Yes |
| **Deterministic Latency** | ✅ Yes | ❌ Varies with object count |
| **Domain Adaptation** | Excellent (NAS) | Good |
| **Small Objects** | Better (attention) | Needs tricks (tiling) |
| **Training Speed** | Slower | Faster |
| **Edge Deployment** | Improving | Excellent (NCNN, TFLite) |

### Decision Guide

```
Use RF-DETR when:
├── You need consistent, deterministic latency
├── Domain differs significantly from COCO
├── Small object detection is critical
├── You can fine-tune for your specific dataset
└── You want state-of-the-art accuracy

Use YOLO when:
├── Edge deployment is priority (TFLite, NCNN)
├── You need fastest training iteration
├── COCO-like domains work well
├── Resources are constrained
└── You need extensive community support
```

---

## 🔧 Integration with yolo-training

### Option 1: Add RF-DETR Inference Service

```python
# service/rfdetr_inference_service.py (proposed)

from rfdetr import RFDETRModel  # pip install rfdetr

class RFDETRInferenceService:
    """
    RF-DETR inference service.
    
    Provides transformer-based detection as alternative to YOLO.
    """
    
    VARIANTS = {
        "nano": "rf-detr-nano",
        "small": "rf-detr-small", 
        "medium": "rf-detr-medium",
        "large": "rf-detr-large",
        "2xlarge": "rf-detr-2xlarge",
    }
    
    def __init__(self, variant: str = "base", device: str = "cuda"):
        """
        Args:
            variant: Model size (nano, small, medium, large, 2xlarge)
            device: Inference device (cuda, cpu)
        """
        self.model = RFDETRModel.from_pretrained(
            self.VARIANTS.get(variant, variant)
        )
        self.model.to(device)
        self.model.eval()
    
    def infer_frame(self, frame, conf_threshold: float = 0.5):
        """
        Run inference on a single frame.
        
        Args:
            frame: BGR image as numpy array
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections
        """
        import torch
        
        with torch.no_grad():
            results = self.model.predict(
                frame,
                conf_threshold=conf_threshold,
            )
        
        return self._parse_results(results)
    
    def _parse_results(self, results):
        """Convert RF-DETR results to standard Detection format."""
        from config import Detection
        
        detections = []
        for box, score, class_id in zip(
            results.boxes, results.scores, results.class_ids
        ):
            detections.append(Detection(
                class_id=int(class_id),
                class_name=self.model.class_names[class_id],
                confidence=float(score),
                bbox=tuple(box.tolist()),
                bbox_normalized=self._normalize_bbox(box, results.image_size),
            ))
        
        return detections
```

### Option 2: Unified Inference Interface

```python
# service/unified_inference.py (proposed)

from enum import Enum
from typing import Union

class DetectorBackend(Enum):
    YOLO = "yolo"
    RFDETR = "rfdetr"
    RTDETR = "rtdetr"

class UnifiedInferenceService:
    """
    Unified interface for multiple detection backends.
    
    Example:
        # Use YOLO
        service = UnifiedInferenceService("yolov11m.pt", backend="yolo")
        
        # Use RF-DETR
        service = UnifiedInferenceService("rf-detr-large", backend="rfdetr")
        
        # Same API for both
        detections = service.infer(frame)
    """
    
    def __init__(self, model: str, backend: str = "yolo"):
        self.backend = DetectorBackend(backend)
        
        if self.backend == DetectorBackend.YOLO:
            from inference_service import InferenceService
            self._service = InferenceService(model)
        elif self.backend == DetectorBackend.RFDETR:
            from rfdetr_inference_service import RFDETRInferenceService
            self._service = RFDETRInferenceService(model)
        elif self.backend == DetectorBackend.RTDETR:
            from ultralytics import RTDETR
            self._model = RTDETR(model)
    
    def infer(self, frame, **kwargs):
        """Unified inference API."""
        return self._service.infer_frame(frame, **kwargs)
```

### Option 3: Add to Pipeline as Swappable Stage

```python
# pipeline/stages/detector_stages.py (proposed)

from pipeline import PipelineStage

class YOLODetectorStage(PipelineStage):
    """YOLO-based detection stage."""
    name = "yolo_detector"
    # ... existing implementation

class RFDETRDetectorStage(PipelineStage):
    """RF-DETR-based detection stage."""
    name = "rfdetr_detector"
    
    def __init__(self, variant: str = "base"):
        from rfdetr import RFDETRModel
        self.model = RFDETRModel.from_pretrained(variant)
    
    def process(self, frame, context):
        return self.model.predict(frame)

# Usage: Swap detectors in pipeline
from pipeline import AttachablePipeline

pipeline = AttachablePipeline()

# Option A: Use YOLO
pipeline.attach("detector", YOLODetectorStage("yolov11m.pt"))

# Option B: Swap to RF-DETR
pipeline.replace("detector", RFDETRDetectorStage("rf-detr-large"))
```

---

## 📈 Training RF-DETR on Custom Data

### Fine-tuning Workflow

```python
# Example: Fine-tune RF-DETR on BuzzSet pollinators
from rfdetr import RFDETRModel, RFDETRTrainer

# 1. Load pre-trained model
model = RFDETRModel.from_pretrained("rf-detr-base")

# 2. Configure training
trainer = RFDETRTrainer(
    model=model,
    train_data="buzzset/images/train",
    val_data="buzzset/images/val",
    epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    # NAS settings
    nas_search=True,  # Enable architecture search
    latency_target=0.05,  # Target 50ms inference
)

# 3. Train with NAS
best_config = trainer.train()

# 4. Export optimized model
model.export(best_config, "buzzset_rfdetr.pt")
```

### Data Format

RF-DETR uses COCO format by default:

```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "width": 256, "height": 256}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "area": 1234,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "honeybee"},
    {"id": 1, "name": "bumblebee"},
    {"id": 2, "name": "unidentified"}
  ]
}
```

---

## 🔗 Related Work

### DETR Family Evolution

```
DETR (2020)
│   └── Slow convergence, poor small object detection
│
├── Deformable DETR (2021)
│       └── Deformable attention, faster convergence
│
├── DAB-DETR (2022)
│       └── Dynamic anchor boxes
│
├── DN-DETR (2022)
│       └── Denoising training
│
├── DINO (2022)
│       └── Contrastive denoising, SOTA
│
├── RT-DETR (2023)
│       └── Real-time DETR (Ultralytics)
│
└── RF-DETR (2024)
        └── NAS-optimized, 60+ AP on COCO
```

### Comparison with RT-DETR

| Aspect | RF-DETR | RT-DETR |
|--------|---------|---------|
| Developer | Roboflow | Baidu/Ultralytics |
| NAS | ✅ Yes | ❌ No |
| Backbone | DINOv2 | ResNet/HGNetv2 |
| Best COCO AP | 60+ | ~54 |
| Ultralytics Support | Planned | ✅ Yes |
| Domain Adaptation | Excellent | Good |

---

## 🎯 Action Items for yolo-training

- [ ] Add `rfdetr_inference_service.py` with RF-DETR support
- [ ] Create `UnifiedInferenceService` for backend switching
- [ ] Add RF-DETR detector stage to pipeline
- [ ] Support COCO format in dataset service
- [ ] Create training script for RF-DETR fine-tuning
- [ ] Benchmark RF-DETR vs YOLO on custom datasets

---

## 📖 Citation

```bibtex
@article{rfdetr2024,
  title={RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
  author={Roboflow Research},
  journal={arXiv preprint arXiv:2511.09554},
  year={2024}
}
```

---

## 🔗 Resources

- **Paper**: [arXiv:2511.09554](https://arxiv.org/abs/2511.09554)
- **GitHub**: [roboflow/rf-detr](https://github.com/roboflow/rf-detr)
- **Roboflow Blog**: [Introducing RF-DETR](https://blog.roboflow.com/rf-detr/)
- **RF100-VL Benchmark**: [Roboflow 100 Visual Language](https://universe.roboflow.com/roboflow-100)
