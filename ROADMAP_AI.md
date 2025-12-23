# 🗺 AI Integration Roadmap

This document outlines the strategic plan for injecting AI-driven capabilities into the YOLO Training and Preprocessing pipeline. The goal is to evolve from traditional computer vision heuristics to a composable, scalable, and semantically-aware ML service.

## 📈 Evolution Phases

### Phase 1: AI-Assisted Cleaning & Labeling (Short Term)
*Focus: Improving data quality with Foundation Models.*

- **Semantic Relevance Filtering**:
  - Implement `SemanticRelevanceCleaner` using CLIP or lightweight VLMs (e.g., Moondream).
  - Automatically remove images that don't match the dataset's target classes (e.g., removing a "Selfie" from a "Pothole" dataset).
- **LLM Auto-Correction**:
  - Use GPT-4o-vision or Gemini Pro Vision to verify bounding boxes.
  - Implementation of `AnnotationRefiner` to adjust box coordinates or class labels based on visual reasoning.

### Phase 2: Generative & Spatial Intelligence (Medium Term)
*Focus: Solving data scarcity and environmental context.*

- **Generative Augmentation**:
  - Implement `WeatherTransform` using Stable Diffusion + ControlNet.
  - Generate synthetic variations for hard-to-find conditions (e.g., snow, heavy rain, night-time).
- **SLAM-Assisted Labeling & Mapping**:
  - Integrate **Simultaneous Localization and Mapping (SLAM)** to track objects across temporal frames.
  - Use 6-DOF camera poses to propagate labels from one keyframe to a whole sequence.
  - Implementation of `SpatialMapper` to project detections into a persistent 3D world map.
- **Occlusion Injection**:
  - Use Segment Anything Model (SAM) to cutout objects and intelligently place them as occlusions.

### Phase 3: Distributed Execution & Scale (Long Term)
*Focus: Infrastructure for high-volume AI processing.*

- **Distributed Task Queue**:
  - Transition from local `ThreadPoolExecutor` to **Ray** or **Celery**.
  - Decouple the API from the GPU workers to allow horizontal scaling.
- **AI Provider Abstraction**:
  - Implementation of an `AIProvider` factory.
  - Seamlessly toggle between local models (Ollama/PyTorch) and Cloud APIs (OpenAI/Anthropic/Google).

---

## 🛠 Design Patterns for Injection

### The "Plugin" Implementation
New AI components will inherit from existing base classes to ensure 100% compatibility with the current `PreprocessingPipeline`.

```python
# Proposed structure for AI implementation
class GenAIAugmentor(BaseTransform):
    def __init__(self, prompt: str, strength: float = 0.5):
        self.prompt = prompt
        self.strength = strength

    def _apply_ai(self, image):
        # AI Logic here
        pass
```

### Async Job Management
To keep the API responsive while running slow AI tasks:
1. **Request**: `POST /api/v1/preprocess` returns `202 Accepted` with a `job_id`.
2. **Processing**: AI tasks are sent to a background worker.
3. **Polling**: `GET /api/v1/jobs/{job_id}` returns status (Pending, Running, Completed, Failed).

---

## 📋 Action Items

1. [ ] Define `AIBase` interface for generic AI provider calls.
2. [ ] Integrate `Ollama` support for low-latency local VLM cleaning.
3. [ ] Implement async processing endpoint structure in `routes.py`.
4. [ ] Create `service/preprocessing/ai/` directory for specialized modules.
