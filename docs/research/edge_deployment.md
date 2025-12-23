# Edge Deployment Techniques

Research on efficient model deployment for resource-constrained devices like Meta glasses.

---

## Papers

### 1. DeltaLLM: Training-Free Edge LLM Inference
**arXiv:2507.19608**

Exploits temporal sparsity in attention for efficient edge LLM inference.

| Component | Description |
|-----------|-------------|
| Delta Matrix | Accuracy/memory-aware temporal sparsity |
| Hybrid Attention | Local full attention + delta approximation |
| Training-Free | No fine-tuning, plug-and-play |

**Results:**
| Model | Sparsity | Impact |
|-------|----------|--------|
| BitNet-2B | 60% (prefill) | Slight accuracy gain |
| Llama-1B | 57% (both stages) | Negligible drop |

**Applications:** VLM inference on Meta glasses without retraining.

---

### 2. Edge-Deployable OCR for Billboard Analysis
**arXiv:2507.11730**

Benchmarks OCR models for outdoor text recognition under weather conditions.

| Model | Type | Edge Suitability |
|-------|------|------------------|
| PaddleOCRv4 | CNN | ✅ Fast, lightweight |
| Qwen 2.5 VL 3B | VLM | ⚠️ Better reasoning, heavier |
| SmolVLM2 | VLM | ⚠️ Compact VLM |

**Key Finding:** Lightweight CNN pipelines achieve competitive accuracy at fraction of VLM cost.

**Applications:** Real-time text reading/translation on glasses.

---

## Integration Notes

| Technique | Module | Benefit |
|-----------|--------|---------|
| Temporal sparsity | VLM attention | 57-60% compute reduction |
| PaddleOCR | New service | Fast text extraction |
| Weather augmentation | Preprocessing | Robustness training |

---

## References

- [DeltaLLM](https://arxiv.org/abs/2507.19608)
- [Edge OCR Survey](https://arxiv.org/abs/2507.11730)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

---

## Additional Papers

### 3. Hermes: Memory-Efficient Pipeline Inference
**arXiv:2409.04249**

PIPELOAD mechanism for edge model inference.

| Model | Speedup | Memory Reduction |
|-------|---------|------------------|
| BERT/ViT | 4.24x | 86.7% |
| GPT-style | 2.58x | 90.3% |

**Technique:** Dynamic memory management + parallel model loading.

---

### 4. AIris: AI-Powered Wearable Assistive Device
**arXiv:2405.07606**

Complete wearable AI system for visually impaired - **direct Meta glasses reference**.

| Task | Implementation |
|------|----------------|
| Face recognition | Real-time identification |
| Scene description | NLP auditory output |
| Text reading | OCR + TTS |
| Object recognition | Detection pipeline |
| Money counting | Specialized classifier |
| Barcode scanning | Product identification |

**Relevance:** Blueprint for Meta glasses AI assistant features.
