# Egocentric Vision Research

Research on first-person video understanding, cross-view generation, and efficient deployment.

---

## Papers

### 1. EgoX: Egocentric Video from Exocentric Views
**arXiv:2512.08269**

Generates first-person video from third-person input using video diffusion models.

| Component | Description |
|-----------|-------------|
| LoRA Adaptation | Lightweight fine-tuning of video diffusion |
| Unified Conditioning | Width/channel concatenation of ego + exo priors |
| Geometry-Guided Attention | Selective attention for spatial coherence |

**Applications:**
- Synthetic egocentric training data from existing datasets
- Cross-view understanding for mixed reality
- Virtual POV from surveillance/drone footage

---

### 2. Multimodal Distillation for Egocentric Action Recognition
**arXiv:2307.07483**

Train with multiple modalities, deploy with RGB only.

| Phase | Modalities | Model |
|-------|------------|-------|
| Training | RGB + Flow + Audio + Objects | Multimodal Teacher |
| Inference | **RGB only** | Distilled Student |

**Key Results (Epic-Kitchens, Something-Something):**
- Students surpass unimodal and multimodal baselines
- Better calibration with fewer input views
- [Code](https://github.com/gorjanradevski/multimodal-distillation)

**Applications:**
- Train with all Meta glasses sensors (camera + IMU + audio)
- Deploy lightweight RGB-only model on edge
- Hand-object interaction recognition

---

## Integration with Platform

| Paper | Module | Use Case |
|-------|--------|----------|
| EgoX | Data Augmentation | Generate ego training data |
| Multimodal Distillation | `scripts/train.py` | Distillation loss for edge models |

### Implementation Ideas

```python
# Distillation training loop
teacher_output = teacher_model(rgb, flow, audio)
student_output = student_model(rgb)  # RGB only

# KL divergence distillation loss
distill_loss = F.kl_div(
    F.log_softmax(student_output / T, dim=-1),
    F.softmax(teacher_output / T, dim=-1),
    reduction='batchmean'
) * (T ** 2)
```

---

## References

- [EgoX](https://arxiv.org/abs/2512.08269)
- [Multimodal Distillation](https://arxiv.org/abs/2307.07483)
- [Epic-Kitchens Dataset](https://epic-kitchens.github.io/)
- [Ego4D Dataset](https://ego4d-data.org/)
