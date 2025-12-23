# Memory-Efficient Training & PEFT Techniques

Research on parameter-efficient fine-tuning and memory optimization for large vision-language models.

---

## Papers Reviewed

### 1. COAP: Correlation-Aware Gradient Projection
**arXiv:2412.00071**

**Problem:** Training large models requires massive memory for optimizer states (Adam stores 2x model size).

**Solution:** COAP projects gradients into low-rank spaces while preserving inter-projection correlation.

| Metric | Result |
|--------|--------|
| LLaMA-1B memory savings | 61% |
| LLaVA-7B speedup vs GaLore | 4x |
| With 8-bit quantization | 81% memory reduction |

**Key Insight:** Unlike LoRA (which constrains updates) or GaLore (which ignores correlation), COAP maintains training dynamics while reducing memory.

---

### 2. PEFT for Remote Sensing Visual Grounding
**arXiv:2503.23083**

**Problem:** Adapting large VLMs (Grounding DINO, OFA) to domain-specific tasks is expensive.

**Techniques Evaluated:**

| Method | Model | Approach |
|--------|-------|----------|
| LoRA | Grounding DINO | Low-rank adapters on attention |
| BitFit | OFA | Only train bias terms |
| Adapters | OFA | Small bottleneck layers |

**Result:** SOTA performance on RS visual grounding with ~90% parameters frozen.

---

## Integration with Meta Glasses Vision

| Technique | Application | Benefit |
|-----------|-------------|---------|
| LoRA | Fine-tune backbone for egocentric | 10-100x fewer trainable params |
| BitFit | Quick domain adaptation | Only bias updates (0.1% params) |
| COAP | Train VLM on-device | 4x faster, 81% less memory |
| Gradient checkpointing | Large batch training | Trade compute for memory |

---

## Implementation in This Project

### Available PEFT Wrappers (`service/layers/peft.py`):

```python
from service.layers.peft import apply_lora, apply_bitfit

# LoRA on backbone
model = DetectionModel(nc=80)
apply_lora(model.backbone, rank=8)

# BitFit (bias-only training)
apply_bitfit(model)
```

---

## References

- [COAP](https://arxiv.org/abs/2412.00071)
- [RS-VG PEFT](https://arxiv.org/abs/2503.23083)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [GaLore](https://arxiv.org/abs/2403.03507)
