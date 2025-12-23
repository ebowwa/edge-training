# Object Counting Techniques for Edge Vision

This document surveys recent advancements in object counting, focusing on techniques applicable to egocentric vision and wearable devices like Meta Ray-Ban glasses.

---

## Papers Reviewed

### 1. CrowdVLM-R1: Fuzzy RL for Crowd Counting
**arXiv:2504.03724**

**Key Innovation:** Fuzzy Group Relative Policy Reward (FGRPR)
- Replaces binary 0/1 accuracy rewards with a **fuzzy reward function** that provides nuanced incentives.
- Encourages more precise outputs by assigning higher rewards to closer approximations.
- Outperforms GPT-4o, LLaMA2-90B, and supervised fine-tuning on 5 in-domain datasets.

**Relevance to Meta Glasses:**
- Could improve real-time crowd estimation in egocentric video.
- Fuzzy rewards are useful for tasks where exact counts are difficult (e.g., moving crowds).

---

### 2. VA-Count: Zero-Shot Object Counting
**arXiv:2407.04948**

**Key Innovation:** Visual Association-based Zero-shot Object Counting
- **Exemplar Enhancement Module (EEM):** Discovers potential exemplars using VLMs.
- **Noise Suppression Module (NSM):** Contrastive learning to filter suboptimal exemplars.
- No manual annotations needed during testing.

**Relevance to Meta Glasses:**
- Enables counting novel object classes without retraining.
- Ideal for dynamic environments where users encounter diverse objects.

---

### 3. AMDCN: Dilated Convolutions for Density Maps
**arXiv:1804.07821**

**Key Innovation:** Aggregated Multicolumn Dilated Convolution Network
- Uses **dilated filters** to capture multiscale information.
- Aggregates features from multiple columns to handle perspective effects.
- Generates **density maps** via regression (integral = object count).

**Relevance to Meta Glasses:**
- Dilated convolutions are efficient and suitable for edge deployment.
- Density maps can be overlaid on the spatial map from SLAM for context.

---

## Integration Opportunities

| Technique | Layer/Module | Purpose |
|-----------|--------------|---------|
| Dilated Convolutions | `service/layers/common.py` | Add `DilatedConv` block for multiscale features |
| Density Map Head | `service/layers/head.py` | Add `DensityHead` for counting tasks |
| Fuzzy Reward Function | `service/training_service.py` | Implement FGRPR for fine-tuning |
| Exemplar Mining | `pipeline/spatial_inference.py` | Zero-shot class discovery via VLM |

---

## Next Steps

1. **Implement DilatedConv block** in `common.py` for multiscale feature extraction.
2. **Add DensityHead** to predict density maps alongside detection boxes.
3. **Research VLM integration** for zero-shot exemplar discovery (VA-Count style).
4. **Prototype FGRPR** reward function for crowd counting fine-tuning.

---

## References

- [CrowdVLM-R1](https://arxiv.org/abs/2504.03724)
- [VA-Count](https://arxiv.org/abs/2407.04948)
- [AMDCN](https://arxiv.org/abs/1804.07821)

---

## Additional Papers

### 4. Learning To Count Everything
**arXiv:2104.08391**

Few-shot counting for any object category.

| Feature | Description |
|---------|-------------|
| Approach | Few-shot regression → density map |
| Input | Query image + few exemplars |
| Adaptation | Novel categories at test time |
| Dataset | 147 categories, 6000+ images |
| [Code](https://github.com/cvlab-stonybrook/LearningToCountEverything) |

**Relevance:** Count any object type on Meta glasses with minimal training.
