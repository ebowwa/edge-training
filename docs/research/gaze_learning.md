# Gaze-Conditioned Learning

Research on using eye movements and gaze as signals for learning, reward, and intent decoding.

---

## Papers

### 1. Decoding Information Seeking Goals from Eye Movements
**arXiv:2505.02872**

Decode what someone is looking for just from their gaze patterns.

| Feature | Description |
|---------|-------------|
| Goal Decoding | Infer intent from eye movements |
| Multimodal LLMs | Text + gaze inputs |
| Free-form Output | Generate goal description text |

**Applications:** Infer user intent without explicit queries.

---

### 2. Eyes on Target: Gaze-Aware Object Detection
**arXiv:2511.01237**

Bias detection toward human-attended regions using gaze features.

| Technique | Benefit |
|-----------|---------|
| Gaze injection into ViT | Enhanced detection in attended areas |
| Attention mechanism | Prioritize where user looks |

---

### 3. Bridging Gaze and VLMs via Attention Regularization
**arXiv:2510.24072**

Integrate gaze patterns into VLM attention during training.

| Application | Improvement |
|-------------|-------------|
| Future event prediction | Better semantic scores |
| Activity understanding | More accurate predictions |

---

### 4. Human Gaze Boosts Object-Centered Learning
**arXiv:2501.02966**

Self-supervised learning from gaze-centered visual crops.

| Finding | Implication |
|---------|-------------|
| Central vision crops | Better object representations |
| Temporal gaze dynamics | Stronger visual learning |

---

## Integration Points

| Module | Gaze Enhancement |
|--------|------------------|
| `ObjectCache` | Priority = gaze duration |
| `DensityHead` | Weight counts by fixation |
| `SlamService` | Anchor gaze-attended objects |
| `InferenceService` | Boost detection in gaze region |

---

## Implementation

See `service/gaze/intent_decoder.py` for gaze-intent decoding stub.

---

## References

- [Goal Decoding from Eye Movements](https://arxiv.org/abs/2505.02872)
- [Gaze-Aware Object Detection](https://arxiv.org/abs/2511.01237)
- [Gaze + VLM Attention](https://arxiv.org/abs/2510.24072)
- [Gaze Boosts Object Learning](https://arxiv.org/abs/2501.02966)
- [Visual Search Asymmetry](https://arxiv.org/abs/2106.02953)

---

## Additional Papers

### 5. Visual Search Asymmetry: Deep Nets and Humans Share Biases
**arXiv:2106.02953**

Models how humans and deep nets share inherent biases in visual search.

| Finding | Implication |
|---------|-------------|
| Search asymmetry | Finding A among B ≠ finding B among A |
| Emerges from training | Statistical properties of training data |
| Eye movement model | Eccentricity-dependent recognition + top-down cues |

**[Code](https://github.com/kreimanlab/VisualSearchAsymmetry)**

**Applications:** Predict where user will look, prioritize likely targets.
