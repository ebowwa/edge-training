# Memory Systems for Continual Learning

Research on episodic memory, continual learning, and avoiding catastrophic forgetting.

---

## Papers

### 1. GEM: Gradient Episodic Memory
**arXiv:1706.08840**

Prevents catastrophic forgetting by constraining gradient updates.

| Component | Description |
|-----------|-------------|
| Episodic Memory | Stores examples from previous tasks |
| Gradient Projection | Ensures updates don't hurt old tasks |
| Backward Transfer | Can improve performance on old tasks |

**Metrics for Continual Learning:**
- Forward Transfer (FWT)
- Backward Transfer (BWT)
- Average Accuracy

---

### 2. Sparsey: Episodic + Semantic Memory via SDR
**arXiv:1710.07829**

Single-trial learning using sparse distributed representations.

| Feature | Benefit |
|---------|---------|
| Sparse Coding | Efficient, biologically plausible |
| Single-Trial | Fast memory formation |
| Similarity-Preserving | Generalization for free |
| Minutes on CPU | Edge-deployable |

**Relevance:** Fast object memory without GPU training.

---

### 3. Latent Learning via Episodic Memory
**arXiv:2509.16189**

Episodic memory enables learning information not immediately relevant.

| Problem | Solution |
|---------|----------|
| Reversal curse | Retrieval from memory |
| Task-specific blindness | Latent learning |
| Data inefficiency | Experience reuse |

**Key Insight:** In-context learning is critical for using retrieved memories.

---

## Integration with Platform

| Paper | Module | Application |
|-------|--------|-------------|
| GEM | `training_service.py` | Avoid forgetting during incremental updates |
| Sparsey | `object_cache.py` | Fast spatial object memory |
| Latent Learning | `SlamService` | Long-term scene memory |

### Implementation Ideas

```python
# GEM-style memory buffer for ObjectCache
class EpisodicBuffer:
    def __init__(self, max_per_class: int = 50):
        self.buffer = defaultdict(list)
    
    def add(self, label: str, embedding: Tensor):
        if len(self.buffer[label]) < self.max_per_class:
            self.buffer[label].append(embedding)
    
    def get_exemplars(self, label: str) -> List[Tensor]:
        return self.buffer.get(label, [])
```

---

## References

- [GEM](https://arxiv.org/abs/1706.08840)
- [Sparsey](https://arxiv.org/abs/1710.07829)
- [Latent Learning](https://arxiv.org/abs/2509.16189)
