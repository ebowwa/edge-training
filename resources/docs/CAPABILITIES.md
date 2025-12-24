# 🧠 Advanced Training Capabilities

Your `yolo-training` system is now upgraded with cognitive and spatial awareness features. These capabilities move beyond standard object detection, enabling the training of **Intelligent Egocentric Assistants**.

## 1. Gaze-Prioritized Learning
Train models that prioritize objects based on user attention.

- **Use Case**: Personal Assistant that learns "what matters to you."
- **Example**: If you stare at a specific tool while repairing a bike, the model learns to detect that tool with higher confidence than background clutter.
- **How it works**:
  - `GazeIntentDecoder`: Infers user attention from eye tracking.
  - `DetectionLoss`: Weights gradients higher for objects in the gaze fixation region.
  - **Result**: Higher recall on attended objects, lower false positives on background noise.

## 2. Dense Scene & Crowd Counting
Train models capable of counting heavily occluded objects.

- **Use Case**: Inventory audits (screws in a bin), biological surveys (bees in a hive), crowd safety.
- **Example**: Counting 500+ bees on a frame where bodies overlap by 80%.
- **How it works**:
  - `DensityHead`: Predicts a per-pixel density map instead of just boxes.
  - `SimOTA (Simplified Optimal Transport Assignment)`: Intelligently assigns multiple positive anchors to dense clusters, preventing NMS (Non-Maximum Suppression) from deleting valid detections.
  - **Result**: Accurate counts in scenes where standard YOLO fails due to overlap.

## 3. World-Locked Egocentric Tracking
Train models that understand 3D object permanence.

- **Use Case**: "Memory Palace" — remembering where things are even when you look away.
- **Example**: You spot your keys on the table, turn your head, and the system remembers their exact 3D location. When you look back, they are re-identified instantly.
- **How it works**:
  - **Optical Flow SLAM**: Tracks camera motion (6DoF pose) using lightweight Lucas-Kanade tracking.
  - **Spatial Anchors**: Locks detections to 3D world coordinates (`world_coords`).
  - **Re-Identification**: Uses `find_anchor_by_bbox` to link new observations to existing 3D memories.
  - **Result**: Stable, flicker-free tracking on wearable devices (glasses).

## 4. Few-Shot Personal Object Learning *(TODO)*
Train custom models with extremely limited data (5-10 images).

> **Note:** This feature has stub code but is not yet integrated into TrainingConfig.

- **Use Case**: Personalized object detection ("My specific blue mug").
- **Example**: Teaching the glasses to recognize your specific medication bottle.
- **How it works** *(planned)*:
  - **PEFT (Parameter-Efficient Fine-Tuning)**: Uses **LoRA** (Low-Rank Adaptation) or **BitFit**.
  - **Efficiency**: Freezes 99% of the massive YOLO backbone and only trains a tiny adapter layer.
  - **Result**: Train a custom model in minutes on a consumer GPU without "catastrophic forgetting" of general knowledge.

## 5. Interactive Counting (Human-in-the-Loop)
Adapt counting models in real-time based on user feedback.

- **Use Case**: Inventory audits where initial counts are wrong.
- **Example**: The model predicts 47 screws, but you can see it's actually ~50-55. You draw a box around the pile and say "this region has 50-55 objects." The model instantly adapts.
- **How it works**:
  - **Based on**: [Interactive Class-Agnostic Object Counting](https://arxiv.org/abs/2309.05277)
  - **InteractiveCountingLoss**: A "soft boundary" loss that penalizes predictions outside user-specified count ranges.
  - **CountingRefinementAdapter**: A lightweight adapter (~10K params) attached to the density head. Only this adapter is updated during interaction—the base model stays frozen.
  - **Result**: 30-40% reduction in counting error with minimal user input (just 2-3 corrections per image).

### Enable SLAM & Gaze in Pipeline
```python
from pipeline.spatial_inference import SpatialInferencePipeline

pipeline = SpatialInferencePipeline(
    model_path="resources/best.pt",
    enable_slam=True,     # Enables Optical Flow & 3D Anchoring
    imu_enabled=True      # Fuses IMU data if available
)

# Run inference
results = pipeline.process(video_stream)
for det in results.detections:
    print(f"Object: {det.class_name}")
    print(f"3D Location: {det.world_coords}")  # Persistent 3D coord
    print(f"Gaze Priority: {det.gaze_boost}")  # Attention weight
```

### Train with PEFT (Low Memory) *(TODO - Not Yet Implemented)*
```python
# TODO: Add use_peft and peft_method to TrainingConfig
# from service.training_service import TrainingService, TrainingConfig
#
# config = TrainingConfig(
#     use_peft=True,
#     peft_method="lora",  # or "bitfit"
#     epochs=50
# )
# TrainingService.train("my_dataset.yaml", config)

# For now, use the stub functions directly:
from service.layers.peft import freeze_base_model, get_trainable_params
freeze_base_model(model)  # This works
print(f"Trainable params: {get_trainable_params(model)}")  # This works
```

### Interactive Counting with User Feedback
```python
from service.layers import (
    DensityHead, CountingRefinementAdapter, 
    InteractiveCountingLoss, RangeFeedback
)

# Base counting model (frozen during adaptation)
density_head = DensityHead(in_channels=[64, 128, 256])
density_map = density_head(features)  # Initial prediction

# Attach lightweight refinement adapter
adapter = CountingRefinementAdapter(in_channels=1)

# User provides feedback: "this region has 50-55 objects"
feedback = [
    RangeFeedback(x1=0.2, y1=0.3, x2=0.8, y2=0.7, min_count=50, max_count=55)
]

# Adapt in real-time (only adapter params update)
refined_density = adapter.adapt(density_map, feedback, steps=10)
final_count = refined_density.sum().item()  # Corrected count
```

---

## 📚 References

1. [Interactive Class-Agnostic Object Counting](https://arxiv.org/abs/2309.05277) - Huang et al., 2023
2. [YOLOX: Exceeding YOLO Series](https://arxiv.org/abs/2107.08430) - SimOTA Assignment
3. [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) - PEFT Method

