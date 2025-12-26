# Claude/z.ai Integration for USD Training Pipeline

> **Status**: NOT IMPLEMENTED - Documentation Only

## API Setup

```bash
# Add Claude API key to Modal
modal secret create claude-secret ANTHROPIC_API_KEY=<your-key>
```

## Potential Integration Points

### 1. Training Monitor
Claude analyzes loss curves during/after training and suggests adjustments:
- Detect overfitting early
- Recommend learning rate changes
- Flag anomalies in metrics

### 2. Prediction QA
After training, Claude reviews model predictions for quality:
- Sample predictions with confidence scores
- Compare against ground truth
- Identify systematic errors

### 3. Error Analysis
Claude explains why certain detections failed:
- False positives analysis
- Missed detections
- Confusion between similar classes (e.g., $1 vs $2 bills)

### 4. Dataset Audit
Claude reviews sample images for labeling issues:
- Incorrect class labels
- Missing annotations
- Poor quality images

## Implementation Pattern

```python
# Add to Modal function
@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("claude-secret"),  # Add this
    ],
)
def train_with_claude_monitoring():
    import anthropic
    import os
    
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    # Use Claude to analyze results
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Analyze these training metrics..."}]
    )
```

## Files Created (Not Active)

- `scripts/train_yolo11_claude.py` - Training with Claude monitoring
- `scripts/claude_evaluator.py` - Post-training evaluation

## Next Steps (When Ready)

1. Create `claude-secret` in Modal with z.ai API key
2. Decide on integration point (monitoring vs post-analysis)
3. Implement and test
