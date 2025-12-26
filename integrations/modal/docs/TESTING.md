# Testing Modal Integration

## Quick Tests

### 1. Test GPU Access & Retry Logic

```bash
# Run basic tests (no dataset upload needed)
uv run modal run scripts/test_modal.py
```

This tests:
- ✅ GPU availability (T4 for cost efficiency)
- ✅ Retry decorator works
- ✅ Modal app creation

### 2. Test with Small Subset

Edit `modal_dedup.py` to only process test split:

```python
for split in ['test']:  # Only test split (350 images)
    split_dir = data_dir / split
```

Then run:

```bash
uv run modal run scripts/modal_dedup.py --threshold 0.99
```

This uploads **~1.5GB** (test split only) to verify:
- Dataset embedding works
- DINOv2 loads on GPU
- FAISS duplicate detection runs
- Results are returned

### 3. Full Dedup

Once test split works, restore full code:

```python
for split in ['train', 'valid', 'test']:  # All splits
```

Run full dedup:

```bash
uv run modal run scripts/modal_dedup.py --threshold 0.99
```

## If Connection Drops Again

The Modal integration now has:
- Automatic retries (up to 5x per upload)
- Better error messages
- Upload time estimation

But if network is still unstable:

**Option 1: Smaller batches**
```python
batch_size: int = 32,  # Reduce from 64
```

**Option 2: Lower GPU**
```python
gpu_type: str = "T4",  # Cheaper, may help with availability
```

**Option 3: Different time**
Try running during off-peak hours when Modal has better availability.

## Expected Output

Successful run should show:

```
Running on cuda
Loading facebook/dinov2-base...
test: 350 images
Total: 350 images
Generating embeddings: 100%|███| 6/6 [00:15<00:00]
Generated 350 embeddings
Found 15 duplicate pairs

=== Results ===
Total images: 350
Duplicate pairs: 15
To remove: 12 (3.4%)
```

Results saved to `dedup_results.json`.

## Debugging

If it fails, check:

```bash
# Check Modal apps
uv run modal app list

# Check logs
uv run modal app logs <app-id>

# Check volume (if using)
uv run modal volume ls usd-dataset
```
