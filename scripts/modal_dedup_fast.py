"""
Optimized Modal Deduplication with Embedding Caching

Run with: uv run modal run scripts/modal_dedup_fast.py --threshold 0.995

Speed optimizations:
- Embeddings cached to volume (reuse across runs)
- Configurable GPU (A10G, A100)
- Larger batch sizes
- Model selection (base/small)
"""

import modal
from pathlib import Path

app = modal.App("usd-dataset-dedup-fast")
volume = modal.Volume.from_name("usd-dataset-test", create_if_missing=False)

# Configuration
GPU_TYPE = "A100"  # Options: T4, A10G, A100 - easy to change here!
BATCH_SIZE = 128   # Larger for A100
MODEL_SIZE = "base"  # Options: small, base, large

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pillow",
        "tqdm",
        "numpy",
        "faiss-cpu",
    )
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,
    volumes={"/data": volume}
)
def generate_and_save_embeddings(
    model_name: str = "facebook/dinov2-base",
    batch_size: int = 128,
    force_regenerate: bool = False
):
    """Generate embeddings once, save to volume for reuse."""
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoImageProcessor, AutoModel
    from pathlib import Path
    
    embeddings_path = Path("/data/embeddings.npy")
    paths_path = Path("/data/image_paths.txt")
    
    # Check if already exists
    if embeddings_path.exists() and not force_regenerate:
        print("✓ Loading cached embeddings...")
        embeddings = np.load(embeddings_path)
        with open(paths_path) as f:
            paths = [line.strip() for line in f]
        return {
            "cached": True,
            "count": len(embeddings),
            "paths": paths
        }
    
    print(f"Generating embeddings on {GPU_TYPE}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    
    # Collect images
    data_dir = Path("/data")
    all_paths = []
    for split in ['test', 'train', 'valid']:
        split_dir = data_dir / split
        if split_dir.exists():
            paths = list(split_dir.glob("*.jpg"))
            all_paths.extend(paths)
    
    print(f"Processing {len(all_paths)} images...")
    
    # Generate embeddings
    embeddings = []
    valid_paths = []
    
    for i in tqdm(range(0, len(all_paths), batch_size), desc="Generating"):
        batch_paths = all_paths[i:i + batch_size]
        batch_images = []
        batch_valid = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                batch_valid.append(str(path))
            except Exception as e:
                print(f"Error: {path}: {e}")
        
        if not batch_images:
            continue
        
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_emb = outputs.last_hidden_state[:, 0].cpu().numpy()
        embeddings.append(batch_emb)
        valid_paths.extend(batch_valid)
    
    embeddings = np.vstack(embeddings)
    
    # Save to volume
    print("Saving embeddings to volume...")
    np.save(embeddings_path, embeddings)
    with open(paths_path, 'w') as f:
        f.write('\n'.join(valid_paths))
    
    # Commit changes to volume
    volume.commit()
    
    print(f"✓ Saved {len(embeddings)} embeddings")
    
    return {
        "cached": False,
        "count": len(embeddings),
        "paths": valid_paths
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=600,
    volumes={"/data": volume}
)
def find_duplicates(threshold: float = 0.995):
    """Fast duplicate detection using cached embeddings."""
    import numpy as np
    import faiss
    from collections import defaultdict
    from pathlib import Path
    
    print(f"Finding duplicates with threshold {threshold}...")
    
    # Load cached embeddings
    embeddings = np.load("/data/embeddings.npy")
    with open("/data/image_paths.txt") as f:
        valid_paths = [line.strip() for line in f]
    
    print(f"Loaded {len(embeddings)} embeddings")
    
    # FAISS duplicate detection
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings_norm.astype(np.float32)
    
    dim = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)
    
    k = min(10, len(embeddings_norm))
    D, I = index.search(embeddings_norm, k)
    
    # Find duplicates
    duplicates = []
    seen = set()
    
    for i in range(len(D)):
        for j in range(1, k):
            if D[i][j] > threshold:
                pair = tuple(sorted([i, I[i][j]]))
                if pair not in seen:
                    seen.add(pair)
                    duplicates.append({
                        'path1': valid_paths[pair[0]],
                        'path2': valid_paths[pair[1]],
                        'similarity': float(D[i][j])
                    })
    
    # Build graph
    graph = defaultdict(set)
    for dup in duplicates:
        graph[dup['path1']].add(dup['path2'])
        graph[dup['path2']].add(dup['path1'])
    
    visited = set()
    components = []
    
    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for path in valid_paths:
        if path not in visited:
            component = []
            dfs(path, component)
            components.append(component)
    
    # Select keepers
    to_keep = []
    to_remove = []
    
    for component in components:
        if len(component) == 1:
            to_keep.append(component[0])
        else:
            keeper = sorted(component)[0]
            to_keep.append(keeper)
            to_remove.extend([p for p in component if p != keeper])
    
    return {
        "total_images": len(valid_paths),
        "duplicates_found": len(duplicates),
        "duplicate_clusters": len([c for c in components if len(c) > 1]),
        "to_keep": len(to_keep),
        "to_remove": len(to_remove),
        "removal_percentage": f"{len(to_remove)/max(len(valid_paths),1)*100:.1f}%",
        "duplicates": duplicates[:100],
        "remove_list": to_remove
    }


@app.local_entrypoint()
def main(
    threshold: float = 0.995,
    model: str = f"facebook/dinov2-{MODEL_SIZE}",
    regenerate: bool = False
):
    """
    Fast deduplication with caching.
    
    Args:
        threshold: Similarity threshold (0.99-0.999)
        model: DINOv2 model (small/base/large)
        regenerate: Force regenerate embeddings
    """
    print(f"🚀 Fast Dedup (GPU: {GPU_TYPE}, Batch: {BATCH_SIZE})")
    print(f"Settings: threshold={threshold}, model={model}\n")
    
    # Step 1: Generate/load embeddings (slow first time, fast after)
    emb_result = generate_and_save_embeddings.remote(
        model_name=model,
        batch_size=BATCH_SIZE,
        force_regenerate=regenerate
    )
    
    if emb_result['cached']:
        print(f"⚡ Reused cached embeddings ({emb_result['count']} images)\n")
    else:
        print(f"💾 Generated new embeddings ({emb_result['count']} images)\n")
    
    # Step 2: Find duplicates (always fast)
    result = find_duplicates.remote(threshold=threshold)
    
    print("\n=== Results ===")
    print(f"Total images: {result['total_images']}")
    print(f"Duplicates: {result['duplicates_found']} pairs")
    print(f"Clusters: {result['duplicate_clusters']}")
    print(f"To keep: {result['to_keep']}")
    print(f"To remove: {result['to_remove']} ({result['removal_percentage']})")
    
    if result['duplicates']:
        print("\n=== Sample Duplicates ===")
        for dup in result['duplicates'][:10]:
            p1 = Path(dup['path1']).name[:40]
            p2 = Path(dup['path2']).name[:40]
            print(f"  {dup['similarity']:.4f}: {p1} <-> {p2}")
    
    # Save results
    import json
    with open("dedup_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Saved to dedup_results.json")
