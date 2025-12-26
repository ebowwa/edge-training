"""
Modal-based USD Dataset Deduplication using DINOv2 Embeddings

Uses volume-mounted data from usd-dataset-test.
Run with: uv run modal run scripts/modal_dedup.py
"""

import modal

app = modal.App("usd-dataset-dedup")

# Reference existing volume
volume = modal.Volume.from_name("usd-dataset-test", create_if_missing=False)

# Lightweight image (no dataset embedded)
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
    gpu="A10G",
    timeout=3600,
    volumes={"/data": volume}  # Mount volume
)
def run_dedup(
    threshold: float = 0.99,
    model_name: str = "facebook/dinov2-base",
    batch_size: int = 64,
):
    """Run deduplication on Modal GPU with volume-mounted data."""
    import json
    import numpy as np
    import torch
    import faiss
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoImageProcessor, AutoModel
    from collections import defaultdict
    from pathlib import Path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    # Load model
    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    
    # Collect images from volume
    data_dir = Path("/data")
    all_paths = []
    for split in ['test', 'train', 'valid']:
        split_dir = data_dir / split
        if split_dir.exists():
            paths = list(split_dir.glob("*.jpg"))
            print(f"{split}: {len(paths)} images")
            all_paths.extend(paths)
    
    print(f"Total: {len(all_paths)} images")
    
    if not all_paths:
        return {"error": "No images found"}
    
    # Generate embeddings
    embeddings = []
    valid_paths = []
    
    for i in tqdm(range(0, len(all_paths), batch_size), desc="Generating embeddings"):
        batch_paths = all_paths[i:i + batch_size]
        batch_images = []
        batch_valid = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                batch_valid.append(str(path))
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if not batch_images:
            continue
        
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_emb = outputs.last_hidden_state[:, 0].cpu().numpy()
        embeddings.append(batch_emb)
        valid_paths.extend(batch_valid)
    
    embeddings = np.vstack(embeddings)
    print(f"Generated {len(embeddings)} embeddings")
    
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
    
    print(f"Found {len(duplicates)} duplicate pairs")
    
    # Build graph and find clusters
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
    
    # Select keepers (keep first alphabetically)
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
    threshold: float = 0.99,
    model: str = "facebook/dinov2-base",
    batch_size: int = 64
):
    """Run USD dataset deduplication."""
    print(f"Starting dedup: threshold={threshold}, model={model}")
    
    result = run_dedup.remote(threshold=threshold, model_name=model, batch_size=batch_size)
    
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return
    
    print("\n=== Results ===")
    print(f"Total images: {result['total_images']}")
    print(f"Duplicate pairs: {result['duplicates_found']}")
    print(f"Duplicate clusters: {result['duplicate_clusters']}")
    print(f"To keep: {result['to_keep']}")
    print(f"To remove: {result['to_remove']} ({result['removal_percentage']})")
    
    if result.get('duplicates'):
        print("\n=== Sample Duplicates ===")
        for dup in result['duplicates'][:10]:
            print(f"  {dup['path1']} <-> {dup['path2']} ({dup['similarity']:.4f})")
    
    # Save results
    import json
    with open("dedup_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved to dedup_results.json")
