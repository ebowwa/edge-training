#!/usr/bin/env python3
"""
USD Dataset Deduplication using DINOv2 Embeddings

Detects and removes near-duplicate images from the USD bill dataset
using cosine similarity on DINOv2 embeddings.

Usage:
    python scripts/deduplicate_dataset.py --dataset-dir datasets/usd_detection --threshold 0.99
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not installed. Using sklearn for similarity.")

from transformers import AutoImageProcessor, AutoModel


class DINOv2Deduplicator:
    """Deduplicate images using DINOv2 embeddings."""
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = None,
        batch_size: int = 32,
        threshold: float = 0.99
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.threshold = threshold
        
        print(f"Loading {model_name} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        
        # Compile for speedup if available
        if hasattr(torch, 'compile') and self.device == "cuda":
            self.model = torch.compile(self.model)
    
    def load_image(self, path: Path) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def generate_embeddings(self, image_paths: list) -> np.ndarray:
        """Generate embeddings for all images."""
        embeddings = []
        valid_paths = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Generating embeddings"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            batch_valid_paths = []
            
            for path in batch_paths:
                img = self.load_image(path)
                if img is not None:
                    batch_images.append(img)
                    batch_valid_paths.append(path)
            
            if not batch_images:
                continue
            
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use CLS token embedding
            batch_emb = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.append(batch_emb)
            valid_paths.extend(batch_valid_paths)
        
        return np.vstack(embeddings), valid_paths
    
    def find_duplicates_faiss(self, embeddings: np.ndarray, image_paths: list) -> list:
        """Find duplicates using FAISS (fast)."""
        # L2 normalize for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings_norm.astype(np.float32)
        
        dim = embeddings_norm.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized
        index.add(embeddings_norm)
        
        # Search k nearest neighbors
        k = min(10, len(embeddings_norm))
        D, I = index.search(embeddings_norm, k)
        
        # Find pairs above threshold
        duplicates = []
        seen = set()
        
        for i in range(len(D)):
            for j in range(1, k):  # Skip self (index 0)
                if D[i][j] > self.threshold:
                    pair = tuple(sorted([i, I[i][j]]))
                    if pair not in seen:
                        seen.add(pair)
                        duplicates.append({
                            'idx1': pair[0],
                            'idx2': pair[1],
                            'path1': str(image_paths[pair[0]]),
                            'path2': str(image_paths[pair[1]]),
                            'similarity': float(D[i][j])
                        })
        
        return duplicates
    
    def find_duplicates_sklearn(self, embeddings: np.ndarray, image_paths: list) -> list:
        """Find duplicates using sklearn (slower, no GPU)."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("Computing similarity matrix (this may take a while)...")
        sim_matrix = cosine_similarity(embeddings)
        
        # Find pairs above threshold
        duplicates = []
        seen = set()
        
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                if sim_matrix[i][j] > self.threshold:
                    duplicates.append({
                        'idx1': i,
                        'idx2': j,
                        'path1': str(image_paths[i]),
                        'path2': str(image_paths[j]),
                        'similarity': float(sim_matrix[i][j])
                    })
        
        return duplicates
    
    def find_duplicates(self, embeddings: np.ndarray, image_paths: list) -> list:
        """Find duplicate pairs."""
        if HAS_FAISS:
            return self.find_duplicates_faiss(embeddings, image_paths)
        else:
            return self.find_duplicates_sklearn(embeddings, image_paths)
    
    def build_duplicate_graph(self, duplicates: list) -> dict:
        """Build graph of duplicate relationships."""
        graph = defaultdict(set)
        for dup in duplicates:
            graph[dup['path1']].add(dup['path2'])
            graph[dup['path2']].add(dup['path1'])
        return graph
    
    def find_connected_components(self, graph: dict, all_paths: list) -> list:
        """Find connected components (duplicate clusters)."""
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for path in all_paths:
            if path not in visited:
                component = []
                dfs(str(path), component)
                components.append(component)
        
        return components
    
    def get_class_from_path(self, path: str, annotations: dict = None) -> str:
        """Extract class from image path or annotations."""
        # Try to get from annotations if provided
        if annotations:
            filename = os.path.basename(path)
            if filename in annotations:
                return annotations[filename]
        
        # Fallback: extract from filename pattern
        filename = os.path.basename(path).lower()
        for denom in ['100', '50', '20', '10', '5', '1']:
            if f'{denom}usd' in filename or f'{denom}dollar' in filename:
                for side in ['front', 'back']:
                    if side in filename:
                        return f'{denom}USD-{side.capitalize()}'
                return f'{denom}USD'
        
        return 'unknown'
    
    def select_keepers(
        self,
        components: list,
        class_counts: dict = None,
        annotations: dict = None
    ) -> tuple:
        """Select which images to keep from duplicate clusters."""
        to_keep = []
        to_remove = []
        
        for component in components:
            if len(component) == 1:
                # Unique image
                to_keep.append(component[0])
            else:
                # Duplicate cluster - select keeper
                if class_counts:
                    # Prefer keeper from underrepresented class
                    component_classes = [
                        (p, self.get_class_from_path(p, annotations))
                        for p in component
                    ]
                    component_classes.sort(
                        key=lambda x: class_counts.get(x[1], 0)
                    )
                    keeper = component_classes[0][0]
                else:
                    # Default: keep first alphabetically
                    keeper = sorted(component)[0]
                
                to_keep.append(keeper)
                to_remove.extend([p for p in component if p != keeper])
        
        return to_keep, to_remove


def load_annotations(dataset_dir: Path) -> dict:
    """Load class info from COCO annotations."""
    annotations = {}
    
    for split in ['train', 'valid', 'test']:
        ann_file = dataset_dir / split / '_annotations.coco.json'
        if not ann_file.exists():
            continue
        
        with open(ann_file) as f:
            coco = json.load(f)
        
        cat_map = {c['id']: c['name'] for c in coco['categories']}
        img_map = {img['id']: img['file_name'] for img in coco['images']}
        
        # Get first annotation per image
        for ann in coco['annotations']:
            img_file = img_map.get(ann['image_id'])
            if img_file and img_file not in annotations:
                annotations[img_file] = cat_map.get(ann['category_id'], 'unknown')
    
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Deduplicate USD dataset")
    parser.add_argument("--dataset-dir", type=str, default="datasets/usd_detection")
    parser.add_argument("--model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="dedup_results")
    parser.add_argument("--dry-run", action="store_true", help="Don't remove files")
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Collect all images
    print("Collecting images...")
    image_paths = []
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_dir / split
        if split_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(split_dir.glob(ext))
    
    print(f"Found {len(image_paths)} images")
    
    # Load annotations for class info
    annotations = load_annotations(dataset_dir)
    print(f"Loaded annotations for {len(annotations)} images")
    
    # Initialize deduplicator
    dedup = DINOv2Deduplicator(
        model_name=args.model,
        batch_size=args.batch_size,
        threshold=args.threshold
    )
    
    # Generate embeddings
    embeddings, valid_paths = dedup.generate_embeddings(image_paths)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)
    with open(output_dir / "image_paths.json", 'w') as f:
        json.dump([str(p) for p in valid_paths], f, indent=2)
    print(f"Saved embeddings to {output_dir}")
    
    # Find duplicates
    print(f"Finding duplicates (threshold={args.threshold})...")
    duplicates = dedup.find_duplicates(embeddings, valid_paths)
    print(f"Found {len(duplicates)} duplicate pairs")
    
    # Save duplicate pairs
    with open(output_dir / "duplicates.json", 'w') as f:
        json.dump(duplicates, f, indent=2)
    
    with open(output_dir / "duplicates.txt", 'w') as f:
        for dup in sorted(duplicates, key=lambda x: -x['similarity']):
            f.write(f"{dup['path1']} <-> {dup['path2']} : {dup['similarity']:.4f}\n")
    
    # Build graph and find clusters
    graph = dedup.build_duplicate_graph(duplicates)
    components = dedup.find_connected_components(graph, [str(p) for p in valid_paths])
    
    # Count clusters
    dup_clusters = [c for c in components if len(c) > 1]
    print(f"Found {len(dup_clusters)} duplicate clusters")
    
    # Count classes for balance-aware selection
    class_counts = defaultdict(int)
    for path in valid_paths:
        cls = dedup.get_class_from_path(str(path), annotations)
        class_counts[cls] += 1
    
    # Select keepers
    to_keep, to_remove = dedup.select_keepers(components, class_counts, annotations)
    
    print(f"\n=== Results ===")
    print(f"Total images: {len(valid_paths)}")
    print(f"To keep: {len(to_keep)}")
    print(f"To remove: {len(to_remove)} ({len(to_remove)/len(valid_paths)*100:.1f}%)")
    
    # Save lists
    with open(output_dir / "keep_images.txt", 'w') as f:
        for p in sorted(to_keep):
            f.write(p + "\n")
    
    with open(output_dir / "remove_images.txt", 'w') as f:
        for p in sorted(to_remove):
            f.write(p + "\n")
    
    print(f"\nResults saved to {output_dir}/")
    print("  - embeddings.npy")
    print("  - duplicates.json / duplicates.txt")
    print("  - keep_images.txt")
    print("  - remove_images.txt")
    
    if args.dry_run:
        print("\n[DRY RUN] No files removed. Review results and run without --dry-run to apply.")
    else:
        # Apply removal
        print("\nApply removal? (y/n): ", end="")
        if input().strip().lower() == 'y':
            backup_dir = output_dir / "backup"
            backup_dir.mkdir(exist_ok=True)
            
            for path in tqdm(to_remove, desc="Moving to backup"):
                src = Path(path)
                if src.exists():
                    dst = backup_dir / src.name
                    src.rename(dst)
            
            print(f"Moved {len(to_remove)} files to {backup_dir}/")


if __name__ == "__main__":
    main()
