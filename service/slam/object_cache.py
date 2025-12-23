"""
Persistent Object Cache for SLAM.
Provides temporal memory for tracking objects across frames.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


@dataclass
class CachedObject:
    """An object tracked across frames."""
    id: int
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, w, h normalized
    spatial_coords: Optional[Tuple[float, float, float]] = None  # 3D position
    embedding: Optional[List[float]] = None  # VLM/feature embedding
    
    # Temporal tracking
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 1
    
    # Spatial tracking
    anchor_id: Optional[int] = None
    
    def update(self, confidence: float, bbox: Tuple[float, float, float, float],
               spatial_coords: Optional[Tuple[float, float, float]] = None):
        """Update object with new observation."""
        self.confidence = max(self.confidence, confidence)  # Keep max confidence
        self.bbox = bbox
        self.last_seen = time.time()
        self.frame_count += 1
        if spatial_coords:
            self.spatial_coords = spatial_coords
    
    @property
    def age_seconds(self) -> float:
        """Time since first seen."""
        return time.time() - self.first_seen
    
    @property
    def staleness_seconds(self) -> float:
        """Time since last seen."""
        return time.time() - self.last_seen
    
    def is_stale(self, max_staleness: float = 5.0) -> bool:
        """Check if object hasn't been seen recently."""
        return self.staleness_seconds > max_staleness


class ObjectCache:
    """
    Thread-safe persistent cache for tracked objects.
    Implements LRU eviction with staleness-based cleanup.
    """
    
    def __init__(self, max_objects: int = 1000, max_staleness: float = 30.0):
        """
        Args:
            max_objects: Maximum objects to keep in cache.
            max_staleness: Seconds before an object is considered stale.
        """
        self.max_objects = max_objects
        self.max_staleness = max_staleness
        
        self._cache: OrderedDict[int, CachedObject] = OrderedDict()
        self._next_id = 0
        self._lock = threading.RLock()
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _generate_id(self) -> int:
        """Generate unique object ID."""
        self._next_id += 1
        return self._next_id
    
    def add(self, label: str, confidence: float, 
            bbox: Tuple[float, float, float, float],
            spatial_coords: Optional[Tuple[float, float, float]] = None,
            embedding: Optional[List[float]] = None) -> CachedObject:
        """
        Add a new object to the cache.
        
        Returns:
            The cached object.
        """
        with self._lock:
            obj = CachedObject(
                id=self._generate_id(),
                label=label,
                confidence=confidence,
                bbox=bbox,
                spatial_coords=spatial_coords,
                embedding=embedding,
            )
            self._cache[obj.id] = obj
            
            # Evict if over capacity
            while len(self._cache) > self.max_objects:
                self._evict_oldest()
            
            return obj
    
    def get(self, obj_id: int) -> Optional[CachedObject]:
        """Get object by ID."""
        with self._lock:
            obj = self._cache.get(obj_id)
            if obj:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(obj_id)
            else:
                self._misses += 1
            return obj
    
    def update(self, obj_id: int, confidence: float,
               bbox: Tuple[float, float, float, float],
               spatial_coords: Optional[Tuple[float, float, float]] = None) -> bool:
        """
        Update an existing object.
        
        Returns:
            True if object was found and updated.
        """
        with self._lock:
            obj = self._cache.get(obj_id)
            if obj:
                obj.update(confidence, bbox, spatial_coords)
                self._cache.move_to_end(obj_id)
                return True
            return False
    
    def find_by_label(self, label: str) -> List[CachedObject]:
        """Find all objects with a given label."""
        with self._lock:
            return [obj for obj in self._cache.values() if obj.label == label]
    
    def find_nearby(self, coords: Tuple[float, float, float], 
                    radius: float = 0.5) -> List[CachedObject]:
        """Find objects within radius of 3D coordinates."""
        with self._lock:
            results = []
            for obj in self._cache.values():
                if obj.spatial_coords:
                    dist = sum((a - b) ** 2 for a, b in zip(coords, obj.spatial_coords)) ** 0.5
                    if dist <= radius:
                        results.append(obj)
            return results
    
    def find_by_iou(self, bbox: Tuple[float, float, float, float], 
                    min_iou: float = 0.5) -> Optional[CachedObject]:
        """
        Find object with highest IoU overlap to given bbox.
        Used for re-identification across frames.
        """
        with self._lock:
            best_obj = None
            best_iou = min_iou
            
            for obj in self._cache.values():
                if obj.is_stale(self.max_staleness):
                    continue
                iou = self._compute_iou(bbox, obj.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_obj = obj
            
            return best_obj
    
    @staticmethod
    def _compute_iou(box1: Tuple[float, float, float, float],
                     box2: Tuple[float, float, float, float]) -> float:
        """Compute IoU between two xywh boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to xyxy
        b1_x1, b1_y1 = x1 - w1/2, y1 - h1/2
        b1_x2, b1_y2 = x1 + w1/2, y1 + h1/2
        b2_x1, b2_y1 = x2 - w2/2, y2 - h2/2
        b2_x2, b2_y2 = x2 + w2/2, y2 + h2/2
        
        # Intersection
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        return inter_area / union_area
    
    def cleanup_stale(self) -> int:
        """Remove stale objects. Returns count of removed objects."""
        with self._lock:
            stale_ids = [
                obj_id for obj_id, obj in self._cache.items()
                if obj.is_stale(self.max_staleness)
            ]
            for obj_id in stale_ids:
                del self._cache[obj_id]
            return len(stale_ids)
    
    def _evict_oldest(self):
        """Evict least recently used object."""
        if self._cache:
            self._cache.popitem(last=False)
            self._evictions += 1
    
    def clear(self):
        """Clear all cached objects."""
        with self._lock:
            self._cache.clear()
            self._next_id = 0
    
    def get_all(self, include_stale: bool = False) -> List[CachedObject]:
        """Get all cached objects."""
        with self._lock:
            if include_stale:
                return list(self._cache.values())
            return [obj for obj in self._cache.values() 
                    if not obj.is_stale(self.max_staleness)]
    
    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)
    
    @property
    def stats(self) -> dict:
        """Cache statistics."""
        return {
            'size': self.size,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate': self._hits / max(1, self._hits + self._misses),
        }
    
    def __len__(self) -> int:
        return self.size
    
    def __contains__(self, obj_id: int) -> bool:
        return obj_id in self._cache


# Singleton instance for global access
_global_cache: Optional[ObjectCache] = None


def get_object_cache(max_objects: int = 1000, max_staleness: float = 30.0) -> ObjectCache:
    """Get or create the global object cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ObjectCache(max_objects, max_staleness)
    return _global_cache
