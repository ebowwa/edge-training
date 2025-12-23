"""
Tests for ObjectCache.
"""

import unittest
import time
from service.slam.object_cache import ObjectCache, CachedObject, get_object_cache


class TestCachedObject(unittest.TestCase):
    """Tests for CachedObject dataclass."""
    
    def test_creation(self):
        """Test object creation."""
        obj = CachedObject(
            id=1,
            label="person",
            confidence=0.9,
            bbox=(0.5, 0.5, 0.1, 0.2),
        )
        self.assertEqual(obj.label, "person")
        self.assertEqual(obj.confidence, 0.9)
        self.assertEqual(obj.frame_count, 1)
    
    def test_update(self):
        """Test object update."""
        obj = CachedObject(id=1, label="car", confidence=0.7, bbox=(0.3, 0.3, 0.2, 0.2))
        obj.update(0.9, (0.4, 0.4, 0.2, 0.2))
        
        self.assertEqual(obj.confidence, 0.9)  # Max confidence
        self.assertEqual(obj.bbox, (0.4, 0.4, 0.2, 0.2))
        self.assertEqual(obj.frame_count, 2)
    
    def test_staleness(self):
        """Test staleness detection."""
        obj = CachedObject(id=1, label="dog", confidence=0.8, bbox=(0.5, 0.5, 0.1, 0.1))
        self.assertFalse(obj.is_stale(max_staleness=5.0))
        
        # Simulate staleness by backdating last_seen
        obj.last_seen = time.time() - 10.0
        self.assertTrue(obj.is_stale(max_staleness=5.0))


class TestObjectCache(unittest.TestCase):
    """Tests for ObjectCache."""
    
    def setUp(self):
        self.cache = ObjectCache(max_objects=10, max_staleness=5.0)
    
    def test_add_and_get(self):
        """Test adding and retrieving objects."""
        obj = self.cache.add("person", 0.9, (0.5, 0.5, 0.1, 0.1))
        self.assertIsNotNone(obj.id)
        
        retrieved = self.cache.get(obj.id)
        self.assertEqual(retrieved.label, "person")
    
    def test_update(self):
        """Test updating existing object."""
        obj = self.cache.add("car", 0.7, (0.3, 0.3, 0.2, 0.2))
        success = self.cache.update(obj.id, 0.95, (0.4, 0.4, 0.2, 0.2))
        
        self.assertTrue(success)
        updated = self.cache.get(obj.id)
        self.assertEqual(updated.confidence, 0.95)
    
    def test_find_by_label(self):
        """Test finding objects by label."""
        self.cache.add("person", 0.9, (0.1, 0.1, 0.1, 0.1))
        self.cache.add("car", 0.8, (0.2, 0.2, 0.1, 0.1))
        self.cache.add("person", 0.85, (0.3, 0.3, 0.1, 0.1))
        
        persons = self.cache.find_by_label("person")
        self.assertEqual(len(persons), 2)
    
    def test_find_by_iou(self):
        """Test IoU-based re-identification."""
        self.cache.add("person", 0.9, (0.5, 0.5, 0.1, 0.1))
        
        # Similar bbox should match
        match = self.cache.find_by_iou((0.51, 0.51, 0.1, 0.1), min_iou=0.5)
        self.assertIsNotNone(match)
        
        # Very different bbox should not match
        no_match = self.cache.find_by_iou((0.9, 0.9, 0.1, 0.1), min_iou=0.5)
        self.assertIsNone(no_match)
    
    def test_eviction(self):
        """Test LRU eviction."""
        cache = ObjectCache(max_objects=3, max_staleness=30.0)
        
        obj1 = cache.add("a", 0.9, (0.1, 0.1, 0.1, 0.1))
        obj2 = cache.add("b", 0.9, (0.2, 0.2, 0.1, 0.1))
        obj3 = cache.add("c", 0.9, (0.3, 0.3, 0.1, 0.1))
        obj4 = cache.add("d", 0.9, (0.4, 0.4, 0.1, 0.1))  # Should evict obj1
        
        self.assertEqual(cache.size, 3)
        self.assertIsNone(cache.get(obj1.id))  # obj1 was evicted
        self.assertIsNotNone(cache.get(obj2.id))
    
    def test_cleanup_stale(self):
        """Test stale object cleanup."""
        obj = self.cache.add("stale", 0.9, (0.5, 0.5, 0.1, 0.1))
        obj.last_seen = time.time() - 10.0  # Make stale
        
        removed = self.cache.cleanup_stale()
        self.assertEqual(removed, 1)
        self.assertIsNone(self.cache.get(obj.id))
    
    def test_stats(self):
        """Test cache statistics."""
        obj = self.cache.add("test", 0.9, (0.5, 0.5, 0.1, 0.1))
        self.cache.get(obj.id)  # Hit
        self.cache.get(999)  # Miss
        
        stats = self.cache.stats
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['hit_rate'], 0.5)


class TestIoUComputation(unittest.TestCase):
    """Tests for IoU computation."""
    
    def test_perfect_overlap(self):
        """Test IoU = 1.0 for identical boxes."""
        iou = ObjectCache._compute_iou(
            (0.5, 0.5, 0.2, 0.2),
            (0.5, 0.5, 0.2, 0.2)
        )
        self.assertAlmostEqual(iou, 1.0)
    
    def test_no_overlap(self):
        """Test IoU = 0.0 for non-overlapping boxes."""
        iou = ObjectCache._compute_iou(
            (0.1, 0.1, 0.1, 0.1),
            (0.9, 0.9, 0.1, 0.1)
        )
        self.assertAlmostEqual(iou, 0.0)
    
    def test_partial_overlap(self):
        """Test IoU for partially overlapping boxes."""
        iou = ObjectCache._compute_iou(
            (0.5, 0.5, 0.2, 0.2),
            (0.55, 0.55, 0.2, 0.2)
        )
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
