"""
Tests for Spatial Inference Pipeline.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add paths
_test_dir = Path(__file__).parent
_pipeline_dir = _test_dir.parent
_root_dir = _pipeline_dir.parent
_service_dir = _root_dir / "service"
_slam_dir = _service_dir / "slam"

sys.path.insert(0, str(_pipeline_dir))
sys.path.insert(0, str(_service_dir))
sys.path.insert(0, str(_slam_dir))

from spatial_inference import (
    SpatialDetection,
    SpatialInferenceResult,
    SpatialInferencePipeline,
)
from slam_service import DevicePose, SpatialAnchor


class TestSpatialDetection(unittest.TestCase):
    """Tests for SpatialDetection dataclass."""
    
    def test_creation(self):
        """Test creating a spatial detection."""
        det = SpatialDetection(
            class_id=0,
            class_name="pothole",
            confidence=0.85,
            bbox=(100, 200, 150, 250),
            bbox_normalized=(0.5, 0.6, 0.1, 0.1),
            anchor_id=1,
            spatial_coords=(0.5, 0.6),
            is_new=True,
        )
        
        self.assertEqual(det.class_name, "pothole")
        self.assertEqual(det.anchor_id, 1)
        self.assertEqual(det.spatial_coords, (0.5, 0.6))
    
    def test_defaults(self):
        """Test default values."""
        det = SpatialDetection(
            class_id=0,
            class_name="obj",
            confidence=0.5,
            bbox=(0, 0, 10, 10),
            bbox_normalized=(0.5, 0.5, 0.1, 0.1),
        )
        
        self.assertIsNone(det.anchor_id)
        self.assertIsNone(det.spatial_coords)
        self.assertTrue(det.is_new)


class TestSpatialInferenceResult(unittest.TestCase):
    """Tests for SpatialInferenceResult dataclass."""
    
    def test_creation(self):
        """Test creating a result."""
        result = SpatialInferenceResult(
            frame_id=1,
            timestamp=123.456,
            image_size=(640, 480),
            pose=DevicePose(timestamp=123.456),
            detections=[],
            inference_time_ms=15.0,
            slam_time_ms=5.0,
            total_time_ms=20.0,
            total_anchors=0,
        )
        
        self.assertEqual(result.frame_id, 1)
        self.assertEqual(result.image_size, (640, 480))
        self.assertEqual(result.inference_time_ms, 15.0)


class TestSpatialInferencePipelineMocked(unittest.TestCase):
    """Tests for SpatialInferencePipeline with mocked services."""
    
    def test_init_with_slam(self):
        """Test initialization with SLAM enabled."""
        with patch('spatial_inference.InferenceService'):
            pipeline = SpatialInferencePipeline(
                model_path="test.pt",
                enable_slam=True,
                imu_enabled=True,
            )
            
            self.assertTrue(pipeline.enable_slam)
            self.assertIsNotNone(pipeline.slam_service)
    
    def test_init_without_slam(self):
        """Test initialization with SLAM disabled."""
        with patch('spatial_inference.InferenceService'):
            pipeline = SpatialInferencePipeline(
                model_path="test.pt",
                enable_slam=False,
            )
            
            self.assertFalse(pipeline.enable_slam)
            self.assertIsNone(pipeline.slam_service)
    
    def test_reset(self):
        """Test pipeline reset."""
        with patch('spatial_inference.InferenceService'):
            pipeline = SpatialInferencePipeline(
                model_path="test.pt",
                enable_slam=True,
            )
            
            # Simulate some state
            pipeline._frame_count = 100
            
            pipeline.reset()
            
            self.assertEqual(pipeline._frame_count, 0)
    
    def test_get_spatial_map_with_slam(self):
        """Test getting spatial map with SLAM enabled."""
        with patch('spatial_inference.InferenceService'):
            pipeline = SpatialInferencePipeline(
                model_path="test.pt",
                enable_slam=True,
            )
            
            # Should return empty list initially
            spatial_map = pipeline.get_spatial_map()
            self.assertEqual(spatial_map, [])
    
    def test_get_spatial_map_without_slam(self):
        """Test getting spatial map with SLAM disabled."""
        with patch('spatial_inference.InferenceService'):
            pipeline = SpatialInferencePipeline(
                model_path="test.pt",
                enable_slam=False,
            )
            
            spatial_map = pipeline.get_spatial_map()
            self.assertEqual(spatial_map, [])


class TestSpatialInferencePipelineIntegration(unittest.TestCase):
    """Integration tests (require mocked inference service)."""
    
    @patch('spatial_inference.InferenceService')
    def test_process_frame_returns_result(self, MockInferenceService):
        """Test that process_frame returns a SpatialInferenceResult."""
        # Setup mock
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.detections = []
        mock_result.image_size = (640, 480)
        mock_service.infer_frame.return_value = mock_result
        MockInferenceService.return_value = mock_service
        
        pipeline = SpatialInferencePipeline(
            model_path="test.pt",
            enable_slam=True,
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        
        self.assertIsInstance(result, SpatialInferenceResult)
        self.assertEqual(result.frame_id, 1)
        self.assertEqual(result.image_size, (640, 480))
    
    @patch('spatial_inference.InferenceService')
    def test_process_frame_with_detections(self, MockInferenceService):
        """Test process_frame with mock detections."""
        # Setup mock detection
        mock_detection = MagicMock()
        mock_detection.class_id = 0
        mock_detection.class_name = "pothole"
        mock_detection.confidence = 0.9
        mock_detection.bbox = (100, 100, 200, 200)
        mock_detection.bbox_normalized = (0.5, 0.5, 0.1, 0.1)
        
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.detections = [mock_detection]
        mock_result.image_size = (640, 480)
        mock_service.infer_frame.return_value = mock_result
        MockInferenceService.return_value = mock_service
        
        pipeline = SpatialInferencePipeline(
            model_path="test.pt",
            enable_slam=True,
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].class_name, "pothole")
        self.assertIsNotNone(result.detections[0].anchor_id)
    
    @patch('spatial_inference.InferenceService')
    def test_frame_count_increments(self, MockInferenceService):
        """Test that frame count increments."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.detections = []
        mock_result.image_size = (640, 480)
        mock_service.infer_frame.return_value = mock_result
        MockInferenceService.return_value = mock_service
        
        pipeline = SpatialInferencePipeline(
            model_path="test.pt",
            enable_slam=False,
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result1 = pipeline.process_frame(frame)
        result2 = pipeline.process_frame(frame)
        result3 = pipeline.process_frame(frame)
        
        self.assertEqual(result1.frame_id, 1)
        self.assertEqual(result2.frame_id, 2)
        self.assertEqual(result3.frame_id, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
