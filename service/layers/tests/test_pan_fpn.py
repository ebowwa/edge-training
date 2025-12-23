"""
Tests for PAN-FPN Implementation.
"""

import sys
import unittest
import torch
from pathlib import Path

# Add project root and service/layers to path
_layers_dir = Path(__file__).parent.parent
_service_dir = _layers_dir.parent
_root_dir = _service_dir.parent

sys.path.insert(0, str(_root_dir))
sys.path.insert(0, str(_service_dir))
sys.path.insert(0, str(_layers_dir))

# Mock the package structure for relative imports if needed
# But here we can just import from the direct files since we added to path
from common import Conv, C2f, SPPF
from neck import PANFPN
from head import Detect


class TestPANFPN(unittest.TestCase):
    """Tests for PAN-FPN logic and tensor shapes."""
    
    def setUp(self):
        # Typical channels for a small/medium YOLO backbone (P3, P4, P5)
        self.c3, self.c4, self.c5 = 128, 256, 512
        self.model = PANFPN(self.c3, self.c4, self.c5)
        self.batch_size = 2

    def test_forward_shapes(self):
        """Verify that output shapes match expected strides."""
        # Input features at strides 8, 16, 32 (for 640x640 input)
        p3 = torch.randn(self.batch_size, self.c3, 80, 80)
        p4 = torch.randn(self.batch_size, self.c4, 40, 40)
        p5 = torch.randn(self.batch_size, self.c5, 20, 20)
        
        outputs = self.model(p3, p4, p5)
        
        self.assertEqual(len(outputs), 3, "Should return 3 scale outputs")
        
        n3, n4, n5 = outputs
        
        # Verify spatial dimensions
        self.assertEqual(n3.shape[2:], (80, 80))
        self.assertEqual(n4.shape[2:], (40, 40))
        self.assertEqual(n5.shape[2:], (20, 20))
        
        # Verify channel dimensions (should be same as input for this implementation)
        self.assertEqual(n3.shape[1], self.c3)
        self.assertEqual(n4.shape[1], self.c4)
        self.assertEqual(n5.shape[1], self.c5)

    def test_bottleneck_depth(self):
        """Test with different number of bottleneck layers."""
        model_deep = PANFPN(self.c3, self.c4, self.c5, n=6)
        p3 = torch.randn(1, self.c3, 40, 40)
        p4 = torch.randn(1, self.c4, 20, 20)
        p5 = torch.randn(1, self.c5, 10, 10)
        
        outputs = model_deep(p3, p4, p5)
        self.assertEqual(outputs[0].shape[2], 40)

    def test_parameter_count(self):
        """Verify the model has parameters (not just Identity)."""
        params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(params, 0, "Model should have trainable parameters")


class TestSPPF(unittest.TestCase):
    """Tests for SPPF logic and tensor shapes."""
    
    def test_sppf_shapes(self):
        """Verify SPPF maintains spatial resolution and correct output channels."""
        c1, c2 = 512, 512
        model = SPPF(c1, c2)
        x = torch.randn(1, c1, 20, 20)
        
        y = model(x)
        
        self.assertEqual(y.shape[2:], (20, 20), "SPPF should maintain spatial resolution")
        self.assertEqual(y.shape[1], c2, "Output channels should match c2")

    def test_sppf_stride(self):
        """SPPF should work on any spatial dimension."""
        model = SPPF(128, 128)
        x = torch.randn(1, 128, 10, 10)
        y = model(x)
        self.assertEqual(y.shape[2], 10)


class TestDetect(unittest.TestCase):
    """Tests for Anchor-free Detection Head."""
    
    def test_detect_output_shape(self):
        """Verify the head outputs the correct number of channels."""
        nc = 80
        ch = (128, 256, 512)
        head = Detect(nc, ch)
        head.eval()
        
        # Mock features from neck
        x = [
            torch.randn(1, 128, 80, 80),
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 512, 20, 20)
        ]
        
        output = head(x)
        
        # no = nc + reg_max * 4 = 80 + 16 * 4 = 144
        # n_anchors = 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
        expected_no = 80 + 16 * 4
        expected_anchors = 80*80 + 40*40 + 20*20
        
        self.assertEqual(output.shape, (1, expected_no, expected_anchors))


class TestEndToEnd(unittest.TestCase):
    """Integration test from neck to head."""
    
    def test_neck_to_head(self):
        """Pass backbone features through neck and then head."""
        c3, c4, c5 = 128, 256, 512
        neck = PANFPN(c3, c4, c5)
        head = Detect(80, (c3, c4, c5))
        head.eval()
        
        p3 = torch.randn(1, c3, 40, 40)
        p4 = torch.randn(1, c4, 20, 20)
        p5 = torch.randn(1, c5, 10, 10)
        
        features = neck(p3, p4, p5)
        output = head(features)
        
        self.assertEqual(output.shape[1], 80 + 16 * 4)
        self.assertEqual(output.shape[2], 40*40 + 20*20 + 10*10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
