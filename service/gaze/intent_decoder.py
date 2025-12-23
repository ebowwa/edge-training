"""
Gaze-Intent Decoder for egocentric vision.
Decodes user intent from gaze patterns to provide implicit rewards/priorities.

Based on: https://arxiv.org/abs/2505.02872
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GazePoint:
    """A single gaze fixation point."""
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    duration_ms: float  # Fixation duration
    timestamp: float  # Unix timestamp


@dataclass
class GazeSequence:
    """A sequence of gaze points with metadata."""
    points: List[GazePoint]
    
    @property
    def total_duration(self) -> float:
        return sum(p.duration_ms for p in self.points)
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Weighted centroid by duration."""
        if not self.points:
            return (0.5, 0.5)
        total_weight = self.total_duration
        if total_weight == 0:
            return (0.5, 0.5)
        x = sum(p.x * p.duration_ms for p in self.points) / total_weight
        y = sum(p.y * p.duration_ms for p in self.points) / total_weight
        return (x, y)
    
    def to_heatmap(self, size: int = 64, sigma: float = 0.1) -> torch.Tensor:
        """Convert gaze sequence to spatial heatmap."""
        heatmap = torch.zeros(1, size, size)
        
        for point in self.points:
            # Create Gaussian blob centered at gaze point
            cx, cy = int(point.x * size), int(point.y * size)
            weight = point.duration_ms / 1000.0  # Normalize to seconds
            
            for i in range(size):
                for j in range(size):
                    dist = ((i - cy) ** 2 + (j - cx) ** 2) / (2 * (sigma * size) ** 2)
                    heatmap[0, i, j] += weight * torch.exp(torch.tensor(-dist))
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap


class GazeEncoder(nn.Module):
    """
    Encodes gaze sequences into feature vectors.
    Can be used for intent prediction or attention weighting.
    """
    
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        # Point embedding: (x, y, duration, delta_t) -> embed_dim
        self.point_embed = nn.Linear(4, embed_dim)
        
        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim * 2, embed_dim)
    
    def forward(self, gaze_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gaze_points: (batch, seq_len, 4) - x, y, duration, delta_t
        
        Returns:
            Gaze embedding (batch, embed_dim)
        """
        # Embed points
        x = self.point_embed(gaze_points)  # (B, S, embed_dim)
        
        # Encode sequence
        output, (h_n, _) = self.lstm(x)
        
        # Use final hidden state
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, hidden*2)
        
        return self.out_proj(h_final)  # (B, embed_dim)


class GazeIntentDecoder(nn.Module):
    """
    Decodes user intent from gaze patterns.
    Outputs attention weights for spatial regions.
    """
    
    def __init__(self, embed_dim: int = 128, num_classes: int = 80):
        super().__init__()
        self.encoder = GazeEncoder(embed_dim=embed_dim)
        self.intent_head = nn.Linear(embed_dim, num_classes)
        self.attention_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, gaze_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gaze_points: (batch, seq_len, 4)
        
        Returns:
            intent_logits: (batch, num_classes) - what user is looking for
            attention_weight: (batch, 1) - how focused the gaze is
        """
        embed = self.encoder(gaze_points)
        intent = self.intent_head(embed)
        attention = torch.sigmoid(self.attention_head(embed))
        return intent, attention


class GazeRewardShaper:
    """
    Shapes rewards based on gaze patterns.
    Provides implicit feedback for detection/counting models.
    """
    
    def __init__(self, duration_threshold_ms: float = 500.0, decay_rate: float = 0.9):
        """
        Args:
            duration_threshold_ms: Minimum fixation to consider "attended"
            decay_rate: How quickly old fixations decay in importance
        """
        self.duration_threshold = duration_threshold_ms
        self.decay_rate = decay_rate
    
    def compute_reward(
        self, 
        detection_boxes: torch.Tensor,  # (N, 4) xywh
        gaze_sequence: GazeSequence,
    ) -> torch.Tensor:
        """
        Compute reward for each detection based on gaze overlap.
        
        Returns:
            rewards: (N,) reward for each detection
        """
        n_boxes = detection_boxes.shape[0]
        rewards = torch.zeros(n_boxes)
        
        for i, box in enumerate(detection_boxes):
            x, y, w, h = box.tolist()
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            
            # Check overlap with each gaze point
            for point in gaze_sequence.points:
                if point.duration_ms < self.duration_threshold:
                    continue
                
                # Point in box?
                if x1 <= point.x <= x2 and y1 <= point.y <= y2:
                    # Reward proportional to fixation duration
                    reward = min(1.0, point.duration_ms / 2000.0)
                    rewards[i] = max(rewards[i], reward)
        
        return rewards
    
    def get_priority_regions(
        self, 
        gaze_sequence: GazeSequence, 
        top_k: int = 3
    ) -> List[Tuple[float, float, float]]:
        """
        Get top-k regions of interest from gaze.
        
        Returns:
            List of (x, y, importance) tuples
        """
        # Filter by duration
        attended = [p for p in gaze_sequence.points if p.duration_ms >= self.duration_threshold]
        
        # Sort by duration
        attended.sort(key=lambda p: p.duration_ms, reverse=True)
        
        # Return top k
        return [
            (p.x, p.y, min(1.0, p.duration_ms / 2000.0))
            for p in attended[:top_k]
        ]
