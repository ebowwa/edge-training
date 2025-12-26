"""Dataset configuration and USD class definitions."""

from dataclasses import dataclass, field
from typing import Dict, List


# 24 classes: 12 regular + 12 counterfeit (front/back for each denomination)
USD_CLASSES: Dict[int, str] = {
    0: "100USD-Back",
    1: "100USD-Front",
    2: "10USD-Back",
    3: "10USD-Front",
    4: "1USD-Back",
    5: "1USD-Front",
    6: "20USD-Back",
    7: "20USD-Front",
    8: "50USD-Back",
    9: "50USD-Front",
    10: "5USD-Back",
    11: "5USD-Front",
    12: "Counterfeit 100 USD Back",
    13: "Counterfeit 100 USD Front",
    14: "Counterfeit 10USD Back",
    15: "Counterfeit 10USD Front",
    16: "Counterfeit 1USD Back",
    17: "Counterfeit 1USD Front",
    18: "Counterfeit 20USD Back",
    19: "Counterfeit 20USD Front",
    20: "Counterfeit 50USD Back",
    21: "Counterfeit 50USD Front",
    22: "Counterfeit 5USD Back",
    23: "Counterfeit 5USD Front",
}


@dataclass
class DatasetConfig:
    """Configuration for USD detection dataset.
    
    Usage:
        config = DatasetConfig()  # defaults to HuggingFace dataset
        config = DatasetConfig(repo_id="my-org/my-dataset")
    """
    
    # HuggingFace settings
    repo_id: str = "ebowwa/usd-side-coco-annotations"
    
    # Modal volume
    volume_name: str = "usd-dataset-hf"
    
    # Classes
    classes: Dict[int, str] = field(default_factory=lambda: USD_CLASSES.copy())
    
    # HuggingFace split names → YOLO directory names
    splits: List[tuple] = field(default_factory=lambda: [
        ("train", "train"),
        ("validation", "valid"),
        ("test", "test"),
    ])
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)
    
    def generate_yaml(self, data_path: str = "/data") -> str:
        """Generate data.yaml content for YOLO training."""
        yaml_lines = [
            f"# USD Detection Dataset - {self.num_classes} classes",
            f"# Source: https://huggingface.co/datasets/{self.repo_id}",
            "",
            f"path: {data_path}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            f"nc: {self.num_classes}",
            "names:",
        ]
        for class_id, class_name in self.classes.items():
            yaml_lines.append(f"  {class_id}: {class_name}")
        
        return "\n".join(yaml_lines)
