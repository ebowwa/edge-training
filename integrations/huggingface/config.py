"""HuggingFace configuration."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace Hub integration.
    
    Usage:
        config = HuggingFaceConfig(repo_id="ebowwa/usd-side-coco-annotations")
        config = HuggingFaceConfig(repo_id="my-org/my-dataset", private=True)
    """
    
    # Repository settings
    repo_id: str = "ebowwa/usd-side-coco-annotations"
    repo_type: str = "dataset"  # dataset, model, space
    
    # Authentication (reads from HF_TOKEN env var by default)
    token_env_var: str = "HF_TOKEN"
    
    # Download settings
    cache_dir: Optional[str] = None
    splits: List[str] = field(default_factory=lambda: ["train", "validation", "test"])
    
    # Upload settings
    private: bool = False
    commit_message: str = "Upload via edge-training"
    
    @property
    def hub_url(self) -> str:
        """Get HuggingFace Hub URL for this repo."""
        type_prefix = "datasets" if self.repo_type == "dataset" else self.repo_type
        return f"https://huggingface.co/{type_prefix}/{self.repo_id}"
