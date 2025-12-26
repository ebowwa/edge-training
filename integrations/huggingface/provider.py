"""HuggingFace provider for dataset operations."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
from dataclasses import dataclass

from .config import HuggingFaceConfig


class HuggingFaceProvider:
    """Provider for interacting with HuggingFace Hub.
    
    Usage:
        provider = HuggingFaceProvider()
        
        # Download dataset
        dataset = provider.load_dataset("train")
        
        # Upload files
        provider.upload_file(Path("model.pt"), "models/best.pt")
    """
    
    def __init__(self, config: Optional[HuggingFaceConfig] = None):
        self.config = config or HuggingFaceConfig()
        self._token: Optional[str] = None
    
    @property
    def token(self) -> Optional[str]:
        """Get HuggingFace token from environment."""
        if self._token is None:
            self._token = os.environ.get(self.config.token_env_var)
        return self._token
    
    def load_dataset(self, split: str = "train") -> Any:
        """Load a dataset split from HuggingFace Hub.
        
        Args:
            split: Dataset split to load (train, validation, test)
            
        Returns:
            HuggingFace Dataset object
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        return load_dataset(
            self.config.repo_id,
            split=split,
            cache_dir=self.config.cache_dir,
            token=self.token,
        )
    
    def iterate_dataset(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        """Iterate over dataset items.
        
        Args:
            split: Dataset split to iterate
            
        Yields:
            Dataset items as dictionaries
        """
        dataset = self.load_dataset(split)
        for item in dataset:
            yield item
    
    def upload_file(
        self, 
        local_path: Path, 
        remote_path: str,
        commit_message: Optional[str] = None,
    ) -> str:
        """Upload a file to HuggingFace Hub.
        
        Args:
            local_path: Local file path
            remote_path: Path in the repository
            commit_message: Optional commit message
            
        Returns:
            URL of uploaded file
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")
        
        api = HfApi(token=self.token)
        
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            commit_message=commit_message or self.config.commit_message,
        )
        
        return f"{self.config.hub_url}/blob/main/{remote_path}"
    
    def upload_folder(
        self,
        local_folder: Path,
        remote_folder: str = "",
        commit_message: Optional[str] = None,
    ) -> str:
        """Upload a folder to HuggingFace Hub.
        
        Args:
            local_folder: Local folder path
            remote_folder: Path prefix in the repository
            commit_message: Optional commit message
            
        Returns:
            Repository URL
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")
        
        api = HfApi(token=self.token)
        
        api.upload_folder(
            folder_path=str(local_folder),
            path_in_repo=remote_folder,
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            commit_message=commit_message or self.config.commit_message,
        )
        
        return self.config.hub_url
    
    def create_repo(self, private: Optional[bool] = None) -> str:
        """Create a new repository on HuggingFace Hub.
        
        Args:
            private: Whether repo should be private (uses config default if None)
            
        Returns:
            Repository URL
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")
        
        api = HfApi(token=self.token)
        
        api.create_repo(
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            private=private if private is not None else self.config.private,
            exist_ok=True,
        )
        
        return self.config.hub_url
