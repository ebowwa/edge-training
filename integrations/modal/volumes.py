"""
Modal volume manager for large datasets.

Handles disk/memory efficiently by using Modal Volumes instead of image embedding.
"""

import modal
from pathlib import Path
from typing import Optional
import logging
import subprocess
import json

logger = logging.getLogger(__name__)


class ModalVolumeManager:
    """
    Manages Modal volumes for efficient data storage.
    
    Benefits:
    - Volumes are persistent across runs
    - Don't bloat image size
    - Can be shared across apps
    - Support incremental updates
    """
    
    def __init__(self, volume_name: str = "usd-dataset"):
        self.volume_name = volume_name
        self.volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    
    def upload_dataset(
        self,
        local_path: Path,
        remote_path: str = "/",
        force: bool = False,
        max_retries: int = 3
    ) -> dict:
        """
        Upload dataset to Modal volume with retry logic.
        
        Args:
            local_path: Local directory to upload
            remote_path: Remote path in volume
            force: Re-upload even if exists
            max_retries: Number of retry attempts
            
        Returns:
            Upload stats with success status
        """
        from .utils import get_directory_size, estimate_upload_time, with_retry
        
        if not local_path.exists():
            raise FileNotFoundError(f"Path not found: {local_path}")
        
        size = get_directory_size(local_path)
        est_time = estimate_upload_time(size)
        
        logger.info(f"Uploading {local_path} ({size / (1024**3):.1f}GB)")
        logger.info(f"Estimated time: {est_time / 60:.1f} minutes")
        
        # Build Modal CLI command
        cmd = [
            "modal", "volume", "put",
            self.volume_name,
            str(local_path),
            remote_path
        ]
        
        if force:
            cmd.append("--force")
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Wrap subprocess call with retry
        @with_retry(max_retries=max_retries, delay=5.0, exceptions=(subprocess.CalledProcessError,))
        def _do_upload():
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=est_time * 2  # 2x estimated time as timeout
            )
            return result
        
        try:
            result = _do_upload()
            logger.info("✓ Upload complete!")
            return {
                "success": True,
                "size_bytes": size,
                "estimated_time_seconds": est_time,
                "volume": self.volume_name,
                "remote_path": remote_path,
                "output": result.stdout
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Upload failed: {e.stderr}")
            return {
                "success": False,
                "error": e.stderr,
                "volume": self.volume_name
            }
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Upload timed out after {est_time * 2:.0f}s")
            return {
                "success": False,
                "error": "Upload timed out",
                "volume": self.volume_name
            }
    
    def get_mount_config(self, remote_path: str = "/data") -> dict:
        """
        Get volume mount configuration for @app.function.
        
        Returns:
            Dict for volumes parameter: {remote_path: volume}
        """
        return {remote_path: self.volume}
    
    def list_files(self, path: str = "/") -> list:
        """
        List files in volume.
        
        Args:
            path: Path within volume to list
            
        Returns:
            List of file/directory entries
        """
        cmd = ["modal", "volume", "ls", self.volume_name, path, "--json"]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            if result.stdout.strip():
                return json.loads(result.stdout)
            return []
        except subprocess.CalledProcessError as e:
            logger.error(f"List failed: {e.stderr}")
            return []
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON output, returning raw stdout")
            return result.stdout.strip().split('\n') if result.stdout else []
    
    def cleanup(self, confirm: bool = False) -> bool:
        """
        Delete volume.
        
        Args:
            confirm: Must be True to actually delete (safety check)
            
        Returns:
            True if deleted successfully
        """
        if not confirm:
            logger.warning(
                f"Cleanup called but not confirmed. "
                f"Pass confirm=True to delete {self.volume_name}"
            )
            return False
        
        cmd = ["modal", "volume", "delete", self.volume_name, "--yes"]
        
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            logger.info(f"✓ Deleted volume: {self.volume_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Delete failed: {e.stderr}")
            return False


def create_memory_optimized_image(
    python_version: str = "3.11",
    pip_packages: Optional[list] = None
) -> modal.Image:
    """
    Create image optimized for memory usage.
    
    - No dataset embedded (use volumes instead)
    - Only essential packages
    - Smaller base image
    """
    image = modal.Image.debian_slim(python_version=python_version)
    
    if pip_packages:
        # Install in chunks to reduce layer size
        for package in pip_packages:
            image = image.pip_install(package)
    
    return image


def get_gpu_memory_config(gpu_type: str) -> dict:
    """
    Get memory specs for GPU types.
    
    Returns:
        Dict with memory_gb, recommended_batch_size
    """
    configs = {
        "T4": {"memory_gb": 16, "batch_size": 32},
        "L4": {"memory_gb": 24, "batch_size": 48},
        "A10G": {"memory_gb": 24, "batch_size": 64},
        "A100": {"memory_gb": 40, "batch_size": 128},
    }
    return configs.get(gpu_type, {"memory_gb": 16, "batch_size": 32})
