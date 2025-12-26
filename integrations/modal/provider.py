"""Modal GPU Provider implementation."""

import logging
import time
import modal
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..base import (
    GPUProvider,
    JobState,
    JobStatus,
    JobResult,
    TrainingJobConfig,
    InferenceJobConfig
)
from .config import ModalConfig, ModalJobConfig
from .utils import with_retry, get_directory_size, estimate_upload_time

logger = logging.getLogger(__name__)


class ModalProvider(GPUProvider):
    """
    Modal GPU provider with automatic retries and chunked uploads.
    
    Usage:
        config = ModalConfig(gpu_type="A10G", secret_names=["hf-token"])
        provider = ModalProvider(config)
        
        # Submit job
        job_id = provider.submit_custom_job(
            app_name="my-app",
            function_name="train",
            args={"epochs": 60}
        )
        
        # Monitor
        result = provider.run_sync_custom(job_config)
    """
    
    def __init__(self, config: ModalConfig):
        super().__init__(config)
        self.config: ModalConfig = config
        self._apps: Dict[str, modal.App] = {}
        self._jobs: Dict[str, Dict[str, Any]] = {}
    
    @property
    def name(self) -> str:
        return "modal"
    
    def create_app(
        self,
        app_name: str,
        pip_packages: Optional[List[str]] = None,
        local_dirs: Optional[Dict[Path, str]] = None
    ) -> modal.App:
        """
        Create Modal app with image and local directory mounts.
        
        Args:
            app_name: Name for the Modal app
            pip_packages: Additional packages to install
            local_dirs: Dict of {local_path: remote_path} to mount
            
        Returns:
            Modal App instance
        """
        if app_name in self._apps:
            return self._apps[app_name]
        
        app = modal.App(app_name)
        
        # Build image
        packages = (pip_packages or []) + self.config.pip_packages
        image = modal.Image.debian_slim(python_version=self.config.python_version)
        
        if packages:
            image = image.pip_install(*packages)
        
        # Add local directories
        if local_dirs:
            for local_path, remote_path in local_dirs.items():
                if not local_path.exists():
                    logger.warning(f"Local path not found: {local_path}")
                    continue
                
                size = get_directory_size(local_path)
                est_time = estimate_upload_time(size)
                logger.info(
                    f"Mounting {local_path} -> {remote_path} "
                    f"({size / (1024**2):.1f}MB, ~{est_time:.0f}s)"
                )
                
                image = image.add_local_dir(local_path, remote_path=remote_path)
        
        # Store for reuse
        self._apps[app_name] = app
        
        return app
    
    @with_retry(max_retries=3, delay=2.0)
    def submit_custom_job(
        self,
        job_config: ModalJobConfig,
        local_dirs: Optional[Dict[Path, str]] = None
    ) -> str:
        """
        Submit a custom Modal job.
        
        Args:
            job_config: Modal job configuration
            local_dirs: Local directories to mount
            
        Returns:
            Job ID (Modal function call ID)
        """
        app = self.create_app(
            job_config.app_name,
            pip_packages=job_config.gpu_config.pip_packages,
            local_dirs=local_dirs
        )
        
        # Job submission would happen via modal.Function.spawn()
        # For now, store config for tracking
        job_id = f"{job_config.app_name}-{int(time.time())}"
        self._jobs[job_id] = {
            "config": job_config,
            "state": JobState.PENDING,
            "submitted_at": time.time()
        }
        
        logger.info(f"Submitted Modal job: {job_id}")
        return job_id
    
    def submit_training_job(self, config: TrainingJobConfig) -> str:
        """Submit training job (standard interface)."""
        job_config = ModalJobConfig(
            app_name="training",
            function_name="train",
            gpu_config=self.config,
            args={
                "model_name": config.model_name,
                "dataset_uri": config.dataset_uri,
                "epochs": config.epochs,
                "imgsz": config.imgsz,
                "batch_size": config.batch_size,
                "hyperparams": config.hyperparams
            }
        )
        return self.submit_custom_job(job_config)
    
    def submit_inference_job(
        self,
        config: InferenceJobConfig,
        inputs: List[Any]
    ) -> str:
        """Submit inference job (standard interface)."""
        job_config = ModalJobConfig(
            app_name="inference",
            function_name="infer",
            gpu_config=self.config,
            args={
                "model_name": config.model_name,
                "inputs": inputs,
                "conf_threshold": config.conf_threshold,
                "iou_threshold": config.iou_threshold
            }
        )
        return self.submit_custom_job(job_config)
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status."""
        if job_id not in self._jobs:
            raise ValueError(f"Unknown job: {job_id}")
        
        job = self._jobs[job_id]
        elapsed = time.time() - job["submitted_at"]
        
        return JobStatus(
            job_id=job_id,
            state=job["state"],
            progress=0.5 if job["state"] == JobState.RUNNING else 0.0,
            elapsed_seconds=elapsed
        )
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get job result."""
        if job_id not in self._jobs:
            raise ValueError(f"Unknown job: {job_id}")
        
        job = self._jobs[job_id]
        if job["state"] != JobState.COMPLETED:
            raise ValueError(f"Job {job_id} not complete (state: {job['state']})")
        
        return JobResult(
            job_id=job_id,
            success=True,
            output=job.get("output", {})
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel job."""
        if job_id not in self._jobs:
            return False
        
        self._jobs[job_id]["state"] = JobState.CANCELLED
        logger.info(f"Cancelled job: {job_id}")
        return True
