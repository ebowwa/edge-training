"""
GPU Provider Integration - Base Classes

Abstract interface for cloud GPU providers (RunPod, Modal, etc.).
All providers implement the same interface for job submission, status, and results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobState(Enum):
    """Unified job state across all providers."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class GPUProviderConfig:
    """Base configuration for GPU providers."""
    api_key: str = ""
    timeout_seconds: int = 300
    poll_interval_seconds: float = 2.0


@dataclass
class TrainingJobConfig:
    """Configuration for training jobs."""
    model_name: str  # e.g., "yolov8m.pt", "rfdetr-base"
    dataset_uri: str  # Path/URL to dataset
    epochs: int = 60
    imgsz: int = 512
    batch_size: int = 32
    hyperparams: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceJobConfig:
    """Configuration for inference jobs."""
    model_name: str
    endpoint_id: Optional[str] = None  # Provider-specific endpoint
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45


@dataclass
class JobStatus:
    """Status of a submitted job."""
    job_id: str
    state: JobState
    progress: float = 0.0  # 0.0 - 1.0
    message: str = ""
    elapsed_seconds: float = 0.0
    logs: List[str] = field(default_factory=list)


@dataclass
class JobResult:
    """Result from a completed job."""
    job_id: str
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path/url
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class GPUProvider(ABC):
    """
    Abstract base class for GPU cloud providers.
    
    Implement this interface to add support for new providers (RunPod, Modal, etc.).
    
    Usage:
        provider = RunPodProvider(config)
        job_id = provider.submit_training_job(training_config)
        
        while True:
            status = provider.get_job_status(job_id)
            if status.state in (JobState.COMPLETED, JobState.FAILED):
                break
        
        result = provider.get_job_result(job_id)
    """
    
    def __init__(self, config: GPUProviderConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'runpod', 'modal')."""
        pass
    
    @abstractmethod
    def submit_training_job(self, config: TrainingJobConfig) -> str:
        """
        Submit a training job to the provider.
        
        Args:
            config: Training job configuration
            
        Returns:
            Job ID for tracking
        """
        pass
    
    @abstractmethod
    def submit_inference_job(
        self, 
        config: InferenceJobConfig, 
        inputs: List[Any]
    ) -> str:
        """
        Submit an inference job.
        
        Args:
            config: Inference configuration
            inputs: List of inputs (images as bytes, paths, or base64)
            
        Returns:
            Job ID for tracking
        """
        pass
    
    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get current status of a job."""
        pass
    
    @abstractmethod
    def get_job_result(self, job_id: str) -> JobResult:
        """
        Get result of a completed job.
        
        Raises:
            ValueError: If job is not complete
        """
        pass
    
    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Returns:
            True if cancelled successfully
        """
        pass
    
    def run_sync(
        self, 
        config: TrainingJobConfig | InferenceJobConfig,
        inputs: Optional[List[Any]] = None
    ) -> JobResult:
        """
        Synchronously run a job and wait for completion.
        
        Blocks until job completes or times out.
        """
        import time
        
        if isinstance(config, TrainingJobConfig):
            job_id = self.submit_training_job(config)
        else:
            job_id = self.submit_inference_job(config, inputs or [])
        
        start = time.time()
        while True:
            status = self.get_job_status(job_id)
            
            if status.state == JobState.COMPLETED:
                return self.get_job_result(job_id)
            
            if status.state == JobState.FAILED:
                result = self.get_job_result(job_id)
                raise RuntimeError(f"Job failed: {result.error}")
            
            if status.state == JobState.CANCELLED:
                raise RuntimeError("Job was cancelled")
            
            if time.time() - start > self.config.timeout_seconds:
                self.cancel_job(job_id)
                raise TimeoutError(f"Job {job_id} timed out")
            
            time.sleep(self.config.poll_interval_seconds)
