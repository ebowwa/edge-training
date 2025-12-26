"""
RunPod GPU Provider

Wraps RunPod serverless API for training and inference jobs.
"""

import base64
import logging
from typing import Any, List, Optional

import runpod

from integrations.base import (
    GPUProvider,
    TrainingJobConfig,
    InferenceJobConfig,
    JobState,
    JobStatus,
    JobResult,
)
from integrations.runpod.config import RunPodConfig

logger = logging.getLogger(__name__)

RUNPOD_STATUS_MAP = {
    "IN_QUEUE": JobState.PENDING,
    "IN_PROGRESS": JobState.RUNNING,
    "COMPLETED": JobState.COMPLETED,
    "FAILED": JobState.FAILED,
    "CANCELLED": JobState.CANCELLED,
    "TIMED_OUT": JobState.TIMEOUT,
}


class RunPodProvider(GPUProvider):
    """RunPod GPU provider for serverless training/inference."""

    def __init__(self, config: RunPodConfig):
        super().__init__(config)
        self._config = config
        runpod.api_key = config.api_key
        logger.info("RunPod initialized")

    @property
    def name(self) -> str:
        return "runpod"

    def submit_training_job(self, config: TrainingJobConfig) -> str:
        if not self._config.training_endpoint_id:
            raise ValueError("training_endpoint_id required")

        endpoint = runpod.Endpoint(self._config.training_endpoint_id)
        run = endpoint.run({
            "input": {
                "model_name": config.model_name,
                "dataset_uri": config.dataset_uri,
                "epochs": config.epochs,
                "imgsz": config.imgsz,
                "batch_size": config.batch_size,
                **config.hyperparams,
            }
        })
        logger.info(f"Training job: {run.job_id}")
        return run.job_id

    def submit_inference_job(self, config: InferenceJobConfig, inputs: List[Any]) -> str:
        endpoint_id = config.endpoint_id or self._config.inference_endpoint_id
        if not endpoint_id:
            raise ValueError("inference_endpoint_id required")

        encoded = [
            base64.b64encode(i).decode() if isinstance(i, bytes) else i
            for i in inputs
        ]

        endpoint = runpod.Endpoint(endpoint_id)
        run = endpoint.run({
            "input": {
                "model_name": config.model_name,
                "images": encoded,
                "conf_threshold": config.conf_threshold,
                "iou_threshold": config.iou_threshold,
            }
        })
        logger.info(f"Inference job: {run.job_id}")
        return run.job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        status = runpod.status(job_id)
        return JobStatus(
            job_id=job_id,
            state=RUNPOD_STATUS_MAP.get(status.get("status"), JobState.PENDING),
            progress=status.get("progress", 0.0),
            message=status.get("message", ""),
            elapsed_seconds=status.get("executionTime", 0.0),
        )

    def get_job_result(self, job_id: str) -> JobResult:
        result = runpod.output(job_id)
        if "error" in result:
            return JobResult(job_id=job_id, success=False, error=result["error"])
        return JobResult(
            job_id=job_id,
            success=True,
            output=result.get("output", {}),
            artifacts=result.get("artifacts", {}),
            metrics=result.get("metrics", {}),
        )

    def cancel_job(self, job_id: str) -> bool:
        try:
            runpod.cancel(job_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def run_inference_sync(
        self, config: InferenceJobConfig, inputs: List[Any], timeout: Optional[float] = None
    ) -> JobResult:
        endpoint_id = config.endpoint_id or self._config.inference_endpoint_id
        if not endpoint_id:
            raise ValueError("inference_endpoint_id required")

        encoded = [
            base64.b64encode(i).decode() if isinstance(i, bytes) else i
            for i in inputs
        ]

        endpoint = runpod.Endpoint(endpoint_id)
        result = endpoint.run_sync(
            {"input": {
                "model_name": config.model_name,
                "images": encoded,
                "conf_threshold": config.conf_threshold,
                "iou_threshold": config.iou_threshold,
            }},
            timeout=timeout or self._config.timeout_seconds,
        )

        if "error" in result:
            return JobResult(job_id=result.get("id", ""), success=False, error=result["error"])
        return JobResult(
            job_id=result.get("id", ""),
            success=True,
            output=result.get("output", result),
        )
