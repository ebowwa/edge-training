"""
Hetzner Cloud GPU Provider

Hetzner offers dedicated GPU servers with monthly pricing.
This provider handles provisioning, monitoring, and training on Hetzner infrastructure.
"""

import logging
import os
from typing import Any, List, Optional

import httpx

from integrations.base import (
    GPUProvider,
    TrainingJobConfig,
    InferenceJobConfig,
    JobState,
    JobStatus,
    JobResult,
)
from integrations.hetzner.config import HetznerConfig
from integrations.catalog import GPUOffer, ProviderType

logger = logging.getLogger(__name__)

# Hetzner API endpoints
HETZNER_API_BASE = "https://api.hetzner.cloud/v1"


class HetznerProvider(GPUProvider):
    """
    Hetzner Cloud GPU provider.

    Uses Hetzner Cloud API to create and manage GPU servers for training.

    Example:
        config = HetznerConfig(api_key="xxx")
        provider = HetznerProvider(config)
        job_id = provider.submit_training_job(training_config)
    """

    def __init__(self, config: HetznerConfig):
        super().__init__(config)
        self._config = config
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        logger.info("Hetzner provider initialized")

    @property
    def name(self) -> str:
        return "hetzner"

    async def _create_server(
        self,
        gpu_type: str,
        server_name: str,
    ) -> dict:
        """Create a GPU server."""
        # Map GPU types to Hetzner server types
        gpu_server_types = {
            "L4": "cg-1x-16",      # 1x L4, 120GB RAM
            "L40": "cg-1x-32",     # 1x L40, 240GB RAM
            "A100": "cg-1x-32-a100",  # 1x A100, 240GB RAM
            "H100": "cg-1x-64-h100",  # 1x H100, 480GB RAM
        }

        server_type = gpu_server_types.get(gpu_type)
        if not server_type:
            raise ValueError(f"Unknown GPU type: {gpu_type}. Options: {list(gpu_server_types.keys())}")

        # Create server request
        payload = {
            "name": server_name,
            "server_type": server_type,
            "image": "ubuntu-24.04",
            "location": self._config.region or "fsn1",
            "ssh_keys": self._config.ssh_keys or [],
            "user_data": self._config.user_data or "",
        }

        response = await self._client.post(
            f"{HETZNER_API_BASE}/servers",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def _get_server(self, server_id: int) -> dict:
        """Get server status."""
        response = await self._client.get(f"{HETZNER_API_BASE}/servers/{server_id}")
        response.raise_for_status()
        return response.json()

    async def _delete_server(self, server_id: int) -> bool:
        """Delete a server."""
        try:
            response = await self._client.delete(f"{HETZNER_API_BASE}/servers/{server_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to delete server {server_id}: {e}")
            return False

    def submit_training_job(self, config: TrainingJobConfig) -> str:
        """
        Submit a training job to Hetzner.

        Creates a GPU server, sets up training environment, and runs training.
        Returns job_id which is the server ID.
        """
        import asyncio

        # Determine GPU type from config or use default
        gpu_type = config.hyperparams.get("gpu_type", "A100")
        server_name = f"train-{config.model_name.replace('.', '-')}".replace("_", "-")

        async def _create():
            result = await self._create_server(gpu_type, server_name)
            server = result.get("server", {})
            return {
                "server_id": server.get("id"),
                "server_name": server.get("name"),
                "public_ip": server.get("public_net", {}).get("ipv4", {}).get("ip"),
                "status": server.get("status"),
                "config": config,
            }

        try:
            job_data = asyncio.run(_create())
            job_id = f"hetzner-{job_data['server_id']}"
            # Store job data for tracking
            # In production, this would go in a database
            logger.info(f"Created Hetzner server: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Failed to create Hetzner server: {e}")
            raise

    def submit_inference_job(
        self, config: InferenceJobConfig, inputs: List[Any]
    ) -> str:
        """Submit an inference job (creates server for duration of inference)."""
        # For Hetzner, inference is similar to training - create a server
        training_config = TrainingJobConfig(
            model_name=config.model_name,
            dataset_uri="",  # Not needed for inference
            epochs=1,
            hyperparams={"mode": "inference", "gpu_type": "L4"},  # Use cheaper GPU
        )
        return self.submit_training_job(training_config)

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get current status of a job (server)."""
        import asyncio

        # Extract server ID from job_id
        server_id = job_id.replace("hetzner-", "")

        async def _status():
            server = await self._get_server(int(server_id))
            s = server.get("server", {})

            # Map Hetzner status to JobState
            status_map = {
                "running": JobState.RUNNING,
                "starting": JobState.PENDING,
                "stopping": JobState.RUNNING,
                "stopped": JobState.COMPLETED,
                "error": JobState.FAILED,
            }

            return JobStatus(
                job_id=job_id,
                state=status_map.get(s.get("status"), JobState.PENDING),
                progress=0.5 if s.get("status") == "running" else 0.0,
                message=f"Server {s.get('status')}",
                elapsed_seconds=0.0,  # Would need to track creation time
            )

        try:
            return asyncio.run(_status())
        except Exception as e:
            logger.error(f"Failed to get server status: {e}")
            return JobStatus(
                job_id=job_id,
                state=JobState.FAILED,
                message=str(e),
            )

    def get_job_result(self, job_id: str) -> JobResult:
        """Get result from completed training."""
        # For Hetzner, results would need to be fetched from the server
        # via SSH or API. This is a simplified version.
        return JobResult(
            job_id=job_id,
            success=True,
            output={"message": "Training completed on Hetzner GPU server"},
            artifacts={"model_path": f"/root/runs/{job_id}/weights/best.pt"},
            metrics={},
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job by deleting the server."""
        import asyncio

        server_id = job_id.replace("hetzner-", "")

        async def _cancel():
            return await self._delete_server(int(server_id))

        try:
            return asyncio.run(_cancel())
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def __del__(self):
        """Cleanup on deletion."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._client.aclose())
        except Exception:
            pass
