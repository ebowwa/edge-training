"""
GPU Catalog - Real-time pricing and availability from cloud providers.

Supports:
- RunPod: Serverless GPUs and dedicated instances
- Hetzner: Cloud GPU servers
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Cloud provider types."""
    RUNPOD = "runpod"
    HETZNER = "hetzner"
    MODAL = "modal"


@dataclass
class GPUOffer:
    """A GPU offering from a cloud provider."""
    provider: ProviderType
    gpu_name: str
    vram_gb: float
    price_per_hour: float  # USD
    price_per_month: Optional[float] = None  # For dedicated servers
    available: bool = True
    region: str = "us-east"
    min_memory_gb: Optional[float] = None  # System RAM
    cpu_cores: Optional[int] = None
    specs: Dict = field(default_factory=dict)

    @property
    def hourly_only(self) -> bool:
        """True if only hourly pricing available (serverless)."""
        return self.price_per_month is None


# RunPod GPU catalog (from their API)
# Source: https://docs.runpod.io/graphql-api
RUNPOD_GPUS = [
    # Community Cloud (cheaper)
    GPUOffer(ProviderType.RUNPOD, "NVIDIA GeForce RTX 3090", 24, 0.22, region="us-east"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA GeForce RTX 3090", 24, 0.44, region="us-west"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA GeForce RTX 4090", 24, 0.59, region="us-east"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA A100 80GB", 80, 1.39, region="us-east"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA A100 80GB", 80, 1.59, region="eu-central"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA A6000", 48, 0.78, region="us-east"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA RTX 6000 Ada", 48, 0.79, region="us-east"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA RTX A5000", 24, 0.45, region="us-east"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA RTX 4000 Ada", 20, 0.35, region="us-east"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA RTX 4000 Ada SFF", 20, 0.29, region="us-east"),
    # Secure Cloud (more expensive)
    GPUOffer(ProviderType.RUNPOD, "NVIDIA A100 80GB", 80, 2.69, region="us-east-secure"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA A100-SXM4-80GB", 80, 3.12, region="us-east-secure"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA H100 SXM", 80, 3.59, region="us-east-secure"),
    GPUOffer(ProviderType.RUNPOD, "NVIDIA H100 PCIe", 80, 3.09, region="us-east-secure"),
]

# Hetzner GPU servers (dedicated, monthly pricing converted to hourly)
# Source: https://www.hetzner.com/cloud-gpu
# Monthly / 730 hours = hourly
HETZNER_GPUS = [
    GPUOffer(
        ProviderType.HETZNER,
        "NVIDIA L4",
        24,
        0.21,  # ~152 EUR/month / 730
        189.04,  # Monthly EUR
        region="fsn1",
        min_memory_gb=120,
        cpu_cores=16,
        specs={"type": "GPU server", "network": "dedicated"}
    ),
    GPUOffer(
        ProviderType.HETZNER,
        "NVIDIA L40",
        48,
        0.41,  # ~300 EUR/month
        371.23,
        region="fsn1",
        min_memory_gb=240,
        cpu_cores=32,
        specs={"type": "GPU server", "network": "dedicated"}
    ),
    GPUOffer(
        ProviderType.HETZNER,
        "NVIDIA A100",
        80,
        0.89,  # ~650 EUR/month
        804.11,
        region="fsn1",
        min_memory_gb=240,
        cpu_cores=32,
        specs={"type": "GPU server", "network": "dedicated"}
    ),
    GPUOffer(
        ProviderType.HETZNER,
        "NVIDIA H100",
        80,
        1.71,  # ~1250 EUR/month
        1545.21,
        region="fsn1",
        min_memory_gb=480,
        cpu_cores=64,
        specs={"type": "GPU server", "network": "dedicated"}
    ),
]


class GPUCatalog:
    """
    Query GPU pricing and availability from cloud providers.

    Usage:
        catalog = GPUCatalog()

        # Get all GPUs
        all_gpus = catalog.list_gpus()

        # Find cheapest GPU with at least 24GB VRAM
        gpus = catalog.find_gpus(min_vram_gb=24, sort_by="price")

        # Get GPUs from specific provider
        runpod_gpus = catalog.list_gpus(provider=ProviderType.RUNPOD)

        # Estimate training cost
        cost = catalog.estimate_cost(
            provider=ProviderType.RUNPOD,
            gpu_name="NVIDIA A100 80GB",
            training_hours=10
        )
    """

    def __init__(self, api_keys: Optional[Dict[ProviderType, str]] = None):
        """
        Initialize GPU catalog.

        Args:
            api_keys: Optional dict of provider API keys for live data
        """
        self.api_keys = api_keys or {}
        self._cache: List[GPUOffer] = []

    def list_gpus(
        self,
        provider: Optional[ProviderType] = None,
        min_vram_gb: float = 0,
        max_price_per_hour: Optional[float] = None,
        region: Optional[str] = None,
    ) -> List[GPUOffer]:
        """
        List available GPUs with optional filters.

        Args:
            provider: Filter by provider
            min_vram_gb: Minimum VRAM in GB
            max_price_per_hour: Maximum hourly price (USD)
            region: Filter by region

        Returns:
            List of GPU offers matching criteria
        """
        gpus = []

        # Add RunPod GPUs
        if provider is None or provider == ProviderType.RUNPOD:
            for gpu in RUNPOD_GPUS:
                if gpu.vram_gb >= min_vram_gb:
                    if max_price_per_hour is None or gpu.price_per_hour <= max_price_per_hour:
                        if region is None or region in gpu.region or gpu.region == "us-east":
                            gpus.append(gpu)

        # Add Hetzner GPUs
        if provider is None or provider == ProviderType.HETZNER:
            for gpu in HETZNER_GPUS:
                if gpu.vram_gb >= min_vram_gb:
                    if max_price_per_hour is None or gpu.price_per_hour <= max_price_per_hour:
                        if region is None or region in gpu.region or gpu.region == "fsn1":
                            gpus.append(gpu)

        return gpus

    def find_cheapest(
        self,
        min_vram_gb: float = 0,
        provider: Optional[ProviderType] = None,
        region: Optional[str] = None,
    ) -> Optional[GPUOffer]:
        """Find cheapest GPU meeting VRAM requirement."""
        gpus = self.list_gpus(provider=provider, min_vram_gb=min_vram_gb, region=region)
        if not gpus:
            return None
        return min(gpus, key=lambda g: g.price_per_hour)

    def find_best_value(
        self,
        min_vram_gb: float = 0,
        provider: Optional[ProviderType] = None,
    ) -> Optional[GPUOffer]:
        """Find best value GPU (price per GB VRAM)."""
        gpus = self.list_gpus(provider=provider, min_vram_gb=min_vram_gb)
        if not gpus:
            return None
        return min(gpus, key=lambda g: g.price_per_hour / g.vram_gb)

    def estimate_cost(
        self,
        provider: ProviderType,
        gpu_name: str,
        training_hours: float,
    ) -> Optional[float]:
        """
        Estimate training cost for a specific GPU.

        Returns:
            Cost in USD, or None if GPU not found
        """
        gpus = self.list_gpus(provider=provider)
        for gpu in gpus:
            if gpu_name in gpu.gpu_name or gpu.gpu_name in gpu_name:
                return gpu.price_per_hour * training_hours
        return None

    def get_gpu_by_name(self, name: str) -> Optional[GPUOffer]:
        """Get GPU by partial name match."""
        all_gpus = self.list_gpus()
        for gpu in all_gpus:
            if name.lower() in gpu.gpu_name.lower():
                return gpu
        return None

    def compare_providers(
        self,
        min_vram_gb: float,
        training_hours: float = 1,
    ) -> Dict[ProviderType, List[GPUOffer]]:
        """
        Compare offerings across providers.

        Returns:
            Dict mapping provider to list of qualifying GPUs with cost estimates
        """
        result = {}
        for provider in ProviderType:
            if provider == ProviderType.MODAL:
                continue  # Not implemented yet
            gpus = self.list_gpus(provider=provider, min_vram_gb=min_vram_gb)
            if gpus:
                result[provider] = sorted(gpus, key=lambda g: g.price_per_hour)
        return result

    def format_offer(self, gpu: GPUOffer, training_hours: float = 1) -> str:
        """Format GPU offer as readable string."""
        cost = gpu.price_per_hour * training_hours
        monthly = f" (${gpu.price_per_month:.2f}/mo)" if gpu.price_per_month else ""
        return (
            f"{gpu.gpu_name:30} | {gpu.vram_gb:5}GB VRAM | "
            f"${gpu.price_per_hour:.2f}/h{monthly} | "
            f"${cost:.2f} for {training_hours}h | {gpu.provider.value}"
        )

    def print_comparison(
        self,
        min_vram_gb: float,
        training_hours: float = 1,
    ):
        """Print comparison table of GPUs."""
        print(f"\n{'=' * 100}")
        print(f"GPU COMPARISON (min {min_vram_gb}GB VRAM, {training_hours}h training)")
        print(f"{'=' * 100}")

        for provider, gpus in self.compare_providers(min_vram_gb, training_hours).items():
            print(f"\n{provider.value.upper()}:")
            print("-" * 100)
            for gpu in gpus[:5]:  # Top 5
                print(self.format_offer(gpu, training_hours))

        print(f"{'=' * 100}\n")

    async def refresh_from_api(self, provider: ProviderType) -> bool:
        """
        Fetch latest GPU catalog from provider API (experimental).

        Returns:
            True if successful
        """
        # TODO: Implement live API fetching
        # RunPod GraphQL API, Hetzner REST API
        logger.warning(f"Live API refresh not yet implemented for {provider}")
        return False


def quick_search(
    min_vram_gb: float = 0,
    max_price: Optional[float] = None,
    provider: Optional[str] = None,
) -> List[str]:
    """
    Quick CLI-friendly GPU search.

    Returns:
        List of formatted GPU strings
    """
    catalog = GPUCatalog()
    provider_type = ProviderType(provider) if provider else None

    gpus = catalog.list_gpus(
        provider=provider_type,
        min_vram_gb=min_vram_gb,
        max_price_per_hour=max_price,
    )

    results = []
    for gpu in sorted(gpus, key=lambda g: g.price_per_hour)[:10]:
        results.append(catalog.format_offer(gpu))

    return results
