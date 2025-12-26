"""
Storage and Compute Planning Utilities for ML Training.
Estimates disk, memory, and GPU requirements before training.
"""

from dataclasses import dataclass, field
from typing import Optional

# Import GPU catalog for live pricing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from integrations.catalog import GPUCatalog, ProviderType


@dataclass
class DatasetSpec:
    """Dataset specifications for storage planning."""
    num_images: int
    image_resolution: tuple  # (width, height)
    has_labels: bool = True
    label_format: str = "txt"  # txt, yaml, json
    augment_factor: int = 1  # How many augmented versions per image


@dataclass
class ModelSpec:
    """Model specifications for storage and compute planning."""
    family: str  # yolo, rfdetr
    variant: str  # n, s, m, l, x for YOLO
    params: Optional[int] = None  # Override param count


@dataclass
class TrainingSpec:
    """Training specifications for compute planning."""
    epochs: int
    batch_size: int
    save_period: int = 10  # Save checkpoint every N epochs


@dataclass
class StorageEstimate:
    """Storage requirement estimates."""
    dataset_mb: float
    dataset_augmented_mb: float
    model_weights_mb: float
    checkpoints_mb: float
    cache_mb: float
    total_mb: float

    def to_dict(self) -> dict:
        """Convert to dictionary with human-readable sizes."""
        def fmt(mb: float) -> str:
            if mb < 1024:
                return f"{mb:.1f} MB"
            return f"{mb / 1024:.2f} GB"

        return {
            "dataset": fmt(self.dataset_mb),
            "dataset_augmented": fmt(self.dataset_augmented_mb),
            "model_weights": fmt(self.model_weights_mb),
            "checkpoints": fmt(self.checkpoints_mb),
            "cache": fmt(self.cache_mb),
            "total": fmt(self.total_mb),
            "total_mb": self.total_mb,
        }


@dataclass
class ComputeEstimate:
    """Compute requirement estimates."""
    vram_mb: float
    recommended_gpu: str
    training_hours: float
    estimated_cost_usd: Optional[float] = None  # If using cloud GPU
    cpu_hours: Optional[float] = None  # CPU-only training time
    recommended_cpu: Optional[str] = None  # CPU recommendation
    ram_gb: Optional[float] = None  # System RAM needed

    def to_dict(self) -> dict:
        """Convert to dictionary with human-readable values."""
        result = {
            "vram": f"{self.vram_mb / 1024:.1f} GB" if self.vram_mb >= 1024 else f"{self.vram_mb:.0f} MB",
            "vram_mb": self.vram_mb,
            "recommended_gpu": self.recommended_gpu,
            "training_hours": f"{self.training_hours:.1f}h",
            "estimated_cost": f"${self.estimated_cost_usd:.2f}" if self.estimated_cost_usd else None,
        }
        if self.cpu_hours:
            result["cpu_hours"] = f"{self.cpu_hours:.1f}h"
        if self.recommended_cpu:
            result["recommended_cpu"] = self.recommended_cpu
        if self.ram_gb:
            result["ram"] = f"{self.ram_gb:.0f} GB"
        return result


# Model parameter counts (approximate)
YOLO_PARAMS = {
    "n": 3.2,   # ~3.2M params
    "s": 11.2,  # ~11.2M params
    "m": 25.9,  # ~25.9M params
    "l": 43.7,  # ~43.7M params
    "x": 68.2,  # ~68.2M params
}

# Model disk sizes (MB) - includes weights + optimizer state overhead
MODEL_SIZE_MB = {
    "n": 12,
    "s": 40,
    "m": 90,
    "l": 150,
    "x": 230,
}


class StoragePlanner:
    """Estimate storage requirements for training."""

    @staticmethod
    def estimate_image_size_mb(resolution: tuple) -> float:
        """Estimate single image size on disk (compressed JPEG)."""
        w, h = resolution
        # Assume JPEG compression ~0.5 bytes per pixel (typical for ML datasets)
        return (w * h * 0.5) / (1024 * 1024)

    @staticmethod
    def estimate_label_size_kb(label_format: str) -> float:
        """Estimate label file size."""
        sizes = {"txt": 0.5, "yaml": 1.0, "json": 2.0}
        return sizes.get(label_format, 1.0)

    @staticmethod
    def plan(
        dataset: DatasetSpec,
        model: ModelSpec,
        training: TrainingSpec,
    ) -> StorageEstimate:
        """
        Estimate total storage requirements.

        Returns:
            StorageEstimate with breakdown in MB
        """
        # Dataset size (images + labels)
        img_size = StoragePlanner.estimate_image_size_mb(dataset.image_resolution)
        label_size = StoragePlanner.estimate_label_size_kb(dataset.label_format) / 1024

        dataset_mb = dataset.num_images * (img_size + (label_size if dataset.has_labels else 0))
        augmented_mb = dataset_mb * dataset.augment_factor

        # Model weights
        model_mb = MODEL_SIZE_MB.get(model.variant, 100)

        # Checkpoints: save_period determines how many snapshots
        # Each checkpoint = model weights + optimizer state (~3x weights)
        num_checkpoints = training.epochs // training.save_period + 1
        checkpoint_mb = model_mb * 4 * num_checkpoints  # 4x for weights + optimizer state

        # Cache (YOLO cache files, ~10% of dataset)
        cache_mb = dataset_mb * 0.1

        total_mb = dataset_mb + augmented_mb + model_mb + checkpoint_mb + cache_mb

        return StorageEstimate(
            dataset_mb=dataset_mb,
            dataset_augmented_mb=augmented_mb,
            model_weights_mb=model_mb,
            checkpoints_mb=checkpoint_mb,
            cache_mb=cache_mb,
            total_mb=total_mb,
        )


class ComputePlanner:
    """Estimate compute requirements for training."""

    # VRAM requirements per image in batch (MB) - model + activations + gradients
    VRAM_PER_IMAGE_BASE = {
        "n": 150,   # ~150MB per image for YOLOv8n
        "s": 250,
        "m": 400,
        "l": 600,
        "x": 850,
    }

    # Training speed (images/second) on reference GPU (RTX 3090)
    IMAGES_PER_SEC_REF = {
        "n": 180,
        "s": 120,
        "m": 80,
        "l": 50,
        "x": 35,
    }

    # CPU training speed (images/second) - much slower than GPU
    IMAGES_PER_SEC_CPU = {
        "n": 8,     # ~8 img/sec on modern CPU
        "s": 5,
        "m": 3,
        "l": 2,
        "x": 1,
    }

    # CPU recommendations by model complexity
    CPU_RECOMMENDATIONS = {
        "n": "6+ cores (i5 / Ryzen 5)",
        "s": "8+ cores (i7 / Ryzen 7)",
        "m": "12+ cores (i9 / Ryzen 9)",
        "l": "16+ cores (Ryzen 9 / Threadripper)",
        "x": "24+ cores (Threadripper / EPYC)",
    }

    # System RAM requirements (GB) - includes model, data, overhead
    RAM_REQUIREMENTS = {
        "n": 8,     # 8GB minimum for YOLOv8n
        "s": 12,
        "m": 16,
        "l": 24,
        "x": 32,
    }

    # GPU recommendations by VRAM requirement
    GPU_RECOMMENDATIONS = {
        (0, 4): "T4 or CPU (slow)",
        (4, 8): "RTX 3060 / T4 / V100",
        (8, 12): "RTX 3070 / 3080 / A10G",
        (12, 16): "RTX 3090 / 4080 / A100",
        (16, 24): "RTX 4090 / A100 40GB",
        (24, 48): "A100 40GB / A6000",
        (48, 80): "A100 80GB",
        (80, 9999): "Multi-GPU setup",
    }

    # Cloud GPU costs (USD/hour) - approximate
    GPU_COST_PER_HOUR = {
        "T4": 0.20,
        "RTX 3060": 0.40,
        "RTX 3070": 0.60,
        "RTX 3080": 0.80,
        "RTX 3090": 1.00,
        "RTX 4080": 1.20,
        "RTX 4090": 1.50,
        "V100": 1.50,
        "A10G": 1.80,
        "A100": 2.50,
        "A100 80GB": 3.50,
        "A6000": 3.00,
    }

    @staticmethod
    def estimate_vram(
        model: ModelSpec,
        training: TrainingSpec,
        image_resolution: tuple,
    ) -> float:
        """
        Estimate VRAM requirement in MB.

        Formula:
        - Base model VRAM per image
        - Scale by resolution (relative to 640x640)
        - Multiply by batch size
        - Add overhead (~2GB base)
        """
        base_vram = ComputePlanner.VRAM_PER_IMAGE_BASE.get(model.variant, 400)

        # Resolution scaling factor
        ref_pixels = 640 * 640
        actual_pixels = image_resolution[0] * image_resolution[1]
        resolution_scale = actual_pixels / ref_pixels

        vram_mb = (base_vram * resolution_scale * training.batch_size) + 2048  # 2GB overhead
        return vram_mb

    @staticmethod
    def recommend_gpu(vram_mb: float) -> str:
        """Recommend GPU based on VRAM requirement."""
        vram_gb = vram_mb / 1024

        for (min_gb, max_gb), gpu in ComputePlanner.GPU_RECOMMENDATIONS.items():
            if min_gb <= vram_gb < max_gb:
                return gpu

        return "Multi-GPU setup"

    @staticmethod
    def estimate_training_time(
        dataset: DatasetSpec,
        model: ModelSpec,
        training: TrainingSpec,
    ) -> float:
        """
        Estimate training time in hours.

        Formula:
        - Images per epoch = num_images * augment_factor
        - Total iterations = images_per_epoch / batch_size * epochs
        - Seconds = iterations / images_per_second
        """
        images_per_epoch = dataset.num_images * dataset.augment_factor
        iterations_per_epoch = images_per_epoch / training.batch_size
        total_iterations = iterations_per_epoch * training.epochs

        # Get reference speed for this model
        ref_speed = ComputePlanner.IMAGES_PER_SEC_REF.get(model.variant, 80)

        # Adjust for image resolution
        ref_pixels = 640 * 640
        actual_pixels = dataset.image_resolution[0] * dataset.image_resolution[1]
        resolution_scale = actual_pixels / ref_pixels

        # Higher resolution = slower processing
        adjusted_speed = ref_speed / (resolution_scale ** 0.5)

        # Seconds = (images / speed) * scale_factor
        total_seconds = (images_per_epoch * training.epochs) / adjusted_speed

        return total_seconds / 3600  # Convert to hours

    @staticmethod
    def estimate_cost(
        training_hours: float,
        gpu_name: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        min_vram_gb: float = 0,
    ) -> Optional[float]:
        """
        Estimate cloud GPU cost in USD using live catalog.

        Args:
            training_hours: Estimated training time
            gpu_name: Specific GPU to use, or None for auto-select
            provider: Cloud provider to query
            min_vram_gb: Minimum VRAM requirement

        Returns:
            Cost in USD, or None if pricing unavailable
        """
        catalog = GPUCatalog()

        # If specific GPU requested, find it and calculate cost
        if gpu_name:
            for p in [provider, ProviderType.RUNPOD, ProviderType.HETZNER]:
                if p is None:
                    continue
                cost = catalog.estimate_cost(p, gpu_name, training_hours)
                if cost:
                    return cost

        # Otherwise find cheapest GPU meeting requirements
        cheapest = catalog.find_cheapest(min_vram_gb=min_vram_gb, provider=provider)
        if cheapest:
            return cheapest.price_per_hour * training_hours

        # Fallback to hardcoded pricing
        if gpu_name:
            cost_per_hour = ComputePlanner.GPU_COST_PER_HOUR.get(gpu_name)
            if cost_per_hour:
                return training_hours * cost_per_hour

        return training_hours * 1.50  # Default fallback

    @staticmethod
    def estimate_cpu_time(
        dataset: DatasetSpec,
        model: ModelSpec,
        training: TrainingSpec,
    ) -> float:
        """
        Estimate CPU-only training time in hours.

        CPU training is ~20-50x slower than GPU depending on model.
        """
        images_per_epoch = dataset.num_images * dataset.augment_factor

        # Get CPU speed for this model
        cpu_speed = ComputePlanner.IMAGES_PER_SEC_CPU.get(model.variant, 3)

        # Adjust for image resolution
        ref_pixels = 640 * 640
        actual_pixels = dataset.image_resolution[0] * dataset.image_resolution[1]
        resolution_scale = actual_pixels / ref_pixels

        # CPU scales poorly with resolution
        adjusted_speed = cpu_speed / (resolution_scale ** 0.7)

        # Calculate time
        total_seconds = (images_per_epoch * training.epochs) / adjusted_speed

        return total_seconds / 3600  # Convert to hours

    @staticmethod
    def recommend_cpu(model_variant: str) -> str:
        """Recommend CPU based on model complexity."""
        return ComputePlanner.CPU_RECOMMENDATIONS.get(model_variant, "8+ cores recommended")

    @staticmethod
    def estimate_ram(
        model_variant: str,
        batch_size: int,
    ) -> float:
        """
        Estimate system RAM requirement in GB.

        CPU training needs more RAM since data and model are in system memory.
        """
        base_ram = ComputePlanner.RAM_REQUIREMENTS.get(model_variant, 16)

        # Batch size scaling (double batch = ~50% more RAM)
        batch_scale = 1 + (batch_size - 16) * 0.05

        return base_ram * max(1.0, batch_scale)

    @staticmethod
    def plan(
        dataset: DatasetSpec,
        model: ModelSpec,
        training: TrainingSpec,
        gpu_name: Optional[str] = None,
        provider: Optional[ProviderType] = None,
    ) -> ComputeEstimate:
        """
        Estimate compute requirements using GPU catalog.

        Returns:
            ComputeEstimate with VRAM, GPU recommendation, training time, cost, CPU estimates
        """
        vram_mb = ComputePlanner.estimate_vram(model, training, dataset.image_resolution)
        vram_gb = vram_mb / 1024

        # Use GPU catalog to find best option
        catalog = GPUCatalog()

        # Get recommended GPU from catalog
        if gpu_name:
            # Use specific GPU
            gpu = gpu_name
        else:
            # Find cheapest GPU that meets VRAM requirement
            best_gpu = catalog.find_cheapest(min_vram_gb=vram_gb, provider=provider)
            gpu = best_gpu.gpu_name if best_gpu else ComputePlanner.recommend_gpu(vram_mb)

        hours = ComputePlanner.estimate_training_time(dataset, model, training)

        # Estimate cost using catalog with actual VRAM requirement
        cost = ComputePlanner.estimate_cost(
            hours,
            gpu_name=gpu_name,
            provider=provider,
            min_vram_gb=vram_gb,
        )

        # CPU estimates
        cpu_hours = ComputePlanner.estimate_cpu_time(dataset, model, training)
        cpu_rec = ComputePlanner.recommend_cpu(model.variant)
        ram_gb = ComputePlanner.estimate_ram(model.variant, training.batch_size)

        return ComputeEstimate(
            vram_mb=vram_mb,
            recommended_gpu=gpu,
            training_hours=hours,
            estimated_cost_usd=cost,
            cpu_hours=cpu_hours,
            recommended_cpu=cpu_rec,
            ram_gb=ram_gb,
        )


def plan_training(
    num_images: int,
    resolution: tuple = (640, 640),
    model_variant: str = "m",
    epochs: int = 60,
    batch_size: int = 32,
    augment_factor: int = 1,
    gpu_name: Optional[str] = None,
    provider: Optional[ProviderType] = None,
) -> dict:
    """
    Quick planning function - returns storage and compute estimates.

    Args:
        num_images: Number of training images
        resolution: Image resolution (width, height)
        model_variant: YOLO model variant (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        augment_factor: Augmentation multiplier
        gpu_name: Specific GPU for cost estimation
        provider: Cloud provider for cost estimation (runpod, hetzner)

    Returns:
        Dict with storage and compute estimates
    """
    dataset = DatasetSpec(
        num_images=num_images,
        image_resolution=resolution,
        augment_factor=augment_factor,
    )

    model = ModelSpec(family="yolo", variant=model_variant)

    training = TrainingSpec(
        epochs=epochs,
        batch_size=batch_size,
        save_period=10,
    )

    storage = StoragePlanner.plan(dataset, model, training)
    compute = ComputePlanner.plan(dataset, model, training, gpu_name, provider)

    return {
        "storage": storage.to_dict(),
        "compute": compute.to_dict(),
        "specs": {
            "dataset": f"{num_images} images @ {resolution[0]}x{resolution[1]}",
            "model": f"YOLOv8{model_variant}",
            "training": f"{epochs} epochs, batch={batch_size}",
        },
    }
