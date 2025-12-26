"""
Validation Service for YOLO.
Handles model validation and benchmarking.
"""

import logging
import os

from ultralytics import YOLO

from service.config import ValidationConfig, ValidationResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ValidationService:
    """Service for model validation operations."""

    @staticmethod
    def validate(
        model_path: str,
        yaml_path: str,
        config: ValidationConfig = None
    ) -> ValidationResult:
        """
        Validate a trained model on a dataset.

        Args:
            model_path: Path to trained model weights.
            yaml_path: Path to data.yaml configuration.
            config: Validation configuration.

        Returns:
            ValidationResult with metrics.

        Raises:
            FileNotFoundError: If model or yaml not found.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML config not found: {yaml_path}")

        if config is None:
            config = ValidationConfig()

        logging.info(f"Validating model: {model_path}")
        logging.info(f"Using dataset: {yaml_path}")

        model = YOLO(model_path)
        results = model.val(
            data=yaml_path,
            imgsz=config.imgsz,
            split=config.split,
            project=config.project,
            name=config.name
        )

        # Extract metrics
        metrics = {}
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict

        return ValidationResult(
            map50=getattr(results, 'map50', 0.0),
            map50_95=getattr(results, 'map', 0.0),
            precision=metrics.get('metrics/precision(B)', 0.0),
            recall=metrics.get('metrics/recall(B)', 0.0),
            metrics=metrics
        )

    @staticmethod
    def benchmark(
        model_path: str,
        yaml_path: str,
        splits: list = None
    ) -> dict:
        """
        Benchmark a model on multiple dataset splits.

        Args:
            model_path: Path to trained model weights.
            yaml_path: Path to data.yaml configuration.
            splits: List of splits to validate on (default: ['val', 'test']).

        Returns:
            Dictionary mapping split -> ValidationResult.
        """
        if splits is None:
            splits = ['val', 'test']

        results = {}
        for split in splits:
            try:
                config = ValidationConfig(split=split, name=f"benchmark_{split}")
                results[split] = ValidationService.validate(
                    model_path, yaml_path, config
                )
                logging.info(
                    f"Split '{split}': mAP@0.5={results[split].map50:.4f}, "
                    f"mAP@0.5:0.95={results[split].map50_95:.4f}"
                )
            except Exception as e:
                logging.warning(f"Could not validate on split '{split}': {e}")

        return results
