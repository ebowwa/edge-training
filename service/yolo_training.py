"""
Training Service for YOLO.
Handles model training lifecycle including fresh training and resume.
"""

import logging
import os

from ultralytics import YOLO

from service.config import TrainingConfig, TrainingResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TrainingService:
    """Service for model training operations."""

    @staticmethod
    def train(yaml_path: str, config: TrainingConfig) -> TrainingResult:
        """
        Train a YOLO model.

        Args:
            yaml_path: Path to data.yaml configuration.
            config: Training configuration.

        Returns:
            TrainingResult with paths and metrics.
        """
        if config.weights:
            logging.info(f"Loading weights from: {config.weights}")
            model = YOLO(config.weights)
        else:
            logging.info(
                f"Starting training with {config.base_model} pretrained weights..."
            )
            model = YOLO(config.base_model)

        results = model.train(
            data=yaml_path,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            project=config.project,
            name=config.name,
            exist_ok=True,
        )

        best_path = os.path.join(
            config.project, config.name, "weights", "best.pt"
        )
        last_path = os.path.join(
            config.project, config.name, "weights", "last.pt"
        )

        return TrainingResult(
            best_model_path=best_path,
            last_model_path=last_path,
            metrics=results.results_dict if hasattr(results, 'results_dict') else {},
            epochs_completed=config.epochs
        )

    @staticmethod
    def resume(project: str, name: str) -> TrainingResult:
        """
        Resume training from last checkpoint.

        Args:
            project: Project directory.
            name: Experiment name.

        Returns:
            TrainingResult with paths and metrics.

        Raises:
            FileNotFoundError: If checkpoint not found.
            ValueError: If training already completed.
        """
        checkpoint_path = os.path.join(project, name, "weights", "last.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Make sure training was run before resuming."
            )

        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)

        try:
            results = model.train(resume=True)
        except AssertionError as e:
            if "training to" in str(e) and "is finished" in str(e):
                raise ValueError(
                    "Training already completed for this run! "
                    "Use train() with weights parameter to continue "
                    "training with more epochs."
                ) from None
            else:
                raise

        best_path = os.path.join(project, name, "weights", "best.pt")
        last_path = checkpoint_path

        return TrainingResult(
            best_model_path=best_path,
            last_model_path=last_path,
            metrics=results.results_dict if hasattr(results, 'results_dict') else {},
            epochs_completed=0  # Unknown when resuming
        )

    @staticmethod
    def get_best_model_path(project: str, name: str) -> str:
        """
        Get path to best model weights.

        Args:
            project: Project directory.
            name: Experiment name.

        Returns:
            Path to best.pt weights.

        Raises:
            FileNotFoundError: If weights not found.
        """
        path = os.path.join(project, name, "weights", "best.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Best model not found: {path}")
        return path

    @staticmethod
    def get_last_model_path(project: str, name: str) -> str:
        """
        Get path to last checkpoint weights.

        Args:
            project: Project directory.
            name: Experiment name.

        Returns:
            Path to last.pt weights.

        Raises:
            FileNotFoundError: If weights not found.
        """
        path = os.path.join(project, name, "weights", "last.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Last checkpoint not found: {path}")
        return path

    @staticmethod
    def check_training_complete(project: str, name: str) -> bool:
        """
        Check if training is already complete for a run.

        Args:
            project: Project directory.
            name: Experiment name.

        Returns:
            True if training is complete, False otherwise.
        """
        checkpoint_path = os.path.join(project, name, "weights", "last.pt")

        if not os.path.exists(checkpoint_path):
            return False

        try:
            model = YOLO(checkpoint_path)
            ckpt = model.ckpt
            if ckpt and 'epoch' in ckpt and 'train_args' in ckpt:
                current_epoch = ckpt['epoch']
                target_epochs = ckpt.get('train_args', {}).get('epochs', 0)
                return current_epoch >= target_epochs
        except Exception:
            pass

        return False
