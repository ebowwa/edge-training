"""
Export Service for YOLO.
Handles model export to various deployment formats.
"""

import logging
import os

from ultralytics import YOLO

from .config import ExportConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ExportService:
    """Service for model export operations."""

    # Supported export formats
    SUPPORTED_FORMATS = [
        'torchscript',
        'onnx',
        'openvino',
        'engine',  # TensorRT
        'coreml',
        'saved_model',  # TensorFlow SavedModel
        'pb',  # TensorFlow GraphDef
        'tflite',
        'edgetpu',
        'tfjs',
        'paddle',
        'ncnn',
    ]

    @staticmethod
    def export(model_path: str, config: ExportConfig = None) -> str:
        """
        Export model to specified format.

        Args:
            model_path: Path to trained model weights.
            config: Export configuration.

        Returns:
            Path to exported model.

        Raises:
            FileNotFoundError: If model not found.
            ValueError: If format not supported.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if config is None:
            config = ExportConfig()

        if config.format not in ExportService.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {config.format}. "
                f"Supported: {ExportService.SUPPORTED_FORMATS}"
            )

        logging.info(f"Exporting model to {config.format} format...")

        model = YOLO(model_path)
        export_path = model.export(
            format=config.format,
            imgsz=config.imgsz,
            half=config.half,
            dynamic=config.dynamic
        )

        logging.info(f"Model exported to: {export_path}")
        return export_path

    @staticmethod
    def export_ncnn(model_path: str, output_dir: str = None) -> str:
        """
        Export model to NCNN format (optimized for mobile/edge).

        Args:
            model_path: Path to trained model weights.
            output_dir: Optional output directory.

        Returns:
            Path to exported NCNN model directory.
        """
        config = ExportConfig(format='ncnn')
        return ExportService.export(model_path, config)

    @staticmethod
    def export_onnx(
        model_path: str,
        output_dir: str = None,
        dynamic: bool = False
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            model_path: Path to trained model weights.
            output_dir: Optional output directory.
            dynamic: Whether to use dynamic input shapes.

        Returns:
            Path to exported ONNX model.
        """
        config = ExportConfig(format='onnx', dynamic=dynamic)
        return ExportService.export(model_path, config)

    @staticmethod
    def export_coreml(model_path: str, half: bool = False) -> str:
        """
        Export model to CoreML format (for iOS/macOS).

        Args:
            model_path: Path to trained model weights.
            half: Whether to use FP16 quantization.

        Returns:
            Path to exported CoreML model.
        """
        config = ExportConfig(format='coreml', half=half)
        return ExportService.export(model_path, config)

    @staticmethod
    def export_tflite(model_path: str, half: bool = False) -> str:
        """
        Export model to TFLite format (for mobile/edge).

        Args:
            model_path: Path to trained model weights.
            half: Whether to use FP16 quantization.

        Returns:
            Path to exported TFLite model.
        """
        config = ExportConfig(format='tflite', half=half)
        return ExportService.export(model_path, config)
