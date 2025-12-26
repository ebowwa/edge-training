"""
Composable Modal Volume Pipeline

Base classes for building reusable GPU processing pipelines with Modal volumes.
"""

import modal
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from abc import ABC, abstractmethod


class VolumeProcessor(ABC):
    """
    Base class for volume-based processing on Modal.
    
    Subclass this to create reusable processors:
    - DedupProcessor
    - SAM2Processor
    - VLMAnnotationProcessor
    - TrainingProcessor
    - ExportProcessor
    """
    
    def __init__(
        self,
        volume_name: str,
        app_name: str,
        gpu: str = "A10G",
        timeout: int = 3600
    ):
        self.volume_name = volume_name
        self.app_name = app_name
        self.gpu = gpu
        self.timeout = timeout
        
        # Create app and volume reference
        self.app = modal.App(app_name)
        self.volume = modal.Volume.from_name(volume_name, create_if_missing=False)
    
    @abstractmethod
    def get_image(self) -> modal.Image:
        """Return Modal image with required dependencies."""
        pass
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """Main processing logic. Runs on GPU."""
        pass
    
    def create_function(self, volumes: Optional[Dict[str, modal.Volume]] = None):
        """
        Create Modal function with volume mounted.
        
        Returns decorated process method.
        """
        if volumes is None:
            volumes = {"/data": self.volume}
        
        @self.app.function(
            image=self.get_image(),
            gpu=self.gpu,
            timeout=self.timeout,
            volumes=volumes
        )
        def run(**kwargs):
            return self.process(**kwargs)
        
        return run
    
    def run_remote(self, **kwargs) -> Dict[str, Any]:
        """Execute processing on Modal."""
        func = self.create_function()
        return func.remote(**kwargs)


class DedupProcessor(VolumeProcessor):
    """Deduplication using DINOv2."""
    
    def get_image(self) -> modal.Image:
        return modal.Image.debian_slim("3.11").pip_install(
            "torch", "transformers", "faiss-cpu", "pillow", "tqdm", "numpy"
        )
    
    def process(
        self,
        threshold: float = 0.99,
        model_name: str = "facebook/dinov2-base",
        batch_size: int = 64,
        reuse_embeddings: bool = True,  # NEW: reuse saved embeddings
        embeddings_path: str = "/data/embeddings.npy"
    ) -> Dict[str, Any]:
        """Run deduplication. See scripts/modal_dedup.py for implementation."""
        # Implementation would go here
        # (current modal_dedup.py logic)
        pass


class SAM2Processor(VolumeProcessor):
    """Segmentation using SAM2."""
    
    def get_image(self) -> modal.Image:
        return modal.Image.debian_slim("3.11").pip_install(
            "torch", "torchvision", "pillow", "numpy",
            "git+https://github.com/facebookresearch/segment-anything-2.git"
        )
    
    def process(
        self,
        output_dir: str = "/data/masks",
        checkpoint: str = "sam2_hiera_large.pt"
    ) -> Dict[str, Any]:
        """Generate segmentation masks for all images."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from PIL import Image
        import torch
        from pathlib import Path
        
        # Load SAM2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2 = build_sam2(checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2)
        
        # Process images
        data_dir = Path("/data")
        processed = 0
        
        for split in ["test", "train", "valid"]:
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
            
            for img_path in split_dir.glob("*.jpg"):
                img = Image.open(img_path)
                predictor.set_image(img)
                
                # Auto-generate masks
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    multimask_output=True
                )
                
                # Save best mask
                best_mask = masks[scores.argmax()]
                # Save logic here
                processed += 1
        
        return {"processed": processed}


class RFDETRTrainer(VolumeProcessor):
    """Train RF-DETR for edge deployment."""
    
    def get_image(self) -> modal.Image:
        return modal.Image.debian_slim("3.11").pip_install(
            "torch", "torchvision", "ultralytics",
            "transformers", "pillow", "pyyaml"
        )
    
    def process(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        model: str = "rtdetr-l.pt"
    ) -> Dict[str, Any]:
        """Train RF-DETR on volume data."""
        from ultralytics import RTDETR
        
        # Load model
        model = RTDETR(model)
        
        # Train on volume data
        results = model.train(
            data="/data/data.yaml",  # COCO format config
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=0  # GPU
        )
        
        # Save to volume
        model.save("/data/models/rfdetr_trained.pt")
        
        return {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "model_path": "/data/models/rfdetr_trained.pt"
        }


class EdgeExporter(VolumeProcessor):
    """Export model for edge deployment."""
    
    def get_image(self) -> modal.Image:
        return modal.Image.debian_slim("3.11").pip_install(
            "torch", "ultralytics", "onnx", "coremltools"
        )
    
    def process(
        self,
        model_path: str,
        format: str = "onnx",  # onnx, coreml, tflite
        quantize: bool = True
    ) -> Dict[str, Any]:
        """Export and optimize for edge."""
        from ultralytics import RTDETR
        
        model = RTDETR(model_path)
        
        # Export
        export_path = model.export(
            format=format,
            int8=quantize,
            imgsz=640
        )
        
        return {
            "export_path": str(export_path),
            "format": format,
            "quantized": quantize
        }


# Composable pipeline builder
class ModalPipeline:
    """
    Compose multiple processors into a pipeline.
    
    Example:
        pipeline = ModalPipeline("usd-dataset-test")
        pipeline.add(DedupProcessor, threshold=0.99)
        pipeline.add(SAM2Processor)
        pipeline.add(RFDETRTrainer, epochs=100)
        pipeline.add(EdgeExporter, format="onnx")
        results = pipeline.run()
    """
    
    def __init__(self, volume_name: str):
        self.volume_name = volume_name
        self.processors = []
    
    def add(self, processor_class: type, **kwargs):
        """Add a processor to the pipeline."""
        self.processors.append((processor_class, kwargs))
        return self
    
    def run(self) -> list:
        """Execute all processors in sequence."""
        results = []
        
        for processor_class, kwargs in self.processors:
            print(f"\n🔄 Running {processor_class.__name__}...")
            
            processor = processor_class(
                volume_name=self.volume_name,
                app_name=f"pipeline-{processor_class.__name__.lower()}"
            )
            
            result = processor.run_remote(**kwargs)
            results.append({
                "processor": processor_class.__name__,
                "result": result
            })
            
            print(f"✓ {processor_class.__name__} complete")
        
        return results