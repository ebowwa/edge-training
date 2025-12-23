"""
Base Pipeline Abstractions.

Provides composable pipeline building blocks with a functional approach.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

# Generic type for pipeline data
T = TypeVar('T')
U = TypeVar('U')


@dataclass
class PipelineContext:
    """
    Shared context passed through pipeline stages.
    Allows stages to share state without tight coupling.
    """
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
    
    def add_error(self, error: str) -> None:
        self.errors.append(error)
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class PipelineStage(ABC, Generic[T, U]):
    """
    Abstract base for a single pipeline stage.
    
    Each stage transforms input T to output U.
    Stages can be composed using the >> operator.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for logging/debugging."""
        pass
    
    @abstractmethod
    def process(self, input_data: T, context: PipelineContext) -> U:
        """Process input and return output."""
        pass
    
    def __rshift__(self, other: 'PipelineStage') -> 'ComposedStage':
        """Compose stages using >> operator."""
        return ComposedStage(self, other)
    
    def __call__(self, input_data: T, context: Optional[PipelineContext] = None) -> U:
        """Allow calling stage directly."""
        if context is None:
            context = PipelineContext()
        return self.process(input_data, context)


class ComposedStage(PipelineStage):
    """Two stages composed into one."""
    
    def __init__(self, first: PipelineStage, second: PipelineStage):
        self.first = first
        self.second = second
    
    @property
    def name(self) -> str:
        return f"{self.first.name} >> {self.second.name}"
    
    def process(self, input_data: Any, context: PipelineContext) -> Any:
        intermediate = self.first.process(input_data, context)
        return self.second.process(intermediate, context)


class FunctionStage(PipelineStage[T, U]):
    """
    Wrap a simple function as a pipeline stage.
    
    Example:
        normalize = FunctionStage("normalize", lambda x, ctx: x / 255.0)
    """
    
    def __init__(self, name: str, func: Callable[[T, PipelineContext], U]):
        self._name = name
        self._func = func
    
    @property
    def name(self) -> str:
        return self._name
    
    def process(self, input_data: T, context: PipelineContext) -> U:
        return self._func(input_data, context)


class Pipeline(Generic[T, U]):
    """
    A complete pipeline composed of multiple stages.
    
    Example:
        pipeline = Pipeline([
            PreprocessStage(),
            InferenceStage(model),
            PostprocessStage(),
        ])
        result = pipeline.run(input_data)
    """
    
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
    
    @property
    def name(self) -> str:
        return " >> ".join(s.name for s in self.stages)
    
    def run(self, input_data: T, context: Optional[PipelineContext] = None) -> U:
        """Run all stages in sequence."""
        if context is None:
            context = PipelineContext()
        
        current = input_data
        for stage in self.stages:
            try:
                current = stage.process(current, context)
            except Exception as e:
                context.add_error(f"[{stage.name}] {str(e)}")
                raise
        
        return current
    
    def add_stage(self, stage: PipelineStage) -> 'Pipeline':
        """Add a stage and return self for chaining."""
        self.stages.append(stage)
        return self
    
    def __call__(self, input_data: T, context: Optional[PipelineContext] = None) -> U:
        return self.run(input_data, context)


class ConditionalStage(PipelineStage[T, T]):
    """
    A stage that conditionally executes based on a predicate.
    
    Example:
        augment_if_training = ConditionalStage(
            "augment",
            AugmentStage(),
            condition=lambda x, ctx: ctx.get("is_training", False)
        )
    """
    
    def __init__(
        self,
        name: str,
        stage: PipelineStage,
        condition: Callable[[Any, PipelineContext], bool],
    ):
        self._name = name
        self.stage = stage
        self.condition = condition
    
    @property
    def name(self) -> str:
        return f"{self._name}?"
    
    def process(self, input_data: T, context: PipelineContext) -> T:
        if self.condition(input_data, context):
            return self.stage.process(input_data, context)
        return input_data


class ParallelStage(PipelineStage[T, Dict[str, Any]]):
    """
    Run multiple stages in parallel on the same input.
    
    Example:
        multi_detect = ParallelStage({
            "objects": ObjectDetector(),
            "faces": FaceDetector(),
            "text": TextDetector(),
        })
    """
    
    def __init__(self, stages: Dict[str, PipelineStage]):
        self.stages_dict = stages
    
    @property
    def name(self) -> str:
        return f"parallel({', '.join(self.stages_dict.keys())})"
    
    def process(self, input_data: T, context: PipelineContext) -> Dict[str, Any]:
        results = {}
        for key, stage in self.stages_dict.items():
            try:
                results[key] = stage.process(input_data, context)
            except Exception as e:
                context.add_error(f"[{key}] {str(e)}")
                results[key] = None
        return results
    
    def attach(self, name: str, stage: PipelineStage) -> 'ParallelStage':
        """Attach a new parallel branch."""
        self.stages_dict[name] = stage
        return self
    
    def detach(self, name: str) -> Optional[PipelineStage]:
        """Detach a parallel branch by name."""
        return self.stages_dict.pop(name, None)


class PipelineRegistry:
    """
    Registry for named pipelines that can be looked up and attached.
    
    Example:
        registry = PipelineRegistry()
        registry.register("preprocessing", PreprocessPipeline())
        registry.register("inference", InferencePipeline())
        
        # Later, compose them
        full = registry.get("preprocessing") >> registry.get("inference")
    """
    
    _instance: Optional['PipelineRegistry'] = None
    
    def __init__(self):
        self._pipelines: Dict[str, Pipeline] = {}
        self._stages: Dict[str, PipelineStage] = {}
    
    @classmethod
    def default(cls) -> 'PipelineRegistry':
        """Get the default global registry."""
        if cls._instance is None:
            cls._instance = PipelineRegistry()
        return cls._instance
    
    def register_pipeline(self, name: str, pipeline: Pipeline) -> None:
        """Register a pipeline by name."""
        self._pipelines[name] = pipeline
    
    def register_stage(self, name: str, stage: PipelineStage) -> None:
        """Register a stage by name."""
        self._stages[name] = stage
    
    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """Get a pipeline by name."""
        return self._pipelines.get(name)
    
    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name."""
        return self._stages.get(name)
    
    def list_pipelines(self) -> List[str]:
        """List all registered pipeline names."""
        return list(self._pipelines.keys())
    
    def list_stages(self) -> List[str]:
        """List all registered stage names."""
        return list(self._stages.keys())
    
    def compose(self, *names: str) -> Pipeline:
        """
        Compose multiple registered pipelines/stages into one.
        
        Example:
            full = registry.compose("preprocess", "inference", "postprocess")
        """
        stages = []
        for name in names:
            if name in self._pipelines:
                stages.extend(self._pipelines[name].stages)
            elif name in self._stages:
                stages.append(self._stages[name])
            else:
                raise KeyError(f"No pipeline or stage named '{name}'")
        return Pipeline(stages)


class AttachablePipeline(Pipeline[T, U]):
    """
    A pipeline that supports dynamic attachment/detachment of stages.
    
    Example:
        pipeline = AttachablePipeline([
            PreprocessStage(),
            InferenceStage(),
        ])
        
        # Attach a new stage
        pipeline.attach("postprocess", PostprocessStage())
        
        # Attach another pipeline
        pipeline.attach_pipeline(export_pipeline)
        
        # Detach a stage
        pipeline.detach("postprocess")
    """
    
    def __init__(self, stages: Optional[List[PipelineStage]] = None):
        super().__init__(stages or [])
        self._named_stages: Dict[str, int] = {}  # name -> index
    
    def attach(self, name: str, stage: PipelineStage, position: Optional[int] = None) -> 'AttachablePipeline':
        """
        Attach a stage with a name for later reference.
        
        Args:
            name: Unique name for the stage
            stage: The stage to attach
            position: Optional position (None = append at end)
        """
        if name in self._named_stages:
            raise ValueError(f"Stage '{name}' already attached")
        
        if position is None:
            position = len(self.stages)
        
        self.stages.insert(position, stage)
        
        # Update indices for stages after insertion
        for n, idx in self._named_stages.items():
            if idx >= position:
                self._named_stages[n] = idx + 1
        
        self._named_stages[name] = position
        return self
    
    def detach(self, name: str) -> Optional[PipelineStage]:
        """
        Detach a named stage.
        
        Returns the detached stage, or None if not found.
        """
        if name not in self._named_stages:
            return None
        
        idx = self._named_stages[name]
        stage = self.stages.pop(idx)
        del self._named_stages[name]
        
        # Update indices for stages after removal
        for n, i in self._named_stages.items():
            if i > idx:
                self._named_stages[n] = i - 1
        
        return stage
    
    def attach_pipeline(self, pipeline: Pipeline, prefix: str = "") -> 'AttachablePipeline':
        """
        Attach all stages from another pipeline.
        
        Args:
            pipeline: Pipeline to attach
            prefix: Optional prefix for stage names
        """
        for i, stage in enumerate(pipeline.stages):
            name = f"{prefix}{stage.name}" if prefix else f"attached_{i}_{stage.name}"
            self.attach(name, stage)
        return self
    
    def replace(self, name: str, new_stage: PipelineStage) -> 'AttachablePipeline':
        """Replace a named stage with a new one."""
        if name not in self._named_stages:
            raise KeyError(f"No stage named '{name}'")
        
        idx = self._named_stages[name]
        self.stages[idx] = new_stage
        return self
    
    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name."""
        if name not in self._named_stages:
            return None
        return self.stages[self._named_stages[name]]
    
    def list_attached(self) -> List[str]:
        """List all attached stage names."""
        return list(self._named_stages.keys())
    
    def __rshift__(self, other: 'Pipeline') -> 'AttachablePipeline':
        """Chain with another pipeline using >> operator."""
        new_pipeline = AttachablePipeline(self.stages.copy())
        new_pipeline._named_stages = self._named_stages.copy()
        new_pipeline.attach_pipeline(other)
        return new_pipeline
