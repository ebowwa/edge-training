"""
Tests for base pipeline abstractions.
"""

import sys
import unittest
from pathlib import Path

# Add paths
_test_dir = Path(__file__).parent
_pipeline_dir = _test_dir.parent
sys.path.insert(0, str(_pipeline_dir))

from base import (
    PipelineContext,
    PipelineStage,
    FunctionStage,
    Pipeline,
    ConditionalStage,
    ParallelStage,
    PipelineRegistry,
    AttachablePipeline,
)


class TestPipelineContext(unittest.TestCase):
    """Tests for PipelineContext."""
    
    def test_get_set(self):
        """Test get/set data."""
        ctx = PipelineContext()
        ctx.set("key", "value")
        self.assertEqual(ctx.get("key"), "value")
    
    def test_get_default(self):
        """Test get with default."""
        ctx = PipelineContext()
        self.assertIsNone(ctx.get("missing"))
        self.assertEqual(ctx.get("missing", "default"), "default")
    
    def test_errors(self):
        """Test error handling."""
        ctx = PipelineContext()
        self.assertFalse(ctx.has_errors())
        
        ctx.add_error("Something went wrong")
        self.assertTrue(ctx.has_errors())
        self.assertEqual(len(ctx.errors), 1)


class TestFunctionStage(unittest.TestCase):
    """Tests for FunctionStage."""
    
    def test_basic_function(self):
        """Test wrapping a simple function."""
        double = FunctionStage("double", lambda x, ctx: x * 2)
        
        result = double(5)
        self.assertEqual(result, 10)
    
    def test_function_with_context(self):
        """Test function that uses context."""
        def add_offset(x, ctx):
            offset = ctx.get("offset", 0)
            return x + offset
        
        stage = FunctionStage("add_offset", add_offset)
        ctx = PipelineContext()
        ctx.set("offset", 10)
        
        result = stage.process(5, ctx)
        self.assertEqual(result, 15)
    
    def test_name(self):
        """Test stage name."""
        stage = FunctionStage("my_stage", lambda x, ctx: x)
        self.assertEqual(stage.name, "my_stage")


class TestPipeline(unittest.TestCase):
    """Tests for Pipeline."""
    
    def test_single_stage(self):
        """Test pipeline with single stage."""
        double = FunctionStage("double", lambda x, ctx: x * 2)
        pipeline = Pipeline([double])
        
        result = pipeline.run(5)
        self.assertEqual(result, 10)
    
    def test_multiple_stages(self):
        """Test pipeline with multiple stages."""
        double = FunctionStage("double", lambda x, ctx: x * 2)
        add_one = FunctionStage("add_one", lambda x, ctx: x + 1)
        
        pipeline = Pipeline([double, add_one])
        result = pipeline.run(5)
        
        # (5 * 2) + 1 = 11
        self.assertEqual(result, 11)
    
    def test_pipeline_name(self):
        """Test pipeline name composition."""
        s1 = FunctionStage("first", lambda x, ctx: x)
        s2 = FunctionStage("second", lambda x, ctx: x)
        
        pipeline = Pipeline([s1, s2])
        self.assertEqual(pipeline.name, "first >> second")
    
    def test_callable(self):
        """Test pipeline as callable."""
        pipeline = Pipeline([FunctionStage("identity", lambda x, ctx: x)])
        result = pipeline(42)
        self.assertEqual(result, 42)
    
    def test_add_stage(self):
        """Test adding stages."""
        pipeline = Pipeline([])
        pipeline.add_stage(FunctionStage("a", lambda x, ctx: x + 1))
        pipeline.add_stage(FunctionStage("b", lambda x, ctx: x * 2))
        
        result = pipeline.run(5)
        # (5 + 1) * 2 = 12
        self.assertEqual(result, 12)
    
    def test_context_shared(self):
        """Test that context is shared across stages."""
        def stage1(x, ctx):
            ctx.set("seen", x)
            return x + 1
        
        def stage2(x, ctx):
            seen = ctx.get("seen")
            return x + seen
        
        pipeline = Pipeline([
            FunctionStage("s1", stage1),
            FunctionStage("s2", stage2),
        ])
        
        result = pipeline.run(5)
        # s1: sets seen=5, returns 6
        # s2: returns 6 + 5 = 11
        self.assertEqual(result, 11)


class TestStageComposition(unittest.TestCase):
    """Tests for stage composition with >> operator."""
    
    def test_compose_two_stages(self):
        """Test composing two stages."""
        double = FunctionStage("double", lambda x, ctx: x * 2)
        add_one = FunctionStage("add_one", lambda x, ctx: x + 1)
        
        composed = double >> add_one
        result = composed(5)
        
        self.assertEqual(result, 11)
    
    def test_composed_name(self):
        """Test composed stage name."""
        s1 = FunctionStage("a", lambda x, ctx: x)
        s2 = FunctionStage("b", lambda x, ctx: x)
        
        composed = s1 >> s2
        self.assertEqual(composed.name, "a >> b")


class TestConditionalStage(unittest.TestCase):
    """Tests for ConditionalStage."""
    
    def test_condition_true(self):
        """Test when condition is true."""
        double = FunctionStage("double", lambda x, ctx: x * 2)
        conditional = ConditionalStage(
            "maybe_double",
            double,
            condition=lambda x, ctx: ctx.get("do_double", False),
        )
        
        ctx = PipelineContext()
        ctx.set("do_double", True)
        
        result = conditional.process(5, ctx)
        self.assertEqual(result, 10)
    
    def test_condition_false(self):
        """Test when condition is false (passthrough)."""
        double = FunctionStage("double", lambda x, ctx: x * 2)
        conditional = ConditionalStage(
            "maybe_double",
            double,
            condition=lambda x, ctx: ctx.get("do_double", False),
        )
        
        ctx = PipelineContext()
        ctx.set("do_double", False)
        
        result = conditional.process(5, ctx)
        self.assertEqual(result, 5)  # Unchanged


class TestParallelStage(unittest.TestCase):
    """Tests for ParallelStage."""
    
    def test_parallel_execution(self):
        """Test parallel execution of multiple stages."""
        parallel = ParallelStage({
            "doubled": FunctionStage("double", lambda x, ctx: x * 2),
            "squared": FunctionStage("square", lambda x, ctx: x ** 2),
            "negated": FunctionStage("negate", lambda x, ctx: -x),
        })
        
        result = parallel(5)
        
        self.assertEqual(result["doubled"], 10)
        self.assertEqual(result["squared"], 25)
        self.assertEqual(result["negated"], -5)
    
    def test_parallel_name(self):
        """Test parallel stage name."""
        parallel = ParallelStage({
            "a": FunctionStage("a", lambda x, ctx: x),
            "b": FunctionStage("b", lambda x, ctx: x),
        })
        
        self.assertIn("a", parallel.name)
        self.assertIn("b", parallel.name)
    
    def test_attach_detach(self):
        """Test attaching and detaching parallel branches."""
        parallel = ParallelStage({
            "a": FunctionStage("a", lambda x, ctx: x + 1),
        })
        
        # Attach a new branch
        parallel.attach("b", FunctionStage("b", lambda x, ctx: x * 2))
        result = parallel(5)
        self.assertEqual(result["a"], 6)
        self.assertEqual(result["b"], 10)
        
        # Detach a branch
        detached = parallel.detach("a")
        self.assertIsNotNone(detached)
        
        result2 = parallel(5)
        self.assertNotIn("a", result2)
        self.assertEqual(result2["b"], 10)


class TestPipelineRegistry(unittest.TestCase):
    """Tests for PipelineRegistry."""
    
    def test_register_and_get_stage(self):
        """Test registering and retrieving stages."""
        registry = PipelineRegistry()
        stage = FunctionStage("double", lambda x, ctx: x * 2)
        
        registry.register_stage("double", stage)
        
        retrieved = registry.get_stage("double")
        self.assertEqual(retrieved, stage)
    
    def test_register_and_get_pipeline(self):
        """Test registering and retrieving pipelines."""
        registry = PipelineRegistry()
        pipeline = Pipeline([
            FunctionStage("a", lambda x, ctx: x + 1),
            FunctionStage("b", lambda x, ctx: x * 2),
        ])
        
        registry.register_pipeline("my_pipeline", pipeline)
        
        retrieved = registry.get_pipeline("my_pipeline")
        self.assertEqual(retrieved, pipeline)
    
    def test_list_stages_and_pipelines(self):
        """Test listing registered items."""
        registry = PipelineRegistry()
        registry.register_stage("s1", FunctionStage("s1", lambda x, ctx: x))
        registry.register_stage("s2", FunctionStage("s2", lambda x, ctx: x))
        registry.register_pipeline("p1", Pipeline([]))
        
        self.assertEqual(set(registry.list_stages()), {"s1", "s2"})
        self.assertEqual(registry.list_pipelines(), ["p1"])
    
    def test_compose(self):
        """Test composing registered items."""
        registry = PipelineRegistry()
        registry.register_stage("add_one", FunctionStage("add_one", lambda x, ctx: x + 1))
        registry.register_stage("double", FunctionStage("double", lambda x, ctx: x * 2))
        
        composed = registry.compose("add_one", "double")
        result = composed.run(5)
        
        # (5 + 1) * 2 = 12
        self.assertEqual(result, 12)
    
    def test_compose_with_pipelines(self):
        """Test composing registered pipelines."""
        registry = PipelineRegistry()
        registry.register_pipeline("prep", Pipeline([
            FunctionStage("add_one", lambda x, ctx: x + 1),
        ]))
        registry.register_pipeline("process", Pipeline([
            FunctionStage("double", lambda x, ctx: x * 2),
        ]))
        
        composed = registry.compose("prep", "process")
        result = composed.run(5)
        
        self.assertEqual(result, 12)


class TestAttachablePipeline(unittest.TestCase):
    """Tests for AttachablePipeline."""
    
    def test_attach_stage(self):
        """Test attaching a stage."""
        pipeline = AttachablePipeline()
        pipeline.attach("add_one", FunctionStage("add_one", lambda x, ctx: x + 1))
        
        result = pipeline.run(5)
        self.assertEqual(result, 6)
    
    def test_attach_at_position(self):
        """Test attaching at specific position."""
        pipeline = AttachablePipeline()
        pipeline.attach("double", FunctionStage("double", lambda x, ctx: x * 2))
        pipeline.attach("add_one", FunctionStage("add_one", lambda x, ctx: x + 1), position=0)
        
        result = pipeline.run(5)
        # (5 + 1) * 2 = 12 (add_one is first)
        self.assertEqual(result, 12)
    
    def test_detach_stage(self):
        """Test detaching a stage."""
        pipeline = AttachablePipeline()
        pipeline.attach("add_one", FunctionStage("add_one", lambda x, ctx: x + 1))
        pipeline.attach("double", FunctionStage("double", lambda x, ctx: x * 2))
        
        detached = pipeline.detach("add_one")
        self.assertIsNotNone(detached)
        
        result = pipeline.run(5)
        # Only double: 5 * 2 = 10
        self.assertEqual(result, 10)
    
    def test_attach_pipeline(self):
        """Test attaching another pipeline."""
        pipeline1 = AttachablePipeline()
        pipeline1.attach("add_one", FunctionStage("add_one", lambda x, ctx: x + 1))
        
        pipeline2 = Pipeline([
            FunctionStage("double", lambda x, ctx: x * 2),
            FunctionStage("add_ten", lambda x, ctx: x + 10),
        ])
        
        pipeline1.attach_pipeline(pipeline2)
        
        result = pipeline1.run(5)
        # (5 + 1) * 2 + 10 = 22
        self.assertEqual(result, 22)
    
    def test_replace_stage(self):
        """Test replacing a stage."""
        pipeline = AttachablePipeline()
        pipeline.attach("transform", FunctionStage("add_one", lambda x, ctx: x + 1))
        
        pipeline.replace("transform", FunctionStage("triple", lambda x, ctx: x * 3))
        
        result = pipeline.run(5)
        self.assertEqual(result, 15)
    
    def test_list_attached(self):
        """Test listing attached stages."""
        pipeline = AttachablePipeline()
        pipeline.attach("a", FunctionStage("a", lambda x, ctx: x))
        pipeline.attach("b", FunctionStage("b", lambda x, ctx: x))
        pipeline.attach("c", FunctionStage("c", lambda x, ctx: x))
        
        attached = pipeline.list_attached()
        self.assertEqual(set(attached), {"a", "b", "c"})
    
    def test_get_stage(self):
        """Test getting a stage by name."""
        pipeline = AttachablePipeline()
        stage = FunctionStage("my_stage", lambda x, ctx: x)
        pipeline.attach("my_stage", stage)
        
        retrieved = pipeline.get_stage("my_stage")
        self.assertEqual(retrieved, stage)
    
    def test_chain_operator(self):
        """Test chaining pipelines with >> operator."""
        pipeline1 = AttachablePipeline()
        pipeline1.attach("add_one", FunctionStage("add_one", lambda x, ctx: x + 1))
        
        pipeline2 = Pipeline([
            FunctionStage("double", lambda x, ctx: x * 2),
        ])
        
        combined = pipeline1 >> pipeline2
        
        result = combined.run(5)
        # (5 + 1) * 2 = 12
        self.assertEqual(result, 12)
        
        # Original should be unchanged
        self.assertEqual(pipeline1.run(5), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
