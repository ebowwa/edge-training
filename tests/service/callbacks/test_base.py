"""Tests for callbacks.base module."""

import pytest

# Import directly from the module to avoid service/__init__.py imports
from callbacks.base import TrainingCallback


class MockCallback(TrainingCallback):
    """Mock callback for testing."""

    def __init__(self):
        self.train_start_called = False
        self.train_end_called = False
        self.epoch_start_called = False
        self.epoch_end_called = False
        self.batch_end_called = False

    def on_train_start(self, metrics):
        self.train_start_called = True

    def on_train_end(self, metrics):
        self.train_end_called = True

    def on_epoch_start(self, metrics):
        self.epoch_start_called = True

    def on_epoch_end(self, metrics):
        self.epoch_end_called = True

    def on_batch_end(self, metrics):
        self.batch_end_called = True


class TestTrainingCallback:
    """Tests for TrainingCallback base class."""

    def test_base_callback_has_all_hook_methods(self):
        """Test that TrainingCallback defines all hook methods."""
        callback = TrainingCallback()

        assert hasattr(callback, "on_train_start")
        assert hasattr(callback, "on_train_end")
        assert hasattr(callback, "on_epoch_start")
        assert hasattr(callback, "on_epoch_end")
        assert hasattr(callback, "on_batch_end")

    def test_base_hook_methods_are_noop(self):
        """Test that base hook methods do nothing (are no-ops)."""
        callback = TrainingCallback()
        metrics = {"epoch": 1, "loss": 0.5}

        # Should not raise any errors
        callback.on_train_start(metrics)
        callback.on_train_end(metrics)
        callback.on_epoch_start(metrics)
        callback.on_epoch_end(metrics)
        callback.on_batch_end(metrics)

    def test_mock_callback_hooks_are_callable(self):
        """Test that mock callback hooks are called."""
        callback = MockCallback()
        metrics = {"epoch": 1, "loss": 0.5}

        assert not callback.train_start_called
        assert not callback.epoch_end_called

        callback.on_train_start(metrics)
        assert callback.train_start_called

        callback.on_epoch_end(metrics)
        assert callback.epoch_end_called

    def test_custom_callback_can_be_subclassed(self):
        """Test that TrainingCallback can be subclassed."""
        class CustomCallback(TrainingCallback):
            def __init__(self):
                self.custom_value = 42

            def on_epoch_end(self, metrics):
                return metrics.get("epoch", 0) * self.custom_value

        callback = CustomCallback()
        assert callback.custom_value == 42
        assert callback.on_epoch_end({"epoch": 2}) == 84
