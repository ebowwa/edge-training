"""Tests for callbacks.manager module."""

from unittest.mock import MagicMock

import pytest

# Import directly from the module to avoid service/__init__.py imports
from callbacks.base import TrainingCallback
from callbacks.manager import CallbackManager


class MockCallback(TrainingCallback):
    """Mock callback for testing."""

    def __init__(self):
        self.train_start_calls = []
        self.train_end_calls = []
        self.epoch_start_calls = []
        self.epoch_end_calls = []
        self.batch_end_calls = []

    def on_train_start(self, metrics):
        self.train_start_calls.append(metrics.copy())

    def on_train_end(self, metrics):
        self.train_end_calls.append(metrics.copy())

    def on_epoch_start(self, metrics):
        self.epoch_start_calls.append(metrics.copy())

    def on_epoch_end(self, metrics):
        self.epoch_end_calls.append(metrics.copy())

    def on_batch_end(self, metrics):
        self.batch_end_calls.append(metrics.copy())


class TestCallbackManager:
    """Tests for CallbackManager class."""

    def test_manager_initializes_empty(self):
        """Test that CallbackManager initializes with empty callback list."""
        manager = CallbackManager()
        assert manager.callbacks == []

    def test_add_callback(self):
        """Test adding a callback."""
        manager = CallbackManager()
        callback = MockCallback()

        result = manager.add(callback)

        assert callback in manager.callbacks
        assert result is manager  # Returns self for chaining

    def test_add_multiple_callbacks(self):
        """Test adding multiple callbacks."""
        manager = CallbackManager()
        cb1 = MockCallback()
        cb2 = MockCallback()

        manager.add(cb1).add(cb2)

        assert len(manager.callbacks) == 2
        assert cb1 in manager.callbacks
        assert cb2 in manager.callbacks

    def test_remove_callback(self):
        """Test removing a callback."""
        manager = CallbackManager()
        callback = MockCallback()
        manager.add(callback)

        manager.remove(callback)

        assert callback not in manager.callbacks

    def test_on_train_start_propagates_to_all_callbacks(self):
        """Test that on_train_start calls all registered callbacks."""
        manager = CallbackManager()
        cb1 = MockCallback()
        cb2 = MockCallback()
        manager.add(cb1).add(cb2)

        metrics = {"epoch": 0}
        manager.on_train_start(metrics)

        assert cb1.train_start_calls == [metrics]
        assert cb2.train_start_calls == [metrics]

    def test_on_train_end_propagates_to_all_callbacks(self):
        """Test that on_train_end calls all registered callbacks."""
        manager = CallbackManager()
        cb1 = MockCallback()
        cb2 = MockCallback()
        manager.add(cb1).add(cb2)

        metrics = {"epoch": 10, "loss": 0.3}
        manager.on_train_end(metrics)

        assert cb1.train_end_calls == [metrics]
        assert cb2.train_end_calls == [metrics]

    def test_on_epoch_start_propagates_to_all_callbacks(self):
        """Test that on_epoch_start calls all registered callbacks."""
        manager = CallbackManager()
        cb1 = MockCallback()
        cb2 = MockCallback()
        manager.add(cb1).add(cb2)

        metrics = {"epoch": 1}
        manager.on_epoch_start(metrics)

        assert cb1.epoch_start_calls == [metrics]
        assert cb2.epoch_start_calls == [metrics]

    def test_on_epoch_end_propagates_to_all_callbacks(self):
        """Test that on_epoch_end calls all registered callbacks."""
        manager = CallbackManager()
        cb1 = MockCallback()
        cb2 = MockCallback()
        manager.add(cb1).add(cb2)

        metrics = {"epoch": 1, "loss": 0.5, "mAP50": 0.85}
        manager.on_epoch_end(metrics)

        assert cb1.epoch_end_calls == [metrics]
        assert cb2.epoch_end_calls == [metrics]

    def test_on_batch_end_propagates_to_all_callbacks(self):
        """Test that on_batch_end calls all registered callbacks."""
        manager = CallbackManager()
        cb1 = MockCallback()
        cb2 = MockCallback()
        manager.add(cb1).add(cb2)

        metrics = {"epoch": 1, "batch": 32, "loss": 0.6}
        manager.on_batch_end(metrics)

        assert cb1.batch_end_calls == [metrics]
        assert cb2.batch_end_calls == [metrics]

    def test_manager_handles_empty_callback_list(self):
        """Test that manager handles no callbacks gracefully."""
        manager = CallbackManager()

        # Should not raise any errors
        manager.on_train_start({})
        manager.on_train_end({})
        manager.on_epoch_start({})
        manager.on_epoch_end({})
        manager.on_batch_end({})

    def test_multiple_hooks_called_in_order(self):
        """Test that hooks are called in the order callbacks were added."""
        manager = CallbackManager()
        cb1 = MockCallback()
        cb2 = MockCallback()
        manager.add(cb1).add(cb2)

        for epoch in range(3):
            manager.on_epoch_end({"epoch": epoch})

        assert len(cb1.epoch_end_calls) == 3
        assert len(cb2.epoch_end_calls) == 3
        assert cb1.epoch_end_calls == cb2.epoch_end_calls
