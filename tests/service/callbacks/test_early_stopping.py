"""Tests for callbacks.early_stopping module."""

from unittest.mock import MagicMock

import pytest

# Import directly from the module to avoid service/__init__.py imports
from callbacks.early_stopping import EarlyStoppingCallback


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback class."""

    def test_initializes_with_defaults(self):
        """Test that callback initializes with default values."""
        callback = EarlyStoppingCallback()

        assert callback.monitor == "mAP50"
        assert callback.patience == 10
        assert callback.min_delta == 0.001
        assert callback.mode == "max"
        assert callback.counter == 0
        assert callback.should_stop is False

    def test_initializes_with_custom_values(self):
        """Test that callback initializes with custom values."""
        stop_fn = MagicMock()
        callback = EarlyStoppingCallback(
            monitor="val_loss",
            patience=5,
            min_delta=0.01,
            mode="min",
            stop_training_fn=stop_fn,
        )

        assert callback.monitor == "val_loss"
        assert callback.patience == 5
        assert callback.min_delta == 0.01
        assert callback.mode == "min"
        assert callback.stop_training_fn is stop_fn

    def test_initializes_best_value_for_max_mode(self):
        """Test that best_value is initialized to -inf for max mode."""
        callback = EarlyStoppingCallback(mode="max")
        assert callback.best_value == float("-inf")

    def test_initializes_best_value_for_min_mode(self):
        """Test that best_value is initialized to inf for min mode."""
        callback = EarlyStoppingCallback(mode="min")
        assert callback.best_value == float("inf")

    def test_is_improvement_for_max_mode(self):
        """Test improvement detection for max mode."""
        callback = EarlyStoppingCallback(mode="max", min_delta=0.01)

        callback.best_value = 0.8
        assert callback._is_improvement(0.82) is True  # Above threshold (0.82 > 0.8 + 0.01)
        assert callback._is_improvement(0.811) is True  # Above min_delta (0.811 > 0.8 + 0.01)
        assert callback._is_improvement(0.805) is False  # Below min_delta (0.805 < 0.8 + 0.01)
        assert callback._is_improvement(0.79) is False  # Decreased

    def test_is_improvement_for_min_mode(self):
        """Test improvement detection for min mode."""
        callback = EarlyStoppingCallback(mode="min", min_delta=0.01)

        callback.best_value = 0.5
        assert callback._is_improvement(0.48) is True  # Below threshold (0.48 < 0.5 - 0.01)
        assert callback._is_improvement(0.489) is True  # Below min_delta (0.489 < 0.5 - 0.01)
        assert callback._is_improvement(0.495) is False  # Above min_delta (0.495 > 0.5 - 0.01)
        assert callback._is_improvement(0.51) is False  # Increased

    def test_on_epoch_end_warns_when_metric_missing(self, caplog):
        """Test that warning is logged when monitored metric is missing."""
        callback = EarlyStoppingCallback(monitor="missing_metric")

        callback.on_epoch_end({"epoch": 1})

        assert "missing_metric" in caplog.text
        assert "not found in metrics" in caplog.text

    def test_on_epoch_end_updates_best_value_on_improvement(self):
        """Test that best_value is updated on improvement."""
        callback = EarlyStoppingCallback(monitor="mAP50", mode="max")

        callback.on_epoch_end({"epoch": 1, "mAP50": 0.80})
        assert callback.best_value == 0.80
        assert callback.counter == 0

        callback.on_epoch_end({"epoch": 2, "mAP50": 0.85})
        assert callback.best_value == 0.85
        assert callback.counter == 0

    def test_on_epoch_end_increments_counter_on_no_improvement(self):
        """Test that counter increments when no improvement."""
        callback = EarlyStoppingCallback(monitor="mAP50", mode="max", patience=3)

        callback.on_epoch_end({"epoch": 1, "mAP50": 0.80})
        assert callback.counter == 0

        callback.on_epoch_end({"epoch": 2, "mAP50": 0.79})
        assert callback.counter == 1

        callback.on_epoch_end({"epoch": 3, "mAP50": 0.78})
        assert callback.counter == 2

    def test_on_epoch_end_triggers_stop_after_patience(self):
        """Test that training stops after patience epochs."""
        callback = EarlyStoppingCallback(monitor="mAP50", mode="max", patience=2)

        callback.on_epoch_end({"epoch": 1, "mAP50": 0.80})  # Best
        callback.on_epoch_end({"epoch": 2, "mAP50": 0.79})  # Counter = 1
        callback.on_epoch_end({"epoch": 3, "mAP50": 0.78})  # Counter = 2, stop

        assert callback.should_stop is True
        assert callback.counter == 2

    def test_on_epoch_end_calls_stop_training_fn_when_triggered(self):
        """Test that stop_training_fn is called when early stopping triggers."""
        stop_fn = MagicMock()
        callback = EarlyStoppingCallback(
            monitor="mAP50", mode="max", patience=2, stop_training_fn=stop_fn
        )

        callback.on_epoch_end({"epoch": 1, "mAP50": 0.80})
        callback.on_epoch_end({"epoch": 2, "mAP50": 0.79})
        callback.on_epoch_end({"epoch": 3, "mAP50": 0.78})

        stop_fn.assert_called_once()

    def test_on_epoch_end_resets_counter_on_improvement(self):
        """Test that counter resets when improvement occurs."""
        callback = EarlyStoppingCallback(monitor="mAP50", mode="max", patience=5)

        callback.on_epoch_end({"epoch": 1, "mAP50": 0.80})  # Best
        callback.on_epoch_end({"epoch": 2, "mAP50": 0.79})  # Counter = 1
        callback.on_epoch_end({"epoch": 3, "mAP50": 0.78})  # Counter = 2
        callback.on_epoch_end({"epoch": 4, "mAP50": 0.81})  # Improvement! Reset
        callback.on_epoch_end({"epoch": 5, "mAP50": 0.80})  # Counter = 1

        assert callback.counter == 1
        assert callback.should_stop is False

    def test_min_delta_respects_threshold(self):
        """Test that min_delta is respected for improvement detection."""
        callback = EarlyStoppingCallback(monitor="mAP50", mode="max", min_delta=0.01)

        callback.on_epoch_end({"epoch": 1, "mAP50": 0.80})
        callback.on_epoch_end({"epoch": 2, "mAP50": 0.805})  # Only 0.005 improvement

        assert callback.counter == 1  # Should count as no improvement

    def test_works_for_minimizing_metrics(self):
        """Test that callback works correctly for minimizing metrics."""
        callback = EarlyStoppingCallback(monitor="val_loss", mode="min", patience=2)

        callback.on_epoch_end({"epoch": 1, "val_loss": 1.0})  # Best
        callback.on_epoch_end({"epoch": 2, "val_loss": 1.1})  # Counter = 1
        callback.on_epoch_end({"epoch": 3, "val_loss": 1.2})  # Counter = 2, stop

        assert callback.should_stop is True
        assert callback.best_value == 1.0

    def test_works_for_maximizing_metrics(self):
        """Test that callback works correctly for maximizing metrics."""
        callback = EarlyStoppingCallback(monitor="mAP50", mode="max", patience=2)

        callback.on_epoch_end({"epoch": 1, "mAP50": 0.80})  # Best
        callback.on_epoch_end({"epoch": 2, "mAP50": 0.79})  # Counter = 1
        callback.on_epoch_end({"epoch": 3, "mAP50": 0.78})  # Counter = 2, stop

        assert callback.should_stop is True
        assert callback.best_value == 0.80
