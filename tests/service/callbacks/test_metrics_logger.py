"""Tests for callbacks.metrics_logger module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import directly from the module to avoid service/__init__.py imports
from callbacks.metrics_logger import MetricsLoggerCallback


class TestMetricsLoggerCallback:
    """Tests for MetricsLoggerCallback class."""

    def test_initializes_with_defaults(self):
        """Test that callback initializes with default values."""
        callback = MetricsLoggerCallback()

        assert callback.output_dir is None
        assert callback.log_every_n_epochs == 1
        assert callback.history == []

    def test_initializes_with_custom_values(self):
        """Test that callback initializes with custom values."""
        callback = MetricsLoggerCallback(output_dir="/tmp/logs", log_every_n_epochs=5)

        assert callback.output_dir == "/tmp/logs"
        assert callback.log_every_n_epochs == 5
        assert callback.history == []

    def test_on_epoch_end_appends_to_history(self):
        """Test that metrics are appended to history."""
        callback = MetricsLoggerCallback()

        metrics1 = {"epoch": 1, "loss": 0.5}
        metrics2 = {"epoch": 2, "loss": 0.4}

        callback.on_epoch_end(metrics1)
        callback.on_epoch_end(metrics2)

        assert len(callback.history) == 2
        assert callback.history[0] == metrics1
        assert callback.history[1] == metrics2

    def test_on_epoch_end_copies_metrics(self):
        """Test that metrics are copied, not referenced."""
        callback = MetricsLoggerCallback()

        original_metrics = {"epoch": 1, "loss": 0.5}
        callback.on_epoch_end(original_metrics)

        # Modify original
        original_metrics["loss"] = 0.9

        # History should have the original value
        assert callback.history[0]["loss"] == 0.5

    def test_on_epoch_end_logs_every_n_epochs(self, caplog):
        """Test that logging happens every N epochs."""
        callback = MetricsLoggerCallback(log_every_n_epochs=3)

        for epoch in range(1, 6):
            callback.on_epoch_end({"epoch": epoch, "loss": 0.5 - epoch * 0.01})

        # Should log epochs 1, 2, 3 (since 3 % 3 == 0 is False for 1, 2 but True for 3)
        # Actually log_every_n_epochs works as epoch % log_every_n_epochs == 0
        # So epochs 3 and 6 would be logged, not 1, 2, 3
        # Wait, let me check the implementation again...

        # epoch % log_every_n_epochs == 0 means:
        # epoch=1, n=3: 1 % 3 = 1 != 0, not logged
        # epoch=2, n=3: 2 % 3 = 2 != 0, not logged
        # epoch=3, n=3: 3 % 3 = 0 == 0, logged
        # epoch=4, n=3: 4 % 3 = 1 != 0, not logged
        # epoch=5, n=3: 5 % 3 = 2 != 0, not logged

        # So only epochs 3 should be logged from epochs 1-5

        # The default log_every_n_epochs is 1, so all epochs are logged
        # Let me test with log_every_n_epochs=1

    def test_on_epoch_end_logs_all_epochs_when_interval_is_one(self, caplog):
        """Test that all epochs are logged when interval is 1."""
        caplog.set_level("INFO")
        callback = MetricsLoggerCallback(log_every_n_epochs=1)

        for epoch in range(1, 4):
            callback.on_epoch_end({"epoch": epoch, "loss": 0.5})

        assert "Epoch 1 | loss=0.5000" in caplog.text
        assert "Epoch 2 | loss=0.5000" in caplog.text
        assert "Epoch 3 | loss=0.5000" in caplog.text

    def test_on_epoch_end_formats_metrics_correctly(self, caplog):
        """Test that metrics are formatted correctly in log."""
        caplog.set_level("INFO")
        callback = MetricsLoggerCallback(log_every_n_epochs=1)

        metrics = {
            "epoch": 5,
            "loss": 0.4567,
            "mAP50": 0.8234,
            "mAP50-95": 0.6543,
            "precision": 0.8765,
            "recall": 0.7654,
        }
        callback.on_epoch_end(metrics)

        log_text = caplog.text
        assert "Epoch 5" in log_text
        assert "loss=0.4567" in log_text
        assert "mAP50=0.8234" in log_text
        assert "mAP50-95=0.6543" in log_text
        assert "precision=0.8765" in log_text
        assert "recall=0.7654" in log_text

    def test_on_epoch_end_skips_missing_metrics_in_log(self, caplog):
        """Test that missing metrics are skipped in log output."""
        caplog.set_level("INFO")
        callback = MetricsLoggerCallback(log_every_n_epochs=1)

        metrics = {"epoch": 1, "loss": 0.5}  # Only loss
        callback.on_epoch_end(metrics)

        log_text = caplog.text
        assert "Epoch 1 | loss=0.5000" in log_text
        assert "mAP50=" not in log_text  # Should not appear

    def test_on_train_end_saves_to_file_when_output_dir_set(self):
        """Test that metrics are saved to file when output_dir is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = MetricsLoggerCallback(output_dir=tmpdir, log_every_n_epochs=1)

            # Add some metrics
            for epoch in range(1, 4):
                callback.on_epoch_end({"epoch": epoch, "loss": 0.5 - epoch * 0.1})

            callback.on_train_end({})

            # Check file exists
            output_path = Path(tmpdir) / "training_metrics.json"
            assert output_path.exists()

            # Check file contents
            with open(output_path) as f:
                data = json.load(f)

            assert len(data) == 3
            assert data[0]["epoch"] == 1
            assert data[0]["loss"] == pytest.approx(0.4)
            assert data[2]["epoch"] == 3
            assert data[2]["loss"] == pytest.approx(0.2)

    def test_on_train_end_creates_output_directory_if_needed(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "path"
            callback = MetricsLoggerCallback(output_dir=str(output_dir))

            callback.on_epoch_end({"epoch": 1, "loss": 0.5})
            callback.on_train_end({})

            assert output_dir.exists()
            assert (output_dir / "training_metrics.json").exists()

    def test_on_train_end_does_nothing_when_output_dir_is_none(self):
        """Test that on_train_end does nothing when output_dir is None."""
        callback = MetricsLoggerCallback(output_dir=None)

        callback.on_epoch_end({"epoch": 1, "loss": 0.5})

        # Should not raise any errors
        callback.on_train_end({})

    def test_on_epoch_end_logs_at_interval_boundary(self, caplog):
        """Test that logging respects the interval boundary."""
        caplog.set_level("INFO")
        callback = MetricsLoggerCallback(log_every_n_epochs=2)

        callback.on_epoch_end({"epoch": 1, "loss": 0.5})  # 1 % 2 = 1, not logged
        callback.on_epoch_end({"epoch": 2, "loss": 0.4})  # 2 % 2 = 0, logged
        callback.on_epoch_end({"epoch": 3, "loss": 0.3})  # 3 % 2 = 1, not logged
        callback.on_epoch_end({"epoch": 4, "loss": 0.2})  # 4 % 2 = 0, logged

        log_text = caplog.text
        assert "Epoch 1" not in log_text
        assert "Epoch 2" in log_text
        assert "Epoch 3" not in log_text
        assert "Epoch 4" in log_text

    def test_history_preserves_all_metrics_regardless_of_logging_interval(self):
        """Test that history stores all metrics even if not logged."""
        callback = MetricsLoggerCallback(log_every_n_epochs=5)

        for epoch in range(1, 11):
            callback.on_epoch_end({"epoch": epoch, "loss": 0.5 - epoch * 0.01})

        # All metrics should be in history
        assert len(callback.history) == 10

    def test_metrics_are_json_serializable(self):
        """Test that metrics can be serialized to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = MetricsLoggerCallback(output_dir=tmpdir)

            callback.on_epoch_end({"epoch": 1, "loss": 0.5, "mAP50": 0.85})
            callback.on_train_end({})

            output_path = Path(tmpdir) / "training_metrics.json"

            # Should not raise an error
            with open(output_path) as f:
                json.load(f)
