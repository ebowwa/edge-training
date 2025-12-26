"""Tests for callbacks.tensorboard module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import directly from the module to avoid service/__init__.py imports
from callbacks.tensorboard import TensorBoardCallback, SummaryWriter


class TestTensorBoardCallback:
    """Tests for TensorBoardCallback class."""

    def test_initializes_with_log_dir(self):
        """Test that callback initializes with log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)
            assert callback.log_dir == tmpdir

    @patch("callbacks.tensorboard.SummaryWriter", None)
    def test_handles_missing_tensorboard(self, caplog):
        """Test that callback handles missing TensorBoard gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("callbacks.tensorboard.SummaryWriter", None):
                callback = TensorBoardCallback(tmpdir)
                assert callback.writer is None

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_creates_summary_writer_when_available(self, mock_writer):
        """Test that callback creates SummaryWriter when available."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)
            assert callback.writer is mock_instance
            mock_writer.assert_called_once_with(log_dir=tmpdir)

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_on_epoch_end_logs_scalar_metrics(self, mock_writer):
        """Test that on_epoch_end logs scalar metrics."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)

            metrics = {
                "epoch": 5,
                "loss": 0.45,
                "train_loss": 0.50,
                "val_loss": 0.40,
                "box_loss": 0.20,
                "cls_loss": 0.25,
                "mAP50": 0.82,
                "mAP50-95": 0.65,
                "precision": 0.88,
                "recall": 0.75,
                "lr": 0.001,
            }
            callback.on_epoch_end(metrics)

            # Check that add_scalar was called for each metric
            # 5 loss metrics + 4 detection metrics + 1 learning rate = 10 total
            assert mock_instance.add_scalar.call_count == 10
            mock_instance.flush.assert_called_once()

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_on_epoch_end_logs_loss_metrics(self, mock_writer):
        """Test that loss metrics are logged under Loss/ prefix."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)

            metrics = {"epoch": 1, "loss": 0.5, "train_loss": 0.6, "val_loss": 0.4}
            callback.on_epoch_end(metrics)

            calls = mock_instance.add_scalar.call_args_list
            assert any(c[0][0] == "Loss/loss" for c in calls)
            assert any(c[0][0] == "Loss/train_loss" for c in calls)
            assert any(c[0][0] == "Loss/val_loss" for c in calls)

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_on_epoch_end_logs_metrics_under_metrics_prefix(self, mock_writer):
        """Test that metrics are logged under Metrics/ prefix."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)

            metrics = {"epoch": 1, "mAP50": 0.85, "mAP50-95": 0.70, "precision": 0.90, "recall": 0.80}
            callback.on_epoch_end(metrics)

            calls = mock_instance.add_scalar.call_args_list
            assert any(c[0][0] == "Metrics/mAP50" for c in calls)
            assert any(c[0][0] == "Metrics/mAP50-95" for c in calls)
            assert any(c[0][0] == "Metrics/precision" for c in calls)
            assert any(c[0][0] == "Metrics/recall" for c in calls)

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_on_epoch_end_logs_learning_rate(self, mock_writer):
        """Test that learning rate is logged under Training/ prefix."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)

            metrics = {"epoch": 1, "lr": 0.001}
            callback.on_epoch_end(metrics)

            mock_instance.add_scalar.assert_called_with("Training/learning_rate", 0.001, 1)

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_on_epoch_end_skips_missing_metrics(self, mock_writer):
        """Test that missing metrics are skipped without error."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)

            # Only provide a subset of metrics
            metrics = {"epoch": 1, "loss": 0.5}
            callback.on_epoch_end(metrics)

            # Should only log loss, not fail on missing metrics
            assert mock_instance.add_scalar.call_count == 1

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_on_epoch_end_defaults_epoch_to_zero(self, mock_writer):
        """Test that epoch defaults to 0 when not in metrics."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)

            metrics = {"loss": 0.5}
            callback.on_epoch_end(metrics)

            # Check that epoch 0 was used
            call_args = mock_instance.add_scalar.call_args_list[0]
            assert call_args[0][2] == 0

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_on_train_end_closes_writer(self, mock_writer):
        """Test that on_train_end closes the writer."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)
            callback.on_train_end({})

            mock_instance.close.assert_called_once()

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_on_train_end_handles_none_writer(self, mock_writer):
        """Test that on_train_end handles None writer gracefully."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)
            callback.writer = None
            callback.on_train_end({})

            # Should not raise any errors

    @patch("callbacks.tensorboard.SummaryWriter")
    def test_on_epoch_end_does_nothing_when_writer_is_none(self, mock_writer):
        """Test that on_epoch_end does nothing when writer is None."""
        mock_instance = MagicMock()
        mock_writer.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(tmpdir)
            callback.writer = None

            callback.on_epoch_end({"epoch": 1, "loss": 0.5})

            # add_scalar should not be called
            mock_instance.add_scalar.assert_not_called()
